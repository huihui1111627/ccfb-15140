"""Service object-lifetime distributions.

A *lifetime model* maps an allocated object to the wall-clock time at which it
would become unreachable (its `death_time`), independent of how often the
collector happens to look at it.  This is the standard semantics used when
people say "object lifetime distribution": the object does not die because a
minor GC happened, it dies when its last reference goes away.  Whether the
collector notices before or after it has been promoted is exactly what we are
studying.

Every service traffic class therefore exposes:

    sample_lifetime(rng, alloc_time_s) -> float   # absolute death time (s)
    size_bytes (mean size, used for occupancy)

Four models are provided, because the acceptance criteria require evaluating
services with quite different lifetime shapes:

* ``RequestLifetime`` -- short-lived request/response objects.  Lognormal
  lifetime with a median in the tens of milliseconds and a small fraction of
  long outliers (slow calls, retries).
* ``CacheLifetime`` -- long-lived cache entries.  Pareto tail with alpha
  slightly above 1 (or a Weibull with a long tail), so a few entries are
  effectively immortal.
* ``SessionLifetime`` -- medium-lived session/context objects: lognormal with
  a median in the seconds-to-tens-of-seconds range, plus a heavy right tail.
  These are the objects that survive a few minor GCs, get promoted, and then
  die in the old generation -- the classic premature-promotion problem.
* ``BimodalLifetime`` -- an arbitrary mixture; used by the "mixed" service.

The distribution objects also know how to report their survival curve
S(t) = P(lifetime > t) by Monte Carlo sampling, which the age-distribution
plots and the calibration analysis use.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .rng import pareto

HOUR = 3600.0


class LifetimeModel:
    """Abstract lifetime model.  All times are wall-clock seconds."""

    name: str = "abstract"

    def sample_lifetime(self, r: random.Random, alloc_time_s: float) -> float:
        raise NotImplementedError

    def survival(self, t: float) -> float:
        """P(lifetime > t) in closed form when available."""
        raise NotImplementedError

    # Vectorised batch samplers.  Models that do not override this fall back
    # to a Python loop, which is correct but slow; the four service-relevant
    # models all override it.
    def sample_lifetimes_np(self, rng: np.random.Generator, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float64)
        py = random.Random(int(rng.integers(0, 2 ** 63 - 1)))
        for i in range(n):
            out[i] = self.sample_lifetime(py, 0.0)
        return out

    @staticmethod
    def _lognormal_batch(rng: np.random.Generator, n: int, med_s: float,
                         sigma: float) -> np.ndarray:
        z = rng.standard_normal(n)
        return np.exp(math.log(med_s) + sigma * z)

    def describe(self) -> str:
        return self.name


@dataclass
class RequestLifetime(LifetimeModel):
    """Short-lived request objects.

    Lognormal body with median ``median_ms`` and a ``long_tail_frac`` share of
    objects drawn from a much longer lognormal (slow RPCs, retry buffers).
    """

    median_ms: float = 8.0
    sigma: float = 0.9
    long_tail_frac: float = 0.03
    long_median_ms: float = 700.0
    long_sigma: float = 1.1
    name: str = "request"

    def sample_lifetime(self, r: random.Random, alloc_time_s: float) -> float:
        use_long = r.random() < self.long_tail_frac
        med, sig = (
            (self.long_median_ms, self.long_sigma)
            if use_long
            else (self.median_ms, self.sigma)
        )
        mu = math.log(med / 1000.0)
        # lognormal draw
        return math.exp(r.gauss(mu, sig))

    def survival(self, t: float) -> float:
        if t <= 0:
            return 1.0
        mu = math.log(self.median_ms / 1000.0)
        return 0.5 * math.erfc((math.log(t) - mu) / (self.sigma * math.sqrt(2.0)))

    def sample_lifetimes_np(self, rng: np.random.Generator, n: int) -> np.ndarray:
        long = rng.random(n) < self.long_tail_frac
        z = rng.standard_normal(n)
        body = np.exp(math.log(self.median_ms / 1000.0) + self.sigma * z)
        z2 = rng.standard_normal(n)
        tail = np.exp(math.log(self.long_median_ms / 1000.0) + self.long_sigma * z2)
        return np.where(long, tail, body)

    def describe(self) -> str:
        return (
            f"request(median={self.median_ms:g}ms,sigma={self.sigma:g},"
            f"long={self.long_tail_frac:.1%}@{self.long_median_ms:g}ms)"
        )


@dataclass
class CacheLifetime(LifetimeModel):
    """Long-lived cache entries.

    Pareto(xm, alpha) lifetime in seconds.  ``alpha`` close to 1 gives a very
    heavy tail; ``alpha <= 2`` means infinite variance, which is realistic for
    cache-entry popularity.  A hard ``max_lifetime_s`` gives a physical bound
    (TTL / refresh).
    """

    min_lifetime_s: float = 30.0
    alpha: float = 1.25
    max_lifetime_s: float = 12 * HOUR
    name: str = "cache"

    def sample_lifetime(self, r: random.Random, alloc_time_s: float) -> float:
        x = pareto(r, self.min_lifetime_s, self.alpha)
        return min(x, self.max_lifetime_s)

    def survival(self, t: float) -> float:
        if t <= self.min_lifetime_s:
            return 1.0
        if t >= self.max_lifetime_s:
            return 0.0
        return (self.min_lifetime_s / t) ** self.alpha

    def sample_lifetimes_np(self, rng: np.random.Generator, n: int) -> np.ndarray:
        u = rng.random(n)
        x = self.min_lifetime_s / np.power(u, 1.0 / self.alpha)
        return np.minimum(x, self.max_lifetime_s)

    def describe(self) -> str:
        return (
            f"cache(pareto xm={self.min_lifetime_s:g}s,alpha={self.alpha:g},"
            f"max={self.max_lifetime_s:g}s)"
        )


@dataclass
class SessionLifetime(LifetimeModel):
    """Medium-lived session objects.

    Lognormal with ``median_s`` and sigma; an optional fraction of sessions is
    drawn from a second, longer lognormal (long-lived sessions).  With a median
    of ~4 s these objects survive roughly N minor GCs depending on young-gen
    size and allocation rate, which is precisely the premature-promotion
    cliff.
    """

    median_s: float = 4.0
    sigma: float = 1.1
    long_frac: float = 0.10
    long_median_s: float = 120.0
    long_sigma: float = 1.3
    name: str = "session"

    def sample_lifetime(self, r: random.Random, alloc_time_s: float) -> float:
        if r.random() < self.long_frac:
            mu = math.log(self.long_median_s)
            sig = self.long_sigma
        else:
            mu = math.log(self.median_s)
            sig = self.sigma
        return math.exp(r.gauss(mu, sig))

    def survival(self, t: float) -> float:
        if t <= 0:
            return 1.0

        def lognormal_surv(med, sig, x):
            return 0.5 * math.erfc(
                (math.log(x) - math.log(med)) / (sig * math.sqrt(2.0))
            )

        return (1 - self.long_frac) * lognormal_surv(
            self.median_s, self.sigma, t
        ) + self.long_frac * lognormal_surv(self.long_median_s, self.long_sigma, t)

    def sample_lifetimes_np(self, rng: np.random.Generator, n: int) -> np.ndarray:
        long = rng.random(n) < self.long_frac
        z = rng.standard_normal(n)
        body = np.exp(math.log(self.median_s) + self.sigma * z)
        z2 = rng.standard_normal(n)
        tail = np.exp(math.log(self.long_median_s) + self.long_sigma * z2)
        return np.where(long, tail, body)

    def describe(self) -> str:
        return (
            f"session(median={self.median_s:g}s,sigma={self.sigma:g},"
            f"long={self.long_frac:.1%}@{self.long_median_s:g}s)"
        )


@dataclass
class DeterministicLifetime(LifetimeModel):
    """Fixed lifetime with small jitter; mainly for tests and calibration."""

    lifetime_s: float = 1.0
    jitter: float = 0.0
    name: str = "deterministic"

    def sample_lifetime(self, r: random.Random, alloc_time_s: float) -> float:
        return max(0.0, self.lifetime_s * (1.0 + r.uniform(-self.jitter, self.jitter)))

    def survival(self, t: float) -> float:
        return 1.0 if t < self.lifetime_s * (1 - self.jitter) else 0.0


@dataclass
class BimodalLifetime(LifetimeModel):
    """Mixture of models with explicit weights."""

    components: Sequence[Tuple[float, LifetimeModel]] = field(default_factory=tuple)
    name: str = "mixture"

    def sample_lifetime(self, r: random.Random, alloc_time_s: float) -> float:
        u = r.random()
        acc = 0.0
        for w, m in self.components:
            acc += w
            if u < acc:
                return m.sample_lifetime(r, alloc_time_s)
        return self.components[-1][1].sample_lifetime(r, alloc_time_s)

    def survival(self, t: float) -> float:
        return sum(w * m.survival(t) for w, m in self.components)

    def describe(self) -> str:
        return "mixture[" + ",".join(
            f"{w:.2f}*{m.describe()}" for w, m in self.components
        ) + "]"


@dataclass
class TrafficClass:
    """A class of allocations with a lifetime model and a size.

    ``share`` is the fraction of *allocations* (not bytes) that belongs to this
    class; the service normalises shares.  ``mean_size`` is in bytes and an
    optional ``size_sigma`` makes sizes lognormal.
    """

    name: str
    lifetime: LifetimeModel
    share: float
    mean_size: int = 128
    size_sigma: float = 0.35
    static_promote: bool = False  # e.g. long-lived metadata / interned objects

    def sample_size(self, r: random.Random) -> int:
        if self.size_sigma <= 0:
            return max(8, int(self.mean_size))
        mu = math.log(self.mean_size) - 0.5 * self.size_sigma ** 2
        return max(8, int(round(math.exp(r.gauss(mu, self.size_sigma)))))

    def describe(self) -> str:
        return f"{self.name}:{self.lifetime.describe()} size={self.mean_size}B share={self.share:.2f}"


def weighted_sizes(classes: Sequence[TrafficClass]) -> Tuple[List[float], List[int]]:
    """Normalise shares and compute mean allocated bytes per allocation."""
    total = sum(c.share for c in classes)
    if total <= 0:
        raise ValueError("traffic class shares must sum to > 0")
    shares = [c.share / total for c in classes]
    mean_bytes = sum(s * c.mean_size for s, c in zip(shares, classes))
    return shares, mean_bytes
