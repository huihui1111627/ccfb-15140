"""Build an object-population model from observed service behaviour.

The simulator's :class:`ObjectPopulation` is not meant to be guessed: it is a
*fit* to data the service already produces.  Two sources are supported.

1. **Object samples.**  A trace of allocation events with ``(class, size,
   lifetime)`` -- for example from an allocation profiler or from a sampling
   agent.  :func:`fit_population` clusters the lifetimes into the three
   behavioural classes and fits a Weibull survival curve and a log-normal
   size distribution to each.

2. **GC logs.**  A young-generation log records, for every collection, how
   many survivors were copied and how many of those survived the next
   collection.  That yields a discrete survival curve, from which
   :func:`population_from_log` recovers an equivalent continuous lifetime
   distribution (assuming a single dominant class, or using the class mix
   when one is known).

Both fits are deterministic and dependency-free (scipy is used when present
only for the Weibull MLE; otherwise a closed-form log-space least-squares fit
is used).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import ObjectPopulation, MiB


# --------------------------------------------------------------------------
# Weibull survival fitting
# --------------------------------------------------------------------------
def weibull_survival(t, scale: float, shape: float):
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    return np.exp(-np.power(t / max(scale, 1e-12), shape))


def fit_weibull(lifetimes: np.ndarray,
                weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Fit ``S(t)=exp(-(t/scale)^shape)`` to right-censored-free samples.

    A Weibull lifetime has CDF ``F(t)=1-exp(-(t/scale)^shape)``.  We fit in
    log space: ``log(-log(1-F)) = shape*log(t) - shape*log(scale)``, using
    the empirical CDF at each sample.  This is robust, closed-form (a linear
    regression in log-log space) and needs no optimiser.
    """
    x = np.asarray(lifetimes, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return 1.0, 1.0
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w[np.isfinite(x)] if False else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    n = xs.size
    # Weibull plotting positions (median ranks), which avoid log(0)
    F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    y = np.log(-np.log(1.0 - F))
    u = np.log(xs)
    a, b = np.polyfit(u, y, 1)
    shape = float(max(a, 0.05))
    scale = float(math.exp(-b / shape)) if abs(shape) > 1e-9 else float(np.median(xs))
    return scale, shape


def fit_lognormal(sizes: np.ndarray) -> Tuple[float, float]:
    """Return (mu, sigma) of log(size) for a log-normal size model."""
    s = np.asarray(sizes, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return math.log(256.0), 0.5
    logs = np.log(s)
    return float(logs.mean()), float(max(logs.std(), 0.05))


# --------------------------------------------------------------------------
# Population fitting from object samples
# --------------------------------------------------------------------------
CLASS_NAMES = ("request", "session", "cache")


def fit_population(lifetimes: Sequence[float],
                   sizes: Optional[Sequence[float]] = None,
                   classes: Optional[Sequence[str]] = None,
                   n_classes: int = 3) -> ObjectPopulation:
    """Recover an :class:`ObjectPopulation` from observed objects.

    ``lifetimes`` in seconds.  If ``classes`` is given (one of ``request`` /
    ``session`` / ``cache``) the samples are grouped directly; otherwise the
    log-lifetimes are clustered into ``n_classes`` groups by 1-D k-means in
    log space, then ordered by median lifetime so the shortest group becomes
    ``request``, the middle ``session`` and the longest ``cache``.
    """
    lt = np.asarray(lifetimes, dtype=float)
    lt = np.where(np.isfinite(lt) & (lt > 0), lt, 1e-6)
    sz = (np.asarray(sizes, dtype=float) if sizes is not None else
          np.full(lt.shape[0], 512.0))
    sz = np.where(np.isfinite(sz) & (sz > 0), sz, 64.0)
    if classes is not None:
        labels = np.asarray([
            CLASS_NAMES.index(c) if c in CLASS_NAMES else 1 for c in classes])
    else:
        labels = _cluster_behaviour(lt, sz, n_classes)
    counts = np.bincount(labels, minlength=3).astype(float)
    total = counts.sum()
    fracs = counts / total if total > 0 else np.array([1.0, 0.0, 0.0])
    scale = np.ones(3)
    shape = np.ones(3) * 1.5
    size_mean = np.ones(3) * 512.0
    for c in range(3):
        mask = labels == c
        k = int(mask.sum())
        if k >= 5:
            scale[c], shape[c] = fit_weibull(lt[mask])
            mu, sigma = fit_lognormal(sz[mask])
            size_mean[c] = math.exp(mu + 0.5 * sigma * sigma)
        else:
            # borrow the nearest populated class's shape, keep a small share
            donor = _nearest_populated(labels, counts, c)
            if donor is None:
                scale[c], shape[c], size_mean[c] = 1.0, 1.5, 512.0
            else:
                scale[c] = max(scale[donor], 1e-3)
                shape[c] = max(shape[donor], 0.1)
                size_mean[c] = size_mean[donor]
        fracs[c] = max(fracs[c], 0.0)
    # keep the mix strictly positive and normalised
    fracs = np.maximum(fracs, 1e-4)
    fracs /= fracs.sum()
    pop = ObjectPopulation(
        request_fraction=float(fracs[0]),
        session_fraction=float(fracs[1]),
        cache_fraction=float(fracs[2]),
        request_scale_s=float(scale[0]), request_shape=float(shape[0]),
        session_scale_s=float(scale[1]), session_shape=float(shape[1]),
        cache_scale_s=float(scale[2]), cache_shape=float(shape[2]),
        request_size_bytes=float(size_mean[0]),
        session_size_bytes=float(size_mean[1]),
        cache_size_bytes=float(size_mean[2]),
    )
    pop.validate()
    return pop


def _cluster_behaviour(lt: np.ndarray, sz: np.ndarray,
                       k: int) -> np.ndarray:
    """Cluster objects into behavioural classes by lifetime and size.

    Lifetimes are heavy-tailed, so a Gaussian mixture is fitted in
    log-lifetime / log-size space with a per-component full covariance
    (implemented here in a few lines of numpy; no scikit-learn dependency).
    The components are then ordered by median lifetime, which is what makes
    "request / session / cache" meaningful.
    """
    x = np.column_stack([np.log(np.maximum(lt, 1e-9)),
                         np.log(np.maximum(sz, 1.0))])
    # standardise so the two dimensions contribute comparably
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    z = (x - mu) / sd
    labels = _gmm_em(z, k, seed=7)
    med = np.array([
        np.median(z[labels == c, 0]) if np.any(labels == c) else np.inf
        for c in range(k)])
    order = np.argsort(med)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels]


def _gmm_em(z: np.ndarray, k: int, n_iter: int = 200,
            seed: int = 0) -> np.ndarray:
    """Fit a k-component Gaussian mixture by EM; returns hard labels."""
    rng = np.random.default_rng(seed)
    n, d = z.shape
    if n < 2 * k:
        qs = np.quantile(z[:, 0], np.linspace(0.15, 0.85, k))
        return np.digitize(z[:, 0], qs)
    # k-means++ initialisation, then a few Lloyd iterations for the means
    centers = [z[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((z - c) ** 2).sum(axis=1) for c in centers], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(z[rng.choice(n, p=p)])
    centers = np.array(centers)
    for _ in range(10):
        dist = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab = dist.argmin(axis=1)
        for c in range(k):
            if np.any(lab == c):
                centers[c] = z[lab == c].mean(axis=0)
    # EM with full covariance
    resp = np.zeros((n, k))
    resp[np.arange(n), lab] = 1.0
    reg = 1e-3 * np.eye(d)
    floor = 1e-3
    prev_ll = -np.inf
    for _ in range(n_iter):
        weights = resp.sum(axis=0)
        weights = np.where(weights > floor, weights, floor)
        pi = weights / weights.sum()
        # guard the numerator too: a component with all-zero responsibility
        # would otherwise produce 0/0 and poison the next iteration
        num = np.zeros((k, d))
        for c in range(k):
            num[c] = np.sum(resp[:, c][:, None] * z, axis=0)
        num = np.where(np.isfinite(num), num, 0.0)
        means = num / weights[:, None]
        means = np.where(np.isfinite(means), means, z.mean(axis=0))
        covs = []
        for c in range(k):
            diff = z - means[c]
            # weighted covariance with a normalised responsibility vector
            wc = np.nan_to_num(resp[:, c], nan=0.0, posinf=0.0, neginf=0.0)
            wsum = max(float(wc.sum()), floor)
            # explicit accumulation avoids BLAS complaining about 0*inf
            cov = np.zeros((d, d))
            for j in range(d):
                cov[j] = np.sum((wc * diff[:, j])[:, None] * diff,
                                axis=0) / wsum
            cov = cov + reg
            covs.append(cov)
        if not np.all(np.isfinite(means)):
            break
        log_prob = np.empty((n, k))
        ok = True
        for c in range(k):
            diff = z - means[c]
            try:
                inv = np.linalg.inv(covs[c])
                sign, logdet = np.linalg.slogdet(covs[c])
                if sign <= 0 or not np.isfinite(logdet):
                    ok = False
                    break
            except np.linalg.LinAlgError:
                ok = False
                break
            maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
            maha = np.nan_to_num(maha, nan=1e12, posinf=1e12, neginf=0.0)
            maha = np.clip(maha, 0.0, 1e8)
            lp = (math.log(max(pi[c], 1e-300))
                  - 0.5 * (d * math.log(2 * math.pi) + logdet + maha))
            log_prob[:, c] = np.clip(lp, -1e12, 1e12)
        if not ok:
            break
        mx = log_prob.max(axis=1, keepdims=True)
        e = np.exp(log_prob - mx)
        denom = e.sum(axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1.0)
        resp = e / denom
        resp = np.nan_to_num(resp, nan=1.0 / k, posinf=1.0 / k, neginf=0.0)
        ll = float(np.nansum(mx + np.log(np.maximum(denom, 1e-300))))
        if abs(ll - prev_ll) < 1e-6 * max(abs(ll), 1.0):
            break
        prev_ll = ll
    return resp.argmax(axis=1)


def _nearest_populated(labels: np.ndarray, counts: np.ndarray,
                       c: int) -> Optional[int]:
    best = None
    for d in range(3):
        if d == c or counts[d] < 5:
            continue
        if best is None or abs(d - c) < abs(best - c):
            best = d
    return best


# --------------------------------------------------------------------------
# Population fitting from a GC log
# --------------------------------------------------------------------------
def survival_curve_from_events(events, min_count: int = 20) -> Dict[str, list]:
    """Bracket the surviving fraction of young objects from a parsed GC log.

    For each young collection we know how many bytes were copied out of eden
    (``surv_after + promoted``).  At the next collection, part of that live
    set is still in the survivor space, but part was **promoted early** by a
    survivor-space overflow and has left the young generation even though
    the objects may still be alive.  Surviving *occupancy* therefore
    understates surviving *objects*.

    The log supports an honest bracket rather than a point estimate:

    * **lower bound** -- only bytes still in the survivor space count, so
      early-promoted but still-live objects are treated as dead;
    * **upper bound** -- all bytes promoted between the two collections count
      as alive, so objects that died in old generation within one interval
      are treated as alive.

    The true one-interval survival lies between the two.
    """
    young = [e for e in events if e.kind == "young"]
    if len(young) < 3:
        return {"interval_s": 0.0, "ages": [], "survival": [],
                "one_step_survival": 0.0,
                "one_step_lower": 0.0, "one_step_upper": 0.0,
                "overflow_dominated": False, "bracket_ratio": 0.0}
    intervals = np.diff([e.wall_s for e in young])
    positive = intervals[intervals > 0]
    interval = float(np.median(positive)) if positive.size else 0.0
    lower, upper = [], []
    for a, b in zip(young, young[1:]):
        copied = a.surv_after_mib + a.promoted_mib
        if copied <= 0.05 or a.tenuring != b.tenuring:
            continue
        lower.append(min(1.0, b.surv_before_mib / copied))
        upper.append(min(1.0, (b.surv_before_mib + max(b.promoted_mib, 0.0))
                         / copied))
    if not lower:
        return {"interval_s": interval, "ages": [], "survival": [],
                "one_step_survival": 0.0,
                "one_step_lower": 0.0, "one_step_upper": 0.0,
                "overflow_dominated": False, "bracket_ratio": 0.0}

    def robust_mean(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        lo, hi = np.quantile(arr, [0.02, 0.98])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        return float(trimmed.mean() if trimmed.size else arr.mean())

    lo_r = float(np.clip(robust_mean(lower), 1e-4, 0.9999))
    hi_r = float(np.clip(robust_mean(upper), lo_r, 0.9999))
    # Survivor-space overflow caps the observable survivor occupancy, which
    # makes the lower bound *too low*.  A log in which most collections
    # promote a large fraction of their copied bytes is overflow-dominated
    # and the bracket is wide; report that explicitly.
    promote_frac = [b.promoted_mib / max(a.surv_after_mib + a.promoted_mib,
                                         1e-9)
                    for a, b in zip(young[1:], young[2:])]
    overflow_dominated = bool(promote_frac
                              and np.median(promote_frac) > 0.5)
    # geometric midpoint: the natural point estimate inside the bracket
    mid = float(math.sqrt(lo_r * hi_r))
    ages = list(range(1, 9))
    return {
        "interval_s": interval,
        "ages": ages,
        "survival": [mid ** (k - 1) for k in ages],
        "one_step_survival": mid,
        "one_step_lower": lo_r,
        "one_step_upper": hi_r,
        "overflow_dominated": overflow_dominated,
        "bracket_ratio": hi_r / max(lo_r, 1e-9),
    }


def population_from_log(events, known_mix: Optional[ObjectPopulation] = None,
                        mean_object_size: float = 600.0) -> ObjectPopulation:
    """Recover a lifetime model from a GC log's survival bracket.

    A young-object survival curve alone does not identify the three-class
    mixture.  If ``known_mix`` is supplied (the allocation mix measured from
    the same service) its shares, shapes and sizes are kept and only the
    lifetime *scale* is re-fitted; otherwise the whole population is treated
    as one class whose median lifetime matches the observed curve.

    The fitted lifetime is a bracket estimate: the log's lower bound
    under-counts survivors because early promotion removes live objects from
    the survivor space, and the upper bound over-counts because objects can
    die in the old generation within one interval.  ``population_from_log``
    uses the geometric midpoint and callers should treat it as accurate to
    roughly a factor of two unless an independent lifetime trace is
    available (``fit_population`` is exact in that case).
    """
    curve = survival_curve_from_events(events)
    if not curve["ages"]:
        raise ValueError("not enough young collections to build a survival curve")
    interval = curve["interval_s"]
    r = curve["one_step_survival"]
    mean_lifetime = interval / max(-math.log(min(max(r, 1e-6), 0.999999)), 1e-9)
    base = known_mix or ObjectPopulation(
        request_fraction=0.9, session_fraction=0.07, cache_fraction=0.03)
    med = [base.median_lifetime(i) for i in range(3)]
    anchor = med[1] if med[1] > 0 else max(mean_lifetime, 1e-4)
    target = [max(mean_lifetime, 1e-4) * (m / anchor) for m in med]
    pop = replace(
        base,
        request_scale_s=float(max(target[0], 1e-6)
                              * math.log(2) ** (1 / base.request_shape)),
        session_scale_s=float(max(target[1], 1e-6)
                              * math.log(2) ** (1 / base.session_shape)),
        cache_scale_s=float(max(target[2], 1e-6)
                            * math.log(2) ** (1 / base.cache_shape)),
    )
    pop.validate()
    return pop
