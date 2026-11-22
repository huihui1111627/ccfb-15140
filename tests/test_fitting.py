"""Recovering the object population from observed behaviour."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gcsim.config import ObjectPopulation, get_scenario, SimulationConfig  # noqa: E402
from gcsim.engine import Simulator  # noqa: E402
from gcsim.fitting import (  # noqa: E402
    fit_population, fit_weibull, population_from_log,
    survival_curve_from_events,
)
from gcsim.gclog import parse_log, render_simple  # noqa: E402


def draw(pop: ObjectPopulation, n: int, seed: int = 1):
    """Draw (class, size, lifetime) samples from a known population."""
    rng = np.random.default_rng(seed)
    fr = np.array(pop.class_mix())
    cls = rng.choice(3, size=n, p=fr)
    scale = np.array([pop.request_scale_s, pop.session_scale_s,
                      pop.cache_scale_s])[cls]
    shape = np.array([pop.request_shape, pop.session_shape,
                      pop.cache_shape])[cls]
    # inverse Weibull
    u = rng.random(n)
    life = scale * np.power(-np.log(np.maximum(u, 1e-12)), 1.0 / shape)
    sizes = np.array([pop.request_size_bytes, pop.session_size_bytes,
                      pop.cache_size_bytes])[cls]
    sizes = sizes * np.exp(0.4 * rng.standard_normal(n))
    return cls, sizes, life


def test_weibull_fit_recovers_parameters():
    rng = np.random.default_rng(3)
    scale, shape = 40.0, 1.6
    u = rng.random(20000)
    life = scale * np.power(-np.log(np.maximum(u, 1e-12)), 1.0 / shape)
    s_hat, k_hat = fit_weibull(life)
    assert s_hat == pytest.approx(scale, rel=0.05)
    assert k_hat == pytest.approx(shape, rel=0.05)


def test_fit_population_recovers_mix_and_lifetimes():
    truth = ObjectPopulation(
        request_fraction=0.85, session_fraction=0.12, cache_fraction=0.03,
        request_scale_s=0.005, request_shape=0.6,
        session_scale_s=30.0, session_shape=1.5,
        cache_scale_s=900.0, cache_shape=4.0,
        request_size_bytes=500.0, session_size_bytes=3000.0,
        cache_size_bytes=1200.0,
    )
    _, sizes, life = draw(truth, 60000)
    got = fit_population(life, sizes)
    # class order must be shortest -> longest, and the mix approximately right
    assert got.request_fraction == pytest.approx(0.85, abs=0.05)
    assert got.session_fraction == pytest.approx(0.12, abs=0.05)
    assert got.cache_fraction == pytest.approx(0.03, abs=0.03)
    assert got.median_lifetime(0) < got.median_lifetime(1) < got.median_lifetime(2)
    assert got.median_lifetime(1) == pytest.approx(truth.median_lifetime(1),
                                                   rel=0.35)
    assert got.session_size_bytes == pytest.approx(3000.0, rel=0.25)


def test_fit_with_known_classes_preserves_labels():
    truth = get_scenario("trading-gateway").population
    cls, sizes, life = draw(truth, 30000)
    names = ["request", "session", "cache"]
    got = fit_population(life, sizes, classes=[names[c] for c in cls])
    assert got.request_fraction == pytest.approx(truth.request_fraction, abs=0.03)
    assert got.session_fraction == pytest.approx(truth.session_fraction, abs=0.03)


def test_survival_curve_from_log_is_consistent():
    sc = get_scenario("trading-gateway")
    sc.sim = SimulationConfig(max_records_per_minor=120)
    sc.workload.day_seconds = 3600
    res = Simulator(sc).run()
    events, _ = parse_log(render_simple(res))
    curve = survival_curve_from_events(events)
    assert curve["ages"], "expected a survival curve"
    assert 0.0 < curve["one_step_survival"] <= 1.0
    s = curve["survival"]
    assert all(a >= b for a, b in zip(s, s[1:])), "survival must be monotone"


def test_population_from_log_preserves_ordering():
    sc = get_scenario("order-lifecycle")
    sc.sim = SimulationConfig(max_records_per_minor=120)
    sc.workload.day_seconds = 3600
    res = Simulator(sc).run()
    events, _ = parse_log(render_simple(res))
    pop = population_from_log(events, known_mix=sc.population)
    assert pop.median_lifetime(0) < pop.median_lifetime(1) < pop.median_lifetime(2)
    pop.validate()
