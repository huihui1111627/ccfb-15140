"""This package simulates a generational garbage collector and tunes it.

Public entry points for library use::

    from gcsim import GCConfig, ServiceProfile, TrafficProfile
    from gcsim.simulator import HeapSimulator
    from gcsim.gclog import GCLogWriter

    svc = ServiceProfile.trading_front_end()
    traffic = TrafficProfile.trading_day()
    sim = HeapSimulator(GCConfig(), svc, traffic)
    stats = sim.run()
    print(stats.pause_percentiles())
"""

from .config import (
    GCConfig,
    HeapConfig,
    MB,
    GB,
    OldCollectorConfig,
    ServiceProfile,
    TrafficProfile,
    YoungCollectorConfig,
)
from .simulator import HeapSimulator, RunStats
from .workload import CohortTrace, build_trace

__all__ = [
    "GCConfig", "HeapConfig", "YoungCollectorConfig", "OldCollectorConfig",
    "ServiceProfile", "TrafficProfile", "HeapSimulator", "RunStats",
    "CohortTrace", "build_trace", "MB", "GB",
]

__version__ = "1.0.0"
