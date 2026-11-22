"""gcsim -- a generational garbage-collector simulator and GC-tuning workbench.

The model follows the mechanics of a HotSpot-style generational collector:

  * the heap is split into a young generation and an old generation;
  * young space consists of one *allocation area* (eden) plus two survivor
    spaces, one of which is always empty (the "to" space);
  * allocation happens in eden; a young collection copies the live objects
    into the survivor space, ageing them by one;
  * objects that survive ``tenuring_threshold`` young collections are
    promoted to the old generation; objects that do not fit in a survivor
    space are promoted early ("premature promotion");
  * reference cycles are unreachable garbage, just like in a real
    tracing collector -- an object is live iff it is reachable from the
    application roots;
  * a young collection pauses the application for a time proportional to
    the number of live objects it must copy;
  * old collections are either a full stop-the-world mark-sweep/compact,
    or a concurrent cycle (initial mark + concurrent mark + remark/remap)
    that only pauses briefly.  A concurrent cycle that cannot keep up with
    allocation (or that is started when the heap is already too full)
    *degenerates* into a full stop-the-world collection.

The package is pure Python 3.9 + numpy, so it runs offline.
"""

__version__ = "1.0.0"

from .config import (  # noqa: F401
    GCConfig, WorkloadConfig, ObjectPopulation, SimulationConfig,
    ServiceScenario, SCENARIOS, get_scenario, run_scenario,
)
