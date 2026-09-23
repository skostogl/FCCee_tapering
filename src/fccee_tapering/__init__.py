"""FCC-ee coarse tapering granularity study.

Dipoles are powered in groups rather than tapered individually, and the
resulting orbit and optics degradation is corrected by various means. The
package generates one lattice per combination of grouping and correction
options, stores each as a checkpoint, and compares them.

Modules:
    grid        scenario names, the build DAG, the registry
    io_ckpt     checkpoint layout on disk
    regions     lattice regions, magnet circuits, dipole trim selection
    stages      one function per build step; the physics lives here
    orbit_corr  the orbit correction algorithms
    metrics     figures of merit and the standard plots, shared by everything
    build       turning scenarios into checkpoints, with stamping
    analysis    summary, delta and granularity-scan tables
"""

from . import grid, io_ckpt, metrics, regions

__all__ = ["grid", "io_ckpt", "metrics", "regions", "stages", "orbit_corr", "build", "analysis"]


def __getattr__(name):
    # These need xtrack, so they are imported on use: `grid` and `io_ckpt`
    # stay usable without the physics stack.
    if name in ("stages", "orbit_corr", "build", "analysis"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
