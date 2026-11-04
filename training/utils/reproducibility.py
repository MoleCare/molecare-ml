"""
Seeding and environment capture for reproducible training runs.

Two runs with the same seed on the same machine should produce the same
metrics; when runs on different machines differ, the captured environment
should explain why (issue #31).

``set_seed`` seeds Python's ``random``, NumPy and TensorFlow, and sets
``PYTHONHASHSEED``. That variable only takes effect for interpreters started
after it is set (worker processes, Metaflow steps launched later), so it is
set here for them; a run that needs hash randomisation fixed in the current
process must also export it before Python starts.

``deterministic=True`` additionally turns on TensorFlow op determinism, which
makes GPU kernels reproducible at a throughput cost. It is off by default.
"""

import importlib.metadata
import json
import os
import platform
import random
import sys
from typing import Any, Dict, Optional

DEFAULT_SEED = 42


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        return None
    return tf


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Seed every random number generator a training run uses."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    tf = _tensorflow()
    if tf is not None:
        tf.random.set_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()


def environment_info() -> Dict[str, Any]:
    """What a run needs to record to explain a difference across machines."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "packages": sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        ),
    }

    tf = _tensorflow()
    if tf is not None:
        build: Dict[str, Any] = {}
        try:
            build = dict(tf.sysconfig.get_build_info())
        except Exception:  # not every build exposes it
            build = {}
        info["tensorflow"] = tf.__version__
        info["cuda"] = build.get("cuda_version")
        info["cudnn"] = build.get("cudnn_version")
        try:
            info["gpus"] = [gpu.name for gpu in tf.config.list_physical_devices("GPU")]
        except Exception:
            info["gpus"] = []
    return info


def save_environment(path: str, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write the environment as JSON next to the run's other artefacts."""
    info = info if info is not None else environment_info()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
    return info
