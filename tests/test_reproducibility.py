"""Seeding and environment capture for training runs (issue #31)."""
import json
import os
import random

import numpy as np
import pytest

from training.utils.reproducibility import environment_info, save_environment, set_seed


def _draws():
    import tensorflow as tf

    return (
        [random.random() for _ in range(3)],
        np.random.rand(3).tolist(),
        tf.random.uniform([3]).numpy().tolist(),
    )


def test_same_seed_same_numbers():
    set_seed(123)
    first = _draws()
    set_seed(123)
    assert _draws() == first


def test_different_seed_different_numbers():
    set_seed(1)
    first = _draws()
    set_seed(2)
    assert _draws() != first


def test_sets_pythonhashseed_for_child_processes():
    set_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


@pytest.mark.parametrize("bad", [-1, 1.5, "42", True, None])
def test_rejects_a_seed_that_is_not_a_non_negative_int(bad):
    with pytest.raises(ValueError):
        set_seed(bad)


def test_environment_names_python_tensorflow_and_packages():
    info = environment_info()
    assert info["python"].count(".") >= 1
    assert "tensorflow" in info
    assert any(p.lower().startswith("numpy==") for p in info["packages"])


def test_save_environment_writes_json(tmp_path):
    path = tmp_path / "run" / "environment.json"
    info = save_environment(str(path))
    assert json.loads(path.read_text()) == json.loads(json.dumps(info, sort_keys=True))
