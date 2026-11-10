"""
Tests for quantum_sim.config.

The point of this module is that a path or a calibration number is defined
exactly once. These tests pin the contract that makes that safe: the config
file is the source of the values, the environment can relocate the data, and
an unknown dataset fails loudly rather than silently reading nothing.
"""

import numpy as np
import pytest

from quantum_sim import config


def test_project_root_is_the_repository():
    assert (config.PATHS.root / "pyproject.toml").is_file()


def test_data_and_results_live_outside_the_source_tree():
    # Data inside scripts/ or src/ is how 1.7 GB ended up a `git add .` away
    # from the history; keep them structurally separate.
    src = config.PATHS.root / "src"
    scripts = config.PATHS.root / "scripts"
    for path in (config.PATHS.raw, config.PATHS.processed, config.PATHS.db, config.PATHS.figures):
        assert src not in path.parents
        assert scripts not in path.parents


def test_derived_helpers_create_their_parent_directory(tmp_path, monkeypatch):
    paths = config.Paths(
        root=tmp_path,
        data=tmp_path / "data",
        results=tmp_path / "results",
        config_file=tmp_path / "config" / "experiments.yaml",
    )
    figure = paths.figure("sub", "a.png")
    derived = paths.derived_file("b.npz")
    assert figure.parent.is_dir()
    assert derived.parent.is_dir()
    assert figure.name == "a.png"


def test_calibration_comes_from_the_config_file():
    cal = config.calibration()
    assert cal.camera
    assert cal.pixel_um > 0
    assert 0.0 <= cal.window_alpha <= 1.0
    assert 0.0 < cal.clip_percentile <= 100.0


def test_apparatus_derives_omega_z_from_the_sheet_frequency():
    app = config.apparatus()
    assert app.omega_z == pytest.approx(2 * np.pi * app.sheet_trap_hz)
    assert app.n_areal_si == pytest.approx(app.n_areal_per_um2 * 1e12)


def test_declared_datasets_are_resolvable():
    names = config.dataset_names()
    assert names, "config/experiments.yaml declares no datasets"
    for name in names:
        ds = config.dataset(name)
        assert ds.name == name
        assert ds.path.is_absolute()
        # Every dataset must say how it is framed, one way or the other.
        assert ds.crop_box is not None or ds.crop_half is not None


def test_dataset_inherits_the_shared_calibration():
    ds = config.dataset(config.dataset_names()[0])
    cal = config.calibration()
    assert ds.camera == cal.camera
    assert ds.pixel_um == cal.pixel_um
    assert ds.rotation_deg == cal.rotation_deg


def test_unknown_dataset_names_the_available_ones():
    with pytest.raises(KeyError, match="Unknown dataset"):
        config.dataset("no_such_dataset")


def test_missing_data_directory_explains_how_to_fix_it(tmp_path):
    ds = config.Dataset(
        name="phantom",
        path=tmp_path / "absent",
        calibration=config.Calibration(),
    )
    with pytest.raises(FileNotFoundError, match="QUANTUM_SIM_DATA_DIR"):
        ds.shot_paths()


def test_shot_paths_ignores_files_that_are_not_shots(tmp_path):
    (tmp_path / "2026-07-17_0022_bec_box2_00.h5").touch()
    (tmp_path / "2026-07-17_0022_bec_box2_00_rep00001.h5").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "stray.h5").touch()

    ds = config.Dataset(name="tmp", path=tmp_path, calibration=config.Calibration())
    found = [p.rsplit("/", 1)[-1] for p in ds.shot_paths("*.h5")]
    assert found == [
        "2026-07-17_0022_bec_box2_00.h5",
        "2026-07-17_0022_bec_box2_00_rep00001.h5",
    ]
