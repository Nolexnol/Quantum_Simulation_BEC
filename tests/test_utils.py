"""
Tests for the shared helpers in quantum_sim.utils.

These functions were previously copy-pasted across several scripts, where
nothing exercised them: the duplicated `rotate_and_crop` was never checked
against a known input, and the three divergent domain-length estimators were
never compared. Each test below pins a property that can be reasoned about
independently of the implementation.
"""

import numpy as np
import pytest

from quantum_sim.utils import domains, images, pca, shots, stats


# --------------------------------------------------------------------------- #
# Shot-name parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, seq, idx, rep",
    [
        ("2026-07-17_0022_bec_box2_00.h5", "2026-07-17_0022", "00", None),
        ("2026-07-17_0022_bec_box2_00_rep00003.h5", "2026-07-17_0022", "00", "00003"),
        ("2026-08-19_0006_bec_box2_7_rep00012.h5", "2026-08-19_0006", "7", "00012"),
    ],
)
def test_parse_shot_name_fields(name, seq, idx, rep):
    parsed = shots.parse_shot_name(name)
    assert parsed["seq"] == seq
    assert parsed["idx"] == idx
    assert parsed["rep"] == rep


@pytest.mark.parametrize(
    "name",
    ["notes.txt", "2026-07-17_0022_bec_box2_00.hdf5", "bec_box2_00.h5", ""],
)
def test_parse_shot_name_rejects_other_files(name):
    assert shots.parse_shot_name(name) is None


def test_repeat_label_calls_the_unsuffixed_shot_base():
    assert shots.repeat_label("2026-07-17_0022_bec_box2_00.h5") == "base"
    assert shots.repeat_label("2026-07-17_0022_bec_box2_00_rep00002.h5") == "00002"


def test_one_shot_per_time_prefers_base_then_earliest():
    grouped = {
        10.0: {"00002": "b", "base": "a"},
        20.0: {"00003": "d", "00001": "c"},  # no base at this time
    }
    chosen = shots.one_shot_per_time(grouped, preferred=("base",))
    assert chosen == {10.0: "a", 20.0: "c"}


# --------------------------------------------------------------------------- #
# Image operations
# --------------------------------------------------------------------------- #
def test_optical_density_matches_the_definition():
    atoms = np.array([[50.0]])
    probe = np.array([[100.0]])
    background = np.array([[10.0]])
    expected = -np.log((50 - 10) / (100 - 10))
    assert images.optical_density(atoms, probe, background) == pytest.approx(expected)


def test_optical_density_floors_a_dark_pixel_instead_of_returning_nan():
    # atoms below background: read noise, not negative absorption.
    od = images.optical_density(np.array([[5.0]]), np.array([[100.0]]), np.array([[10.0]]))
    assert np.isfinite(od).all()


def test_rotate_and_crop_leaves_no_zero_padding():
    # A uniform disc of ones: any padding introduced by the rotation is zero,
    # so an all-positive result proves the inscribed square was cut correctly.
    field = np.ones((101, 101))
    out = images.rotate_and_crop(field, angle=45.0)
    assert out.size > 0
    assert (out > 0.5).all(), "rotation padding leaked into the crop"


def test_rotate_and_crop_is_square_and_smaller_than_the_input():
    out = images.rotate_and_crop(np.ones((120, 120)), angle=45.0)
    assert out.shape[0] == out.shape[1]
    assert out.shape[0] < 120


def test_threshold_centroid_finds_an_offset_blob():
    field = np.zeros((64, 64))
    field[40:46, 20:26] = 1.0
    row, col = images.threshold_centroid(field, sigma=1.0, top_frac=0.05)
    assert row == pytest.approx(42.5, abs=1.5)
    assert col == pytest.approx(22.5, abs=1.5)


def test_register_shifts_recovers_a_known_translation():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(64, 64))
    stack = np.stack([base, np.roll(base, 3, axis=0), np.roll(base, -2, axis=1)])
    shifts = images.register_shifts(stack)
    # Shifts are measured against the stack mean, so differences are the
    # meaningful quantity: shot 1 sits 5 px below shot 2 in y.
    assert shifts[1, 0] - shifts[0, 0] == pytest.approx(3.0, abs=1.0)
    assert shifts[2, 1] - shifts[0, 1] == pytest.approx(-2.0, abs=1.0)


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def test_pca_eigenvalues_are_a_normalised_variance_share():
    rng = np.random.default_rng(1)
    stack = rng.normal(size=(8, 16, 16))
    evals, evecs = pca.pca_svd(stack)
    assert evals.sum() == pytest.approx(1.0)
    assert (evals >= 0).all()
    assert (np.diff(evals) <= 1e-12).all(), "eigenvalues must be descending"
    assert evecs.shape[:2] == (16, 16)


def test_pca_vectors_are_orthonormal():
    rng = np.random.default_rng(2)
    evals, evecs = pca.pca_svd(rng.normal(size=(6, 10, 10)))
    basis = evecs.reshape(-1, evecs.shape[-1])
    gram = basis.T @ basis
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-8)


def test_pca_rejects_a_constant_stack():
    with pytest.raises(ValueError, match="non-zero variance"):
        pca.pca_svd(np.zeros((4, 8, 8)))


def test_align_signs_removes_the_arbitrary_svd_sign():
    rng = np.random.default_rng(3)
    mode = rng.normal(size=(8, 8))
    frames = [mode, -mode, mode, -mode]
    aligned = pca.align_signs(frames)
    # Consecutive frames must now agree in sign.
    for previous, current in zip(aligned, aligned[1:]):
        assert np.dot(previous.ravel(), current.ravel()) > 0


def test_remove_components_projects_out_exactly_that_subspace():
    rng = np.random.default_rng(4)
    stack = rng.normal(size=(5, 12, 12))
    _, evecs = pca.pca_svd(stack)
    cleaned, removed = pca.remove_components(stack[0], evecs, 2)

    assert np.allclose(cleaned + removed, stack[0])
    # What remains must be orthogonal to the components taken out.
    basis = evecs[..., :2].reshape(-1, 2)
    assert np.allclose(basis.T @ cleaned.ravel(), 0.0, atol=1e-8)


# --------------------------------------------------------------------------- #
# Domain statistics
# --------------------------------------------------------------------------- #
def test_domain_length_recovers_a_known_stripe_wavelength():
    # A single sinusoid of wavelength 16 px has all its power at one |k|, so
    # the first moment of S must return that wavelength.
    n, wavelength = 128, 16.0
    x = np.arange(n)
    field = np.sin(2 * np.pi * x / wavelength)[None, :] * np.ones((n, 1))
    assert domains.domain_length_px(field) == pytest.approx(wavelength, rel=0.02)


def test_domain_length_um_scales_with_the_pixel_size():
    rng = np.random.default_rng(5)
    field = rng.normal(size=(64, 64))
    px = domains.domain_length_px(field)
    assert domains.domain_length_um(field, 0.66) == pytest.approx(px * 0.66)


def test_domain_length_ignores_an_additive_offset():
    # The de-meaning step means overall brightness must not change the answer.
    rng = np.random.default_rng(6)
    field = rng.normal(size=(48, 48))
    assert domains.domain_length_px(field + 100.0) == pytest.approx(
        domains.domain_length_px(field)
    )


def test_structure_factor_drops_the_uniform_mode():
    assert domains.structure_factor(np.ones((8, 8)))[0, 0] == 0.0


@pytest.mark.parametrize("kind", ["tukey", "hann", "none"])
def test_window_2d_shape_and_range(kind):
    window = domains.window_2d((32, 24), alpha=0.5, kind=kind)
    assert window.shape == (32, 24)
    assert 0.0 <= window.min() and window.max() <= 1.0 + 1e-12


def test_window_2d_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown window kind"):
        domains.window_2d((8, 8), kind="blackman")


def test_radial_autocorr_starts_at_one():
    rng = np.random.default_rng(7)
    r, g = domains.radial_autocorr(rng.normal(size=(32, 32)), px_um=0.66)
    assert g[0] == pytest.approx(1.0)
    assert r[0] == 0.0


def test_first_zero_interpolates_the_crossing():
    r = np.array([0.0, 1.0, 2.0, 3.0])
    g = np.array([1.0, 0.5, -0.5, -1.0])  # crosses midway between r=1 and r=2
    assert domains.first_zero(r, g) == pytest.approx(1.5)


def test_first_zero_is_nan_when_there_is_no_crossing():
    assert np.isnan(domains.first_zero(np.arange(4.0), np.ones(4)))


def test_local_exponents_recovers_a_pure_power_law():
    # L = A t^(1/3) must give 1/z = 1/3 for every pair.
    times = np.array([40.0, 80.0, 160.0, 320.0])
    lengths = 2.0 * times ** (1 / 3)
    _, _, one_over_z, _ = domains.local_exponents(times, lengths, t_min=30.0)
    assert len(one_over_z) > 0
    assert np.allclose(one_over_z, 1 / 3, atol=1e-9)


def test_local_exponents_returns_empty_arrays_when_no_pairs_qualify():
    t_mid, L_mid, z_inv, err = domains.local_exponents(
        np.array([10.0, 11.0]), np.array([1.0, 1.1]), t_min=30.0
    )
    assert len(t_mid) == len(L_mid) == len(z_inv) == len(err) == 0


# --------------------------------------------------------------------------- #
# Robust statistics
# --------------------------------------------------------------------------- #
def test_modified_z_flags_the_outlier_not_its_neighbours():
    z = stats.modified_z([10.0, 10.1, 9.9, 10.05, 50.0])
    assert abs(z[-1]) > stats.Z_THRESHOLD
    assert (np.abs(z[:-1]) < stats.Z_THRESHOLD).all()


def test_modified_z_is_not_inflated_by_the_outlier_it_measures():
    # The reason for median/MAD over mean/SD: one extreme value must not
    # enlarge the spread enough to hide itself.
    clean = np.concatenate([np.full(20, 10.0) + np.arange(20) * 0.01, [1000.0]])
    assert abs(stats.modified_z(clean)[-1]) > stats.Z_THRESHOLD


def test_modified_z_reports_nothing_when_every_value_is_identical():
    assert np.all(stats.modified_z([5.0] * 6) == 0.0)


def test_modified_z_handles_an_empty_input():
    assert stats.modified_z([]).size == 0


def test_cube_root_law():
    assert stats.cube_root(8.0, 2.0) == pytest.approx(4.0)
