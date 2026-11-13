"""V1 — the validation corpus is frozen and its roles are enforced."""

import sys
from pathlib import Path

import pytest

VAL = Path(__file__).resolve().parent / "validation_data"
ROOT = VAL.parents[1]
sys.path.insert(0, str(VAL))

import corpus  # noqa: E402


@pytest.fixture(scope="module")
def manifest():
    return corpus.load_manifest()


def test_every_dataset_has_a_valid_role(manifest):
    for name, entry in manifest["datasets"].items():
        assert entry["role"] in corpus.ROLES, name
        assert entry["files"], f"{name} has no hashed files — run --freeze"


def test_there_is_a_frozen_test_set(manifest):
    assert corpus.campaigns_with_role("test", manifest), (
        "No dataset holds the 'test' role, so no generalisation number can "
        "be quoted."
    )


def test_data_on_disk_matches_the_freeze(manifest):
    """Editing, adding or removing a data file must be a deliberate act."""
    report = corpus.verify(manifest)
    checked = 0
    for name, r in report.items():
        if not r["present"]:
            continue        # not redistributable / not fetched: nothing to check
        checked += 1
        assert not r["changed"], f"{name}: files changed: {r['changed'][:5]}"
        assert not r["missing"], f"{name}: files missing: {r['missing'][:5]}"
        assert not r["added"], (
            f"{name}: files not in the manifest: {r['added'][:5]} — if they "
            "belong to the corpus, run corpus.py --freeze deliberately."
        )
    if not checked:
        pytest.skip("no validation data on disk")


def test_test_set_cannot_be_fitted(manifest):
    for name in corpus.campaigns_with_role("test", manifest):
        corpus.require_role([name], "evaluate", manifest)
        for purpose in ("tune", "calibrate"):
            with pytest.raises(corpus.CorpusError, match="frozen"):
                corpus.require_role([name], purpose, manifest)


def test_calibration_set_cannot_be_tuned_on(manifest):
    for name in corpus.campaigns_with_role("calibration", manifest):
        corpus.require_role([name], "calibrate", manifest)
        with pytest.raises(corpus.CorpusError):
            corpus.require_role([name], "tune", manifest)


def test_aliases_resolve(manifest):
    for name, entry in manifest["datasets"].items():
        for alias in entry.get("aliases", ()):
            assert corpus.role_of(alias, manifest) == entry["role"]
    with pytest.raises(corpus.CorpusError, match="not in the validation corpus"):
        corpus.role_of("no_such_campaign", manifest)


def test_shipped_model_error_is_not_calibrated_on_the_test_set(manifest):
    """The one shipped artifact fitted to field data must respect the freeze."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_model_error_covariance as build

    corpus.require_role(build.CAL_CAMPAIGNS, "calibrate", manifest)
    assert corpus.role_of(build.HOLDOUT, manifest) == "test"

    from biosnicar.inverse.model_error import ModelErrorCovariance
    meta = ModelErrorCovariance.load().meta
    corpus.require_role(meta["calibration_campaigns"], "calibrate", manifest)


def test_default_model_error_uses_no_field_data():
    """The default error budget is computed from the forward model alone, so
    there is no campaign it could have been fitted to."""
    from biosnicar.inverse.model_error import PhysicsModelError
    meta = PhysicsModelError.load().meta
    assert meta["source"] == "physics"
    assert not any("campaign" in k for k in meta)
