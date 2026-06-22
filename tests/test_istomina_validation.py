"""Fast guard for the Istomina (2016) IceArc validation loader.

The full retrieval validation lives in
``tests/validation_data/istomina_retrieval_validation.py`` and is run manually
(it retrieves ~120 spectra and takes minutes, like the SHEBA/Smith scripts).
This test only exercises the PANGAEA ``.tab`` parser so a regression in the
loader is caught in CI without paying the retrieval cost. It is skipped when the
PANGAEA data are not checked out.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

VAL = Path(__file__).resolve().parent / "validation_data"
DATA = VAL / "Istomina_2016" / "datasets"

pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="Istomina_2016 PANGAEA data not present"
)


def _load_module():
    sys.path.insert(0, str(VAL))
    spec = importlib.util.spec_from_file_location(
        "istomina_retrieval_validation", VAL / "istomina_retrieval_validation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loader_covers_all_stations():
    recs = _load_module().load_istomina()
    stations = {r["station"] for r in recs}
    assert len(stations) >= 6, f"expected 6 stations, got {sorted(stations)}"
    assert len(recs) >= 100, f"expected >=100 surface spectra, got {len(recs)}"


def test_labels_and_arrays_well_formed():
    mod = _load_module()
    recs = mod.load_istomina()
    valid = {"ice", "open_pond", "frozen_pond"}
    for r in recs:
        assert r["surface"] in valid
        assert r["alb"].shape == r["wl_nm"].shape
        assert 20 <= r["sza"] <= 89
        assert r["direct"] in (0, 1)
    # sky/incident references must be excluded from surface records
    assert mod._classify_surface("sky over frozen pond") is None
    assert mod._classify_surface("turq pond whole fov e") == "open_pond"
    assert mod._classify_surface("melting darker ice e") == "ice"
