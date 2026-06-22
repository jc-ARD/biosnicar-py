#!/usr/bin/env python3
"""Independent retrieval validation against Istomina et al. (2016) IceArc spectra.

A third independent campaign (after SHEBA/Grenfell 1998 and Smith/MOSAiC 2020):

  Istomina, L., Nicolaus, M. & Perovich, D. K. (2016): Surface spectral albedo
  complementary to ROV transmittance measurements at 6 ice stations during
  POLARSTERN cruise ARK-XXVII/3 (IceArc) in 2012. PANGAEA,
  https://doi.org/10.1594/PANGAEA.867292   (CC-BY-3.0)

  * year/site : 2012, Central Arctic ~83 N (vs SHEBA 1998 ~76 N, MOSAiC 2020 ~82 N)
  * instrument: ASD FieldSpecPro 3, 350-2500 nm
  * surfaces  : summer sea ice (white/melting/ridged) and melt ponds, the latter
                mostly REFROZEN (thin ice lids, "ice 2-3 cm") at the Aug-Sep dates
  * labels    : per-spectrum free-text Comment in the station documentation.tab
                (e.g. "surf ice", "surf dark frozen pond 3") — independent of the
                model, but coarse and operator-written.

Honest scope (why this is treated like the Smith hold-out, not a clean test):
  * Refrozen ponds are optically thin ice over water, NOT the liquid-water
    `FYI_pond` class — they are expected to read as bare/young ice, and are
    reported in their own row rather than scored as pond failures.
  * Summer melting ice is subject to the snow/SSL/bare-ice melt-season
    degeneracy documented in sea_ice_validation.md (6.4-6.5).
  * Skies are largely overcast (diffuse); the illumination flag is taken from
    each station's recorded sky conditions.

Usage::

    python tests/validation_data/istomina_retrieval_validation.py [--json out.json]
"""

import argparse
import collections
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from run_global_validation import _solar_zenith  # noqa: E402  (shared SZA helper)

from biosnicar.sea_ice.retrieve import retrieve_sea_ice  # noqa: E402
from biosnicar.sea_ice.spectral_utils import (  # noqa: E402
    MODEL_WAVELENGTHS_NM, resample_to_model_grid,
)

DATA = HERE / "Istomina_2016" / "datasets"
WL_UM = MODEL_WAVELENGTHS_NM / 1000.0


# ── PANGAEA .tab parsing ─────────────────────────────────────────────────────

def _split_header(path):
    """Return (header_lines, table_lines) splitting on the closing '*/'."""
    lines = path.read_text(errors="replace").splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == "*/":
            return lines[:i], lines[i + 1:]
    return [], lines


def _station_meta(alb_header):
    """Lat, lon, ISO date, and direct/diffuse flag from an ALB-R header."""
    text = "\n".join(alb_header)
    lat = float(re.search(r"MEDIAN LATITUDE:\s*([-\d.]+)", text).group(1))
    lon = float(re.search(r"MEDIAN LONGITUDE:\s*([-\d.]+)", text).group(1))
    date = re.search(r"DATE/TIME START:\s*(\S+)", text).group(1)
    sky = ""
    m = re.search(r"Sky conditions:\s*([^\n*]+)", text)
    if m:
        sky = m.group(1).lower()
    # "solar disk visible" => direct beam available; otherwise overcast/diffuse.
    direct = 1 if ("solar disk visible" in sky and "not visible" not in sky) else 0
    return lat, lon, date, sky, direct


def _load_doc_labels(doc_path):
    """{spectrum_id: comment_text} from a station documentation.tab."""
    _, rows = _split_header(doc_path)
    labels = {}
    for ln in rows[1:]:                      # skip the column-name row
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        fname, comment = parts[1], parts[2]
        sid = fname.split("|")[-1].strip()   # "ROV 1 | 110812...e00000" -> id
        labels[sid] = comment.strip().lower()
    return labels


_ID_RE = re.compile(r"Alb \(([^)]+)\)")


def _load_alb(alb_path):
    """Return (spectrum_ids, wl_nm, albedo[nspec, nwl]) from an ALB-R.tab."""
    _, rows = _split_header(alb_path)
    header = rows[0].split("\t")
    ids = [_ID_RE.match(c).group(1) if _ID_RE.match(c) else None for c in header[1:]]
    wl, cols = [], []
    for ln in rows[1:]:
        v = ln.split("\t")
        try:
            wl.append(float(v[0]))
        except (ValueError, IndexError):
            continue
        cols.append([float(x) if x.strip() else np.nan for x in v[1:]])
    wl = np.array(wl)
    alb = np.array(cols).T                   # (nspec, nwl)
    return ids, wl, alb


def _classify_surface(comment):
    """Coarse expected surface from the operator field comment: 'ice',
    'open_pond', 'frozen_pond', or None (incident-sky reference).

    These are operator field notes, not controlled surface-type labels — the
    convention varies between stations ("surf ice" vs "melting darker ice e"
    vs "turq pond whole fov e"), so the labelling is necessarily coarse and
    pond labels are FOV-contaminated (e.g. "pond edge", "pond at edge of fov").
    Used for the informational breakdown only, never as a pass/fail gate.
    """
    c = comment
    if "sky" in c:                           # incident-sky reference, not albedo
        return None
    is_pond = "pond" in c
    # Refrozen / thin-lid ponds: optically thin ice over water, NOT FYI_pond.
    is_frozen = any(k in c for k in ("frozen", " fr ", "fr ", "cm", "snow cover",
                                     "ice formation", "ice at bottom"))
    if is_pond and is_frozen:
        return "frozen_pond"
    if is_pond:
        return "open_pond"
    return "ice"                             # ice, ridge, sastrugi, white/melting


# ── load + retrieve ──────────────────────────────────────────────────────────

def load_istomina():
    """Return per-spectrum records with observed albedo, label, geometry."""
    recs = []
    for alb_path in sorted(DATA.glob("*ALB-R*.tab")):
        station = alb_path.stem.split("_ALB")[0]          # e.g. PS80_224-1
        doc_path = DATA / f"{station}_documentation.tab"
        if not doc_path.exists():
            continue
        alb_head, _ = _split_header(alb_path)
        lat, lon, date, sky, direct = _station_meta(alb_head)
        labels = _load_doc_labels(doc_path)
        ids, wl, alb = _load_alb(alb_path)
        y, mo, d = int(date[:4]), int(date[5:7]), int(date[8:10])
        sza = _solar_zenith(lat, lon, y, mo, d, 11.0)     # ~local noon-ish
        for j, sid in enumerate(ids):
            if sid is None or sid not in labels:
                continue
            surf = _classify_surface(labels[sid])
            if surf is None:
                continue
            recs.append(dict(station=station, sid=sid, surface=surf,
                             comment=labels[sid], wl_nm=wl, alb=alb[j],
                             sza=float(np.clip(sza, 20, 89)), direct=direct,
                             month=mo))
    return recs


def run():
    rows = []
    for r in load_istomina():
        obs = resample_to_model_grid(r["alb"], r["wl_nm"], method="linear")
        mask = np.isfinite(obs) & (obs >= 0) & (obs <= 1.0)
        if mask.sum() < 100:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = retrieve_sea_ice(observed=np.nan_to_num(np.clip(obs, 0, 1)),
                                   wavelength_mask=mask, solzen=r["sza"],
                                   direct=r["direct"], known_month=r["month"])
        vis = mask & (MODEL_WAVELENGTHS_NM >= 400) & (MODEL_WAVELENGTHS_NM <= 1000)
        if vis.sum() < 20:                   # spectrum unusable in the VIS
            continue
        rms = float(np.sqrt(np.mean((res.predicted_albedo[vis] - obs[vis]) ** 2)))
        rows.append(dict(station=r["station"], surface=r["surface"],
                         comment=r["comment"], stype=res.surface_type,
                         conf=float(res.confidence), rms_vis=rms,
                         flags=[k for k, v in res.quality_flag_description().items() if v]))
    return rows


def summarise(rows):
    n = len(rows)
    print(f"\nIstomina et al. (2016) IceArc — independent retrieval validation")
    print(f"  {n} labelled surface spectra from {len({r['station'] for r in rows})} stations\n")
    # Classification breakdown per labelled surface
    for surf in ("ice", "open_pond", "frozen_pond"):
        sub = [r for r in rows if r["surface"] == surf]
        if not sub:
            continue
        br = collections.Counter(r["stype"] for r in sub)
        rms = np.median([r["rms_vis"] for r in sub])
        print(f"  labelled '{surf}' (n={len(sub)}, med VIS RMS {rms:.3f}):")
        for t, c in br.most_common():
            print(f"      {t:12s} {c:3d}  ({c/len(sub):.0%})")
    # Honest expectation checks (not pass/fail gates — informational)
    ice = [r for r in rows if r["surface"] == "ice"]
    ice_icelike = sum(r["stype"] in ("FYI_bare", "FYI_summer", "MYI_bare", "FYI_snow")
                      for r in ice)
    pond = [r for r in rows if r["surface"] == "open_pond"]
    pond_pondlike = sum(r["stype"] in ("FYI_pond", "open_water") for r in pond)
    print(f"\n  labelled ice -> ice-like class: {ice_icelike}/{len(ice)}"
          if ice else "")
    if pond:
        print(f"  labelled open pond -> pond/water class: {pond_pondlike}/{len(pond)}")
    med = np.median([r["rms_vis"] for r in rows])
    print(f"  overall median VIS RMS: {med:.3f}")
    return dict(n=n,
                ice_icelike=[ice_icelike, len(ice)],
                open_pond_pondlike=[pond_pondlike, len(pond)],
                median_rms_vis=float(med))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = run()
    summary = summarise(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary,
             "rows": [{k: r[k] for k in ("station", "surface", "comment",
                                         "stype", "conf", "rms_vis", "flags")}
                      for r in rows]}, indent=2))
        print(f"  json: {args.json}")


if __name__ == "__main__":
    main()
