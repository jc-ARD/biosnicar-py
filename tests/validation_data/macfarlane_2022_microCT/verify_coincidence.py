#!/usr/bin/env python3
"""Verify space/time coincidence of the MOSAiC microCT snow profiles
(Macfarlane 2022, PANGAEA 952794) with the Smith spectral albedo lines
(Smith/Light/Perovich, ADC A2FT8DK8Z).

Produces `coincidence_evidence.csv` and prints the summary documented in
COINCIDENCE_EVIDENCE.md. Pure stdlib — no model, no heavy deps.

    python verify_coincidence.py
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMDIR = (HERE.parent / "smith_2021" / "resource_map_doi_10_18739_A2FT8DK8Z"
         / "data" / "SpectralAlbedoData")
INDEX = HERE / "index_summer2020.csv"

# Explicit MOSAiC Central-Observatory site aliases: canonical site → substrings
# that identify it in EITHER dataset's naming. Auditable by construction.
SITES = {
    "STERN": ["STERN"],
    "LDL": ["LDL"],
    "ROV": ["ROV", "ROV4"],
    "RS": ["RS5", "SNOW5_RS", "SNOW_RS"],
    "SYI": ["SYI"],
    "ECO": ["ECO"],
    "MET": ["METCITY", "MET-CITY", "SNOW5_MET"],
    "BOP": ["BOP"],
    "RBB": ["RBB"],
    "KINDER": ["KINDER"],
}


def canon(raw: str) -> set[str]:
    u = raw.upper()
    return {s for s, subs in SITES.items() if any(k in u for k in subs)}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_smith() -> list[dict]:
    out = []
    for f in glob.glob(str(SMDIR / "*.csv")):
        b = os.path.basename(f)
        m = re.match(r"spectralalbedo_mosaic_(\d{8})(?:_(\d{3,4}))?_(.+)_processedv0", b)
        if not m:
            continue
        date, tm, site = m.groups()
        lat = lon = None
        start = tm or ""
        for line in open(f, encoding="utf-8", errors="replace"):
            if line.startswith("Start time"):
                mm = re.search(r"(\d{3,4})", line)
                start = mm.group(1) if mm else start
            elif "Ship latitude" in line:
                lat = float(line.split(",")[1])
            elif "Ship longitude" in line:
                lon = float(line.split(",")[1])
                break
        out.append(dict(date=date, site=site, canon=canon(site),
                        lat=lat, lon=lon, start=start, file=b))
    return out


def load_microct() -> list[dict]:
    rows = list(csv.reader(open(INDEX)))
    ix = {k: i for i, k in enumerate(rows[0])}
    seen, out = set(), []
    for r in rows[1:]:
        ev = r[ix["Event (Snow pit sampling)"]]
        if ev in seen:
            continue
        seen.add(ev)
        loc = r[ix["Location"]]
        out.append(dict(date=r[ix["Date/Time (Snow pit sampling)"]].replace("-", ""),
                        loc=loc, canon=canon(loc),
                        lat=float(r[ix["Latitude (Snow pit sampling)"]]),
                        lon=float(r[ix["Longitude (Snow pit sampling)"]]),
                        event=ev, comment=r[ix["Comment"]][:45]))
    return out


def main() -> int:
    smith, mc = load_smith(), load_microct()
    pairs = {}
    for s in smith:
        for m in mc:
            if s["date"] != m["date"]:
                continue
            shared = s["canon"] & m["canon"]
            if not shared:
                continue
            d = haversine_km(s["lat"], s["lon"], m["lat"], m["lon"]) if s["lat"] else None
            pairs[(s["date"], s["file"], m["event"])] = dict(
                date=s["date"], site="|".join(sorted(shared)),
                dist_km=round(d, 2) if d is not None else "",
                albedo_time=s["start"], albedo_file=s["file"],
                microct_event=m["event"], microct_loc=m["loc"],
                microct_comment=m["comment"])
    pairs = list(pairs.values())
    days = sorted({p["date"] for p in pairs})
    dists = [p["dist_km"] for p in pairs if p["dist_km"] != ""]

    with open(HERE / "coincidence_evidence.csv", "w", newline="") as f:
        cols = ["date", "site", "dist_km", "albedo_time", "albedo_file",
                "microct_event", "microct_loc", "microct_comment"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in sorted(pairs, key=lambda x: (x["date"], x["site"])):
            w.writerow(p)

    print(f"Smith spectra: {len(smith)} | microCT summer pits: {len(mc)}")
    print(f"VERIFIED co-located pairs: {len(pairs)} across {len(days)} days")
    print(f"sites: {sorted({p['site'] for p in pairs})}")
    if dists:
        print(f"ship↔pit distance km: median {statistics.median(dists):.1f}, "
              f"min {min(dists):.2f}, max {max(dists):.1f}")
    print(f"days: {', '.join(days)}")
    print(f"→ wrote {HERE / 'coincidence_evidence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
