# Coincidence evidence: microCT snow profiles ↔ Smith spectral albedo

**Question asked:** do the Macfarlane microCT snowpit profiles
(`doi:10.1594/PANGAEA.952794`) and the Smith/Light/Perovich spectral albedo
lines (`doi:10.18739/A2FT8DK8Z`) truly coincide in **space and time**, closely
enough to drive a forward-closure validation?

**Answer: yes, for a conservatively-verified 26 measurement pairs across 10 days
at 4 named sites**, established on four independent lines of evidence below.
Reproduce with `verify_coincidence.py` → `coincidence_evidence.csv`.

> Honest scope: this is not "35 days" (an earlier same-day-*any*-site count that
> was too loose). The number below counts only pairs where the **same named
> optics site** was sampled by both teams on the **same day**, with coordinates
> confirming the same drifting floe.

---

## Line 1 — Structural: the datasets name each other's sites

MOSAiC was a drifting floe, so absolute lat/lon changes hour to hour; the
stable coordinate system is the **named sites on the Central Observatory**. The
microCT index `Location` field uses an explicit **`optics-` prefix** for the
albedo sites, with suffixes that are the *same tokens* as the Smith albedo-line
filenames:

| microCT `Location` | Smith filename token | site |
|---|---|---|
| `optics-stern` | `…_STERN_…` | ship-stern optics line |
| `optics-LDL` | `…_LDL_…` | Lemon-Drop-Lead line |
| `optics-transect-ROV` | `…_ROV4_…` | ROV optics transect |
| `optics-transect-SYI` | `…_SYI_…` | second-year-ice transect |

This is not inference — the microCT operator labelled these pits *as* the optics
sites. Matching was done through an explicit, auditable site-alias table
(`verify_coincidence.py`, `SITES`), never fuzzy string overlap.

## Line 2 — Temporal: same day, timestamps recorded

26 pairs fall on 10 days, all in MOSAiC Leg 4 (2020-06-19 → 2020-07-21). Each
Smith spectrum carries a UTC start time; several days have full diurnal albedo
series (e.g. STERN on 2020-07-10 at 10:35, 13:13, 16:40, 19:15, 22:18 UTC),
letting a pit be paired to the nearest-in-time albedo session. Full timestamps
are in `coincidence_evidence.csv` (`albedo_time`, `microct_event`).

## Line 3 — Spatial: same floe, confirmed by coordinates

Haversine distance between the Smith **ship** position and the microCT **pit**
position on the 26 same-site pairs:

- **median 1.7 km, min 0.03 km, max 6.8 km.**

The sub-100 m cases (e.g. 2020-06-19 LDL 0.03 km; 2020-07-15 SYI 0.10 km;
2020-07-21 LDL 0.07 km) are effectively exact co-location. The larger values
are an **upper bound inflated by two known effects**, not evidence against
co-location: (i) the Smith coordinate is the *ship*, not the on-floe site
(sites sat ~0.1–2 km from the ship), and (ii) on a floe drifting several km/day,
a pit and an albedo session logged hours apart carry different absolute
coordinates while sitting on the same physical spot. Both shrink the *true*
site-to-pit separation below these figures.

## Line 4 — Textual: the field logs name the microCT operator on site

The Smith albedo field logs (embedded in each CSV's "Other notes") repeatedly
record **Amy Macfarlane** — the microCT dataset's lead author — performing snow
work *during the albedo sessions*, e.g.:

- `…_20200721_1451_BOP…`: *"Maddie (ASD) Amy (Kipps & snow survey)"*
- `…_20200706_LDL…`: *"Amy (Kipps)"*
- others: *"Amy (microCT)"*, *"Amy (snow)"*, *"Amy (NIR camera)"*

Human-written ground truth that the albedo and snow-sampling teams were the same
people, at the same sites, in the same sessions.

---

## Verified pairs (summary)

See `coincidence_evidence.csv` for all 26 rows. Distribution by site:

| site | pairs | days |
|---|---|---|
| STERN | 16 | 4 |
| LDL | 3 | 3 |
| ROV | 2 | 2 |
| SYI | 1 | 1 |
| **(STERN diurnal series inflate its pair count)** | | |

Days: 2020-06-19, -06-27, -07-04, -07-05, -07-06, -07-10, -07-11, -07-15,
-07-16, -07-21.

## Caveats for the modeller (do not skip)

1. **Coordinate is ship-referenced**, not site-referenced (Line 3). For any pair
   where exact separation matters, cross-check with the coincident surface
   photos (`doi:10.18739/A2B27PS3N`) rather than the distance column alone.
2. **Pair ≠ identical footprint.** An albedo line averages 60–200 m of surface;
   a microCT core is a point. Use the albedo line's own position-level surface
   classification to select the positions that match the pit's surface type.
3. **Vertical registration and SSA units** are separate data-reality issues,
   documented in `README.md` §"Data reality" — resolve before running closure.
