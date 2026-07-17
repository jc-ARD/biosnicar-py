"""Metadata -> prior adapter for sea-ice retrieval (roadmap C1).

Turns external context (calendar month now; location / ice-age / skin
temperature via the caller channel or future providers) into:

* **per-surface-type class log-priors** — relative, 0 = neutral, negative =
  disfavoured; combined with the spectral evidence so context can adjust which
  label wins;
* **parameter priors** ``{param: (mu, sigma)}`` — steer the fit (e.g. summer
  temperature near melting);
* **hard exclusions** — physically impossible types for the context
  (young ice in the melt season; liquid ponds / active melt in deep winter).

Design (the C1 gate):

* **Modular providers**, each selectable by name, so a standalone deployment
  uses everything while an ensemble withholds the providers whose signal the
  fusion layer already owns (avoiding double-counting) — pass ``sources=`` to
  choose. ``known_month`` is now the ``season`` provider, not hardwired
  special-casing.
* **Caller channel** ``extra_class_priors`` for information this module has no
  validated model for yet (location, ice age, skin temperature — the C2-C4
  roadmap items): IceNav/SARSAR pass their own per-class log-priors and this
  adapter composes them with provenance.
* **Provenance**: the combined :class:`PriorSet` records which source set which
  class log-prior, so A3's prior-influence audit can attribute the outcome.

Honesty about magnitudes:

* Hard exclusions are physical (a melt pond does not exist in January).
* Soft class log-prior magnitudes are conservative expert-judgment defaults
  (~1 nat ≈ 3:1), documented and overridable — they are priors, not
  measurements. Band-mode evidence gaps sit in the few-nat range
  (docs/OE_MODEL_ERROR_EXPERIMENT.md §8b), so a ~1-nat prior has real but
  bounded leverage; formal calibration of these values is roadmap A6/C6.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

SURFACE_TYPES = (
    "FYI_bare", "FYI_snow", "FYI_summer", "MYI_bare",
    "FYI_pond", "young_ice", "open_water",
)

# Default soft class-prior magnitude in nats (~e:1 ≈ 2.7:1 odds).
# Deliberately small: it must nudge, not override, the spectrum. The value is
# order-of-magnitude expert judgement, not calibrated — it is set to the scale
# of the band-mode / model_error evidence gaps (a few nats; see
# docs/OE_MODEL_ERROR_EXPERIMENT.md §8b) so it has leverage there yet stays
# powerless against a confident spectrum. Formal calibration is roadmap A6;
# callers can override per class via retrieve_sea_ice(class_priors=...).
SOFT_NAT = 1.0


@dataclass
class Contribution:
    """One provider's output."""
    class_log_prior: Dict[str, float] = field(default_factory=dict)
    param_prior: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    excluded: FrozenSet[str] = frozenset()


@dataclass
class PriorSet:
    """Combined metadata priors, with per-source provenance."""
    class_log_prior: Dict[str, float] = field(default_factory=dict)
    param_prior: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    excluded: FrozenSet[str] = frozenset()
    provenance: Dict[str, Contribution] = field(default_factory=dict)

    @property
    def sources(self):
        return tuple(self.provenance)

    def is_empty(self):
        return not (self.class_log_prior or self.param_prior or self.excluded)


# ── providers ────────────────────────────────────────────────────────────────

def season_provider(context) -> Optional[Contribution]:
    """Priors from calendar month (``context["month"]``, 1-12).

    Reproduces the original ``known_month`` parameter priors and young-ice
    melt-season exclusion exactly, and adds:
    * soft disfavouring of fresh snow cover in peak/late melt (Jul-Sep) —
      the surface is actively melting; late-summer snowfall keeps it soft;
    * hard exclusion of liquid ponds and active-melt SSL in deep winter.
    """
    month = context.get("month")
    if month is None:
        return None
    m = int(month)
    clp: Dict[str, float] = {}
    pp: Dict[str, Tuple[float, float]] = {}
    excl = set()

    # Parameter priors (values unchanged from the original known_month logic;
    # they gate the FIT, keeping temperature physically plausible per season —
    # the classic failure is an August spectrum fitted at T=-25 C. Gaussian
    # (mu, sigma), wide enough to admit the real spread, narrow enough to
    # exclude the impossible tail). SHEBA-validated as a set: widening the
    # summer mu toward -2 C collapsed summer accuracy 16/16 -> 9/16.
    if 5 <= m <= 9:                              # melt season: near-melting ice
        pp["sea_ice_temperature"] = (-4.0, 3.0)
    elif m in (11, 12, 1, 2, 3):                 # deep winter: well below freezing
        pp["sea_ice_temperature"] = (-15.0, 8.0)
    if m in (10, 11, 12, 1, 2):                  # freeze-up: young ice plausible
        pp["ice_thickness_cm"] = (5.0, 8.0)      # thin, not grease
        pp.setdefault("sea_ice_temperature", (-12.0, 6.0))

    # Hard physical exclusions (0/-inf priors — no magnitude to justify, just
    # physics): a surface type that cannot exist in the season is removed from
    # the candidate fleet upstream.
    if 5 <= m <= 9:
        excl.add("young_ice")                    # melts out / can't persist
    if m in (12, 1, 2):
        excl.update(("FYI_pond", "FYI_summer"))  # no liquid ponds / no melt SSL

    # Soft class log-priors: peak/late melt (Jul-Sep) disfavours a dry
    # snow-covered surface, because the ice is actively ablating. Confined to
    # Jul-Sep on purpose — NOT May/Jun: spring snow cover is entirely normal
    # then, so a snow penalty there would wrongly demote genuine spring snow
    # (SHEBA spring is Apr-May and must stay FYI_snow). Kept soft, not a hard
    # exclusion, because late-summer snowfall does occur.
    if m in (7, 8, 9):
        clp["FYI_snow"] = -SOFT_NAT

    return Contribution(class_log_prior=clp, param_prior=pp,
                        excluded=frozenset(excl))


PROVIDERS = {"season": season_provider}


# ── composition ──────────────────────────────────────────────────────────────

def build_prior_set(context=None, sources=None, extra_class_priors=None):
    """Compose the active providers plus the caller channel into a PriorSet.

    Parameters
    ----------
    context : dict or None
        Metadata for the providers (e.g. ``{"month": 7}``).
    sources : sequence of str or None
        Provider names to run (default: all). Pass a subset to withhold
        providers the ensemble owns; ``()`` disables all built-in providers.
    extra_class_priors : dict or None
        Caller-supplied per-type class log-priors (relative, 0 = neutral),
        composed additively and attributed to the ``"caller"`` source. This
        is how location / ice-age / temperature priors enter until they have
        validated providers of their own.
    """
    context = context or {}
    active = (list(PROVIDERS) if sources is None
              else [s for s in sources if s in PROVIDERS])

    clp: Dict[str, float] = {}
    pp: Dict[str, Tuple[float, float]] = {}
    excl = set()
    prov: Dict[str, Contribution] = {}

    for name in active:
        contrib = PROVIDERS[name](context)
        if contrib is None:
            continue
        for t, v in contrib.class_log_prior.items():
            _check_type(t)
            clp[t] = clp.get(t, 0.0) + v
        pp.update(contrib.param_prior)           # later providers override keys
        excl |= set(contrib.excluded)
        prov[name] = contrib

    if extra_class_priors:
        caller = {}
        for t, v in extra_class_priors.items():
            _check_type(t)
            clp[t] = clp.get(t, 0.0) + float(v)
            caller[t] = float(v)
        prov["caller"] = Contribution(class_log_prior=caller)

    return PriorSet(class_log_prior=clp, param_prior=pp,
                    excluded=frozenset(excl), provenance=prov)


def _check_type(t):
    if t not in SURFACE_TYPES:
        raise ValueError(
            f"unknown surface type {t!r} in class prior; valid: {SURFACE_TYPES}"
        )
