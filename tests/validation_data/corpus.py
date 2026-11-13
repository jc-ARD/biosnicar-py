#!/usr/bin/env python3
"""The validation corpus, its roles, and the freeze that keeps them apart (V1).

Every field dataset under ``tests/validation_data/`` has exactly one role:

``development``
    Free to look at, fit against and tune on.  Numbers quoted from it are
    in-sample and say how well the model *can* be made to fit, not how well
    it generalises.

``calibration``
    Feeds a shipped artifact (today: the opt-in field-fitted model-error
    covariance, ``data/model_error/field_se_v1.npz``).  May be fitted against
    for that artifact; any number quoted from it is in-sample for the
    artifact.  The default error budget, ``physics_se_v1.npz``, is computed
    from the forward model and uses no dataset in any role.

``test``
    Frozen.  Never used to tune the forward model, an emulator, a prior, a
    band mask, a threshold or a covariance.  Headline generalisation numbers
    are quoted from here and only from here.  Looking at a test-set result
    and then changing the model to improve it converts the dataset to
    ``development`` — there is no way back, so the freeze date and every
    such conversion are recorded in the manifest.

The roles live in ``corpus_manifest.json`` next to this file, together with a
SHA-256 of every data file, so the corpus cannot drift silently: editing,
adding or removing a file in a frozen dataset fails
``tests/test_validation_corpus.py``.

The enforcement point for code is :func:`require_role`.  Anything that fits
or calibrates calls it with the campaigns it is about to use, and it raises
if one of them is a test set.

Usage::

    uv run python tests/validation_data/corpus.py            # verify
    uv run python tests/validation_data/corpus.py --status   # roles + counts
    uv run python tests/validation_data/corpus.py --freeze   # (re)write hashes

``--freeze`` rewrites the hashes for the files on disk and keeps the roles;
it is for adding a new dataset, not for making a failing check pass.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "corpus_manifest.json"

ROLES = ("development", "calibration", "test")

# Roles a campaign may hold for each purpose.  A test set may be *evaluated*
# and nothing else.
_ALLOWED = {
    "tune": {"development"},
    "calibrate": {"development", "calibration"},
    "evaluate": set(ROLES),
}

_SKIP_DIRS = {"__pycache__", "figures"}
_SKIP_SUFFIXES = {".py", ".pyc", ".png", ".json", ".md"}


class CorpusError(RuntimeError):
    """A validation dataset is being used outside its role."""


def load_manifest(path=MANIFEST):
    with open(path) as fh:
        return json.load(fh)


def _data_files(root):
    """Data files of one dataset: everything except code, figures, reports."""
    out = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        if _SKIP_DIRS & set(f.relative_to(root).parts[:-1]):
            continue
        if f.suffix.lower() in _SKIP_SUFFIXES:
            continue
        out.append(f)
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def role_of(campaign, manifest=None):
    """Role of *campaign* (a manifest key or one of its aliases)."""
    manifest = manifest or load_manifest()
    for name, entry in manifest["datasets"].items():
        if campaign == name or campaign in entry.get("aliases", ()):
            return entry["role"]
    raise CorpusError(
        f"{campaign!r} is not in the validation corpus manifest "
        f"({MANIFEST.name}). Add it with a role before using it."
    )


def campaigns_with_role(role, manifest=None):
    manifest = manifest or load_manifest()
    return sorted(n for n, e in manifest["datasets"].items()
                  if e["role"] == role)


def require_role(campaigns, purpose, manifest=None):
    """Raise :class:`CorpusError` unless every campaign may serve *purpose*.

    *purpose* is ``"tune"`` (forward model, emulators, priors, masks,
    thresholds), ``"calibrate"`` (a shipped statistical artifact) or
    ``"evaluate"``.
    """
    if purpose not in _ALLOWED:
        raise ValueError(f"purpose must be one of {sorted(_ALLOWED)}")
    manifest = manifest or load_manifest()
    bad = [(c, role_of(c, manifest)) for c in campaigns
           if role_of(c, manifest) not in _ALLOWED[purpose]]
    if bad:
        detail = ", ".join(f"{c} (role: {r})" for c, r in bad)
        raise CorpusError(
            f"cannot {purpose} on {detail}. The test corpus is frozen "
            f"(V1): it may be evaluated, never fitted. See "
            f"tests/validation_data/corpus.py."
        )


def verify(manifest=None, root=HERE):
    """Compare the files on disk with the manifest.

    Returns ``{dataset: {"missing": [...], "changed": [...], "added": [...],
    "present": bool}}``.  A dataset whose directory holds none of its
    manifest files is reported ``present=False`` and not otherwise checked:
    some campaigns are not redistributable and are fetched separately.
    """
    manifest = manifest or load_manifest()
    report = {}
    for name, entry in manifest["datasets"].items():
        base = root / entry["path"]
        want = entry["files"]
        have = ({f.relative_to(base).as_posix(): f for f in _data_files(base)}
                if base.is_dir() else {})
        present = any(rel in have for rel in want)
        r = dict(present=present, missing=[], changed=[], added=[])
        if present:
            r["missing"] = sorted(set(want) - set(have))
            r["added"] = sorted(set(have) - set(want))
            r["changed"] = sorted(
                rel for rel in set(want) & set(have)
                if _sha256(have[rel]) != want[rel]
            )
        report[name] = r
    return report


def freeze(root=HERE, path=MANIFEST):
    """Rewrite the file hashes from disk, keeping roles and notes."""
    manifest = load_manifest(path)
    for name, entry in manifest["datasets"].items():
        base = root / entry["path"]
        files = _data_files(base) if base.is_dir() else []
        if not files:
            print(f"  {name}: no data on disk — hashes left unchanged")
            continue
        entry["files"] = {f.relative_to(base).as_posix(): _sha256(f)
                          for f in files}
        print(f"  {name}: {len(files)} files hashed")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--freeze", action="store_true",
                    help="rewrite file hashes from disk (roles unchanged)")
    ap.add_argument("--status", action="store_true",
                    help="print roles and file counts")
    args = ap.parse_args()

    if args.freeze:
        freeze()
        return 0

    manifest = load_manifest()
    if args.status:
        print(f"corpus frozen {manifest['frozen_on']}\n")
        for name, e in manifest["datasets"].items():
            print(f"  {name:12} {e['role']:12} {len(e['files']):4} files  "
                  f"{e['description']}")
        return 0

    report = verify(manifest)
    ok = True
    for name, r in report.items():
        role = manifest["datasets"][name]["role"]
        if not r["present"]:
            print(f"  {name:12} {role:12} not on disk (skipped)")
            continue
        problems = {k: v for k, v in r.items() if k != "present" and v}
        if problems:
            ok = False
            print(f"  {name:12} {role:12} DRIFTED: "
                  + "; ".join(f"{k} {len(v)}" for k, v in problems.items()))
            for k, v in problems.items():
                for rel in v[:5]:
                    print(f"      {k}: {rel}")
        else:
            print(f"  {name:12} {role:12} ok")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
