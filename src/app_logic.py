"""Logic behind the solvency-signal app. No Streamlit, no shap, no scripts imports, no test split.

Policy (docs/decisions.md, Phase 8 protocol):
- A firm whose Attr21 is blank is not scored (policy A).
- A firm with a refused blank (the attributes in reports/phase9_blank_rule.json, read at run time, fail closed) is not scored.
- Other blank attributes are scored with LightGBM's native missing-value handling, with a warning that
  lists them and says their effect is untested. More than MAX_OTHER_BLANKS of the other 63 attributes
  blank blocks scoring.
- Anything not finite, or text that is not a number or a recognised blank marker, blocks scoring.
- Explanations are LightGBM's native contributions times the Platt slope (calibrated log-odds), pinned
  equal to src.explain.explain_firm by tests.
"""

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import calibrate as C
from src.polish_features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
ATTRS = [f[0] for f in FEATURES]
NAME_BY_ATTR = {f[0]: f[1] for f in FEATURES}
ATTR_BY_NAME = {f[1]: f[0] for f in FEATURES}
KEY_ATTR = "Attr21"
MAX_OTHER_BLANKS = 6  # more than 6 of the other 63 attributes (more than 10%) blocks scoring
MAX_ROWS = 500
BLANK_MARKERS = {"", "?", "na", "n/a", "nan", "null", "none"}
FROZEN = ("models/lightgbm.joblib", "artifacts/calibration/calibrators.json", "artifacts/calibration/bands.json")
TEXT_FROZEN = {"artifacts/calibration/calibrators.json", "artifacts/calibration/bands.json"}
TOP_K = 6


RULE_PATH = "reports/phase9_blank_rule.json"


class FrozenArtifactError(RuntimeError):
    pass


class BlankRuleError(RuntimeError):
    """The blank rule file is missing or unreadable. Scoring fails closed."""


# ------------------------------------------------------------------------------- blank rule
def block_reason(attr: str, d: dict) -> str:
    """Refusal text for a blank attribute whose blank is refused, from the rule file's detail and the feature dictionary."""
    t, v = d["train"], d["val"]
    up = d["direction"] == "up"
    word = "higher" if up else "lower"
    flag = d.get("dictionary_flag") or ""
    parts = [f"{attr} ({d['formula']}) is blank. In the training data a blank {attr} came with a much {word} bankrupt rate "
             f"({t['blank_bankrupt']} of {t['blank_firms']} train firms against {t['present_rate']:.3f} when present; "
             f"{v['blank_bankrupt']} of {v['blank_firms']} validation firms against {v['present_rate']:.3f}), and the model "
             f"{'adds' if up else 'removes'} {abs(d['mean_contribution_when_blank_train']):.2f} calibrated log-odds for it on average. "
             "A score would reflect the blank, not the firm's finances, so the app does not score it."]
    if flag:
        parts.append(f"Dictionary note: {flag}.")
    if "undefined" in flag:
        parts.append("The data does not show whether a blank here means a zero denominator or missing data (UNVERIFIED).")
    return " ".join(parts)


def load_rule(root: Path = ROOT) -> dict:
    """Read the blank rule written by scripts/run_phase9_blanks.py. Missing or malformed raises BlankRuleError."""
    try:
        rule = json.loads((Path(root) / RULE_PATH).read_text(encoding="utf-8"))
        blocked = list(rule["blocked"]) if rule["applied"] else []
        detail = rule.get("blocked_detail", {})
        missing = [a for a in blocked if a not in detail or a not in ATTRS]
        if missing:
            raise KeyError(f"no detail for {missing}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BlankRuleError(f"the blank rule file {RULE_PATH} is missing or unreadable, scoring is disabled: {exc}") from exc
    return {"blocked": {a: block_reason(a, detail[a]) for a in blocked}, "rule": rule}


_RULE_CACHE: dict = {}


def blocked_blanks(root: Path = ROOT) -> dict:
    """{attribute: refusal reason}. Read once per root, fails closed."""
    key = str(Path(root).resolve())
    if key not in _RULE_CACHE:
        _RULE_CACHE[key] = load_rule(root)["blocked"]
    return _RULE_CACHE[key]


# --------------------------------------------------------------------------- frozen artifacts
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_matches(rel: str, data: bytes, expected: str) -> bool:
    """Raw hash always. For the two JSON text files, the same bytes with LF or CRLF line endings also match."""
    if _sha(data) == expected:
        return True
    if rel in TEXT_FROZEN:
        lf = data.replace(b"\r\n", b"\n")
        return expected in (_sha(lf), _sha(lf.replace(b"\n", b"\r\n")))
    return False


def verify_frozen(root: Path = ROOT) -> dict:
    proto = json.loads((Path(root) / "artifacts" / "phase5_protocol.json").read_text(encoding="utf-8"))
    bad = []
    for rel in FROZEN:
        p = Path(root) / rel
        if not p.exists() or not hash_matches(rel, p.read_bytes(), proto["files"][rel]["sha256"]):
            bad.append(rel)
    if bad:
        raise FrozenArtifactError(f"frozen artifact hash mismatch, scoring is disabled: {bad}")
    return {rel: proto["files"][rel]["sha256"] for rel in FROZEN}


def load_frozen(root: Path = ROOT) -> dict:
    """Verify, then load. Nothing is fitted."""
    root = Path(root)
    hashes = verify_frozen(root)
    art = C.load_artifacts(root / "artifacts" / "calibration")
    bands = art["bands"]["lightgbm"]
    return {"model": joblib.load(root / "models" / "lightgbm.joblib"),
            "calibrator": art["calibrators"]["lightgbm"][bands["selected_method"]], "bands": bands,
            "dictionary": json.loads((root / "artifacts" / "explain" / "feature_dictionary.json").read_text(encoding="utf-8")),
            "hashes": hashes}


# ---------------------------------------------------------------------------- input parsing
def resolve_columns(columns) -> tuple:
    """Map uploaded column names (Attr1..Attr64, any case, or dictionary names) to attribute codes."""
    mapping, ignored = {}, []
    for c in columns:
        key = str(c).strip()
        low = key.lower()
        attr = None
        if low.startswith("attr") and low[4:].isdigit() and f"Attr{int(low[4:])}" in NAME_BY_ATTR:
            attr = f"Attr{int(low[4:])}"
        elif key in ATTR_BY_NAME:
            attr = ATTR_BY_NAME[key]
        if attr is None:
            ignored.append(str(c))
        elif attr in mapping.values():
            raise ValueError(f"column {attr} appears more than once")
        else:
            mapping[c] = attr
    return mapping, ignored


def parse_upload(data) -> dict:
    """CSV bytes or a DataFrame to a numeric frame with one column per attribute.

    Returns {"frame", "problems" (row index -> reason, rows that cannot be scored), "ignored_columns"}.
    Raises ValueError for file-level problems (missing columns, too many rows, duplicate columns)."""
    if isinstance(data, (bytes, bytearray)):
        raw = pd.read_csv(io.BytesIO(bytes(data)), dtype=str, keep_default_na=False)
    else:
        raw = data.astype(str).where(data.notna(), "")
    if len(raw) == 0:
        raise ValueError("the file has no rows")
    if len(raw) > MAX_ROWS:
        raise ValueError(f"at most {MAX_ROWS} rows per upload, this file has {len(raw)}")
    mapping, ignored = resolve_columns(raw.columns)
    missing = [a for a in ATTRS if a not in mapping.values()]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}. The file needs all 64 attributes, Attr1 to Attr64.")
    raw = raw.rename(columns=mapping)[ATTRS]
    frame = pd.DataFrame(index=raw.index, columns=ATTRS, dtype=float)
    problems = {}
    for a in ATTRS:
        for i, v in raw[a].items():
            s = str(v).strip()
            if s.lower() in BLANK_MARKERS:
                frame.at[i, a] = np.nan
                continue
            try:
                x = float(s)
            except ValueError:
                problems.setdefault(i, f"{a} is not a number: {s[:20]!r}")
                frame.at[i, a] = np.nan
                continue
            if not np.isfinite(x):
                problems.setdefault(i, f"{a} is not a finite number")
                frame.at[i, a] = np.nan
            else:
                frame.at[i, a] = x
    return {"frame": frame, "problems": problems, "ignored_columns": ignored}


# --------------------------------------------------------------------------------- scoring
def assess(row: pd.Series, problem: str = "", blocked: dict = None) -> dict:
    """Policy for one firm: status, reason, blank attributes. blocked is {attribute: reason} for refused blanks
    (default: the Phase 9 rule file, which fails closed)."""
    blocked = blocked_blanks() if blocked is None else blocked
    blanks = [a for a in ATTRS if pd.isna(row[a])]
    other = [a for a in blanks if a != KEY_ATTR]
    refused_blank = [a for a in other if a in blocked]
    out = {"blank_attrs": blanks, "n_other_blank": len(other), "status": "scored", "reason": "", "warning": ""}
    if problem:
        out.update(status="refused", reason=problem)
    elif KEY_ATTR in blanks:
        out.update(status="refused", reason="Attr21 (sales growth) is blank. It needs last year's sales, and the model is not scored without it.")
    elif refused_blank:
        out.update(status="refused", reason=" ".join(blocked[a] for a in refused_blank))
    elif len(other) > MAX_OTHER_BLANKS:
        out.update(status="refused", reason=f"{len(other)} of the other 63 attributes are blank, above the limit of {MAX_OTHER_BLANKS}.")
    elif other:
        out["warning"] = (f"{len(other)} attribute(s) are blank ({', '.join(other)}). The model handles blanks natively, but the effect of "
                          "these blanks on the score has not been tested.")
    return out


def score_frame(frozen: dict, frame: pd.DataFrame, problems: dict = None, blocked: dict = None) -> pd.DataFrame:
    """One result row per firm, in input order. Refused firms get no probability and no band."""
    problems = problems or {}
    blocked = blocked_blanks() if blocked is None else blocked
    rows = []
    for i, (idx, r) in enumerate(frame.iterrows()):
        rows.append({"row": i + 1, "index": idx, **assess(r, problems.get(idx, ""), blocked)})
    res = pd.DataFrame(rows)
    res["probability"], res["band"] = np.nan, ""
    ok = res["status"] == "scored"
    if ok.any():
        X = frame.loc[res.loc[ok, "index"], ATTRS].rename(columns=NAME_BY_ATTR).astype(float)
        p = frozen["calibrator"].transform(frozen["model"].predict_proba(X)[:, 1])
        res.loc[ok, "probability"] = p
        res.loc[ok, "band"] = C.assign_bands(p, frozen["bands"]["t_watch"], frozen["bands"]["t_high"])
    return res


def explain_native(frozen: dict, row: pd.DataFrame, top_k: int = TOP_K) -> dict:
    """Top drivers for one scored firm: native LightGBM contributions times the Platt slope (calibrated log-odds)."""
    model, cal, bands = frozen["model"], frozen["calibrator"], frozen["bands"]
    X = row[ATTRS].rename(columns=NAME_BY_ATTR).astype(float)
    out = model.predict(X, pred_contrib=True)
    phi, base = out[0, :-1], float(out[0, -1])
    platt = isinstance(cal, C.PlattCalibrator)
    a, b = (cal.a, cal.b) if platt else (1.0, 0.0)
    contrib = phi * a
    p = float(cal.transform(model.predict_proba(X)[:, 1])[0])
    dic = frozen["dictionary"]
    drivers = []
    for j in np.argsort(-np.abs(contrib))[:top_k]:
        val = X.iloc[0, j]
        missing = bool(pd.isna(val))
        shown = "blank" if missing else f"{val:.4g}"
        verb = "raises" if contrib[j] > 0 else "lowers"
        note = f" [UNVERIFIED definition, {dic[j]['attr']}]" if dic[j]["unverified"] else ""
        drivers.append({"attr": dic[j]["attr"], "label": dic[j]["label"], "unverified": dic[j]["unverified"],
                        "value": None if missing else float(val), "contribution": float(contrib[j]), "direction": verb,
                        "sentence": f"{dic[j]['attr']} ({dic[j]['formula']}) is {shown} and {verb} the risk score by "
                                    f"{abs(contrib[j]):.2f}{note}"})
    return {"probability": p, "band": str(C.assign_bands([p], bands["t_watch"], bands["t_high"])[0]),
            "base_probability": float(1 / (1 + np.exp(-(base * a + b)))),
            "scale": "calibrated_log_odds" if platt else "raw_log_odds", "drivers": drivers}


# ------------------------------------------------------------------------------ files and text
def load_json(rel: str, root: Path = ROOT) -> dict:
    return json.loads((Path(root) / rel).read_text(encoding="utf-8"))


def demo_firms(root: Path = ROOT) -> dict:
    """app/demo_firms.json: {"rule", "firms": [{"id", "role", "dataset_label", "values": {Attr: value or null}}]}."""
    return load_json("app/demo_firms.json", root)


def demo_frame(demo: dict) -> pd.DataFrame:
    rows = [{a: (np.nan if f["values"][a] is None else float(f["values"][a])) for a in ATTRS} for f in demo["firms"]]
    return pd.DataFrame(rows, columns=ATTRS, index=[f["id"] for f in demo["firms"]])


def flattered_sentence(numbers: dict) -> str:
    """One plain sentence, from the generated numbers, that the all-firms figures are flattered by missing Attr21."""
    n = numbers["numbers"]
    tr, va = n["train_oof"], n["val"]
    return (f"The all-firms figures are flattered by firms with Attr21 blank: {tr['attr21_missing_bankrupt']} of the {tr['attr21_missing_firms']} "
            f"train firms and {va['attr21_missing_bankrupt']} of the {va['attr21_missing_firms']} validation firms with Attr21 blank are bankrupt, "
            "and the app does not score such firms.")


def floor_statement(numbers: dict) -> str:
    """The band cuts were set on all firms. Compare High precision on Attr21-present firms with the floor."""
    n = numbers["numbers"]
    floor = C.BAND_RULE["high_min_precision"]
    v, o = n["val"]["present"]["high"]["precision"], n["train_oof"]["present"]["high"]["precision"]
    rel = "under" if max(v, o) < floor else ("at or above" if min(v, o) >= floor else "on both sides of")
    return (f"The band cuts were set on all firms, not on this subset. For firms with Attr21 present, High precision is {v * 100:.1f}% on validation and "
            f"{o * 100:.1f}% on train out-of-fold, {rel} the {floor * 100:.0f}% floor the High cut was built to reach. The cuts were not changed.")


def band_history(numbers: dict, band: str) -> list:
    """Historical bankrupt rate for a band on Attr21-present firms: [(label, firms, bankrupt, rate, lo, hi)]."""
    out = []
    for label, key in (("train out-of-fold", "train_oof"), ("validation", "val")):
        r = next(x for x in numbers["numbers"][key]["present"]["band_table"] if x["band"] == band)
        out.append((label, r["firms"], r["bankrupt"], r["bankrupt_rate"], r["wilson_95"][0], r["wilson_95"][1]))
    return out


def blank_history(numbers: dict, attrs: list) -> list:
    """Descriptive history for blank attributes, from train and validation (Attr21-present firms). Not a test of the effect."""
    eff = numbers.get("blank_effects", {})
    out = []
    for a in attrs:
        e = eff.get(a)
        if not e:
            continue
        t, v = e["train"], e["val"]
        out.append(f"{a} blank: bankrupt {t['blank']['bankrupt']} of {t['blank']['firms']} train firms "
                   f"({t['blank']['rate']:.3f}) against {t['present']['rate']:.3f} when present; validation "
                   f"{v['blank']['bankrupt']} of {v['blank']['firms']} ({v['blank']['rate']:.3f}) against {v['present']['rate']:.3f}. "
                   f"Mean contribution to the score when blank in train: {e['mean_contribution_when_blank_train']:+.2f} "
                   f"calibrated log-odds, against {e['mean_contribution_when_present_train']:+.2f} when present.")
    return out


def refused_blank_names(rule_blocked: dict) -> str:
    """Attribute codes whose blank is refused, as 'Attr24 and Attr27' text."""
    names = sorted(rule_blocked, key=lambda a: int(a[4:]))
    return " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]


def template_csv() -> str:
    """Header-only template with the 64 attribute codes."""
    return ",".join(ATTRS) + "\n"
