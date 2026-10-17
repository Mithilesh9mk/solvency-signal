"""Phase 9: how blank attributes change the score (train and validation only, frozen model).

Run: python -m scripts.run_phase9_blanks

Writes reports/phase9_blanks.{md,json} and reports/phase9_blank_rule.json. The frozen model, calibrator and cuts are unchanged
and the test split is never loaded. Criteria, ceiling and the deep-dive design are fixed in docs/decisions.md (Phase 9
protocol), written before this ran. All randomness uses SEED = 42.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.run_phase6 import verify_frozen
from scripts.run_phase8_prep import subset_block
from src import app_logic as A
from src import calibrate as C
from src import explain as E
from src.polish_features import FEATURES
from src.preprocess import ROOT
from src.train import load_split, make_lgbm

SEED = 42
N_BOOT = 1000
N_PERM = 1000
NL = "\n"
KEY = "sales_growth"
KEY_ATTR = "Attr21"
MIN_TRAIN_BLANK = 20
MIN_VAL_BLANK = 10
RATIO_UP, RATIO_DOWN = 3.0, 1 / 3
VAL_RATIO_UP, VAL_RATIO_DOWN = 2.0, 0.5
MIN_CONTRIB = 1.0
CEIL_FIRMS, CEIL_BANKRUPT = 0.10, 0.30
REG_MAX_BLANK_RATE = 0.01
REG_MAX_DROP = 0.05
GROUPS = (("0 blanks", 0, 0), ("1 to 2 blanks", 1, 2), ("3 to 6 blanks", 3, 6), ("more than 6 blanks", 7, 10 ** 6))
REPORTS = ROOT / "reports"
NAME = {f[1]: f[0] for f in FEATURES}
COL = {f[0]: f[1] for f in FEATURES}
FORMULA = {f[0]: f[2] for f in FEATURES}


def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


# -------------------------------------------------------------------------------- criteria
def rate(y, m):
    return float(y[m].mean()) if m.any() else None


def crit1(k_b, n_b, k_p, n_p):
    """Criterion 1 on train counts. Returns "up", "down" or None."""
    if n_b == 0 or n_p == 0 or k_p == 0:
        return None
    rb, rp = k_b / n_b, k_p / n_p
    lo, hi = E.wilson(k_b, n_b)
    if rb >= RATIO_UP * rp and lo > rp:
        return "up"
    if rb <= RATIO_DOWN * rp and hi < rp:
        return "down"
    return None


def crit3(d, k_b, n_b, k_p, n_p):
    """Criterion 3 on validation counts. Returns True, False or None (not evaluable)."""
    if n_b < MIN_VAL_BLANK:
        return None
    if n_p == 0 or k_p == 0:
        return None
    ratio = (k_b / n_b) / (k_p / n_p)
    return bool(ratio >= VAL_RATIO_UP) if d == "up" else bool(ratio <= VAL_RATIO_DOWN)


def evaluate_attr(ytr, btr, yva, bva, contrib_blank) -> dict:
    """The three criteria for one attribute, from labels and blank masks on Attr21-present firms."""
    k_b, n_b, k_p, n_p = int(ytr[btr].sum()), int(btr.sum()), int(ytr[~btr].sum()), int((~btr).sum())
    d = crit1(k_b, n_b, k_p, n_p)
    v_kb, v_nb, v_kp, v_np = int(yva[bva].sum()), int(bva.sum()), int(yva[~bva].sum()), int((~bva).sum())
    c1 = d is not None
    c2 = bool(c1 and ((d == "up" and contrib_blank >= MIN_CONTRIB) or (d == "down" and contrib_blank <= -MIN_CONTRIB)))
    c3 = crit3(d, v_kb, v_nb, v_kp, v_np) if c1 else None
    return {"direction": d, "c1": c1, "c2": c2, "c3": c3, "blocked": bool(c1 and c2 and c3 is True)}


# ----------------------------------------------------------------------------- history
def history(X_tr, y_tr, X_va, y_va, model, calibrator, oof_p, p_va) -> dict:
    """Per-candidate history, criteria and calibration among blank firms, on Attr21-present firms."""
    a = calibrator.a if isinstance(calibrator, C.PlattCalibrator) else 1.0
    ptr = ~X_tr[KEY].isna().to_numpy()
    pva = ~X_va[KEY].isna().to_numpy()
    Xt, yt, Xv, yv = X_tr[ptr], np.asarray(y_tr)[ptr], X_va[pva], np.asarray(y_va)[pva]
    ot, ov = np.asarray(oof_p)[ptr], np.asarray(p_va)[pva]
    contrib = model.predict(Xt, pred_contrib=True)[:, :-1] * a
    cols = list(Xt.columns)
    out = {}
    for j, c in enumerate(cols):
        if c == KEY:
            continue
        bt, bv = Xt[c].isna().to_numpy(), Xv[c].isna().to_numpy()
        if bt.sum() < MIN_TRAIN_BLANK:
            continue
        rec = {"name": c, "formula": FORMULA[NAME[c]], "train": {}, "val": {}}
        for split, y, b, p in (("train", yt, bt, ot), ("val", yv, bv, ov)):
            rec[split] = {tag: {"firms": int(m.sum()), "bankrupt": int(y[m].sum()), "rate": rate(y, m),
                                "wilson_95": E.wilson(int(y[m].sum()), int(m.sum())),
                                "mean_predicted": float(p[m].mean()) if m.any() else None}
                          for tag, m in (("blank", b), ("present", ~b))}
        rec["mean_contribution_when_blank_train"] = float(contrib[bt, j].mean())
        rec["mean_contribution_when_present_train"] = float(contrib[~bt, j].mean())
        rec["criteria"] = evaluate_attr(yt, bt, yv, bv, rec["mean_contribution_when_blank_train"])
        out[NAME[c]] = rec
    return out


def null_false_positives(X_tr, y_tr, X_va, y_va, hist: dict, n_perm: int = N_PERM) -> dict:
    """Expected number of attributes passing all three criteria if labels were unrelated to blanks."""
    ptr = ~X_tr[KEY].isna().to_numpy()
    pva = ~X_va[KEY].isna().to_numpy()
    yt, yv = np.asarray(y_tr)[ptr], np.asarray(y_va)[pva]
    rng = np.random.default_rng(SEED)
    masks = {a: (X_tr.loc[ptr, h["name"]].isna().to_numpy(), X_va.loc[pva, h["name"]].isna().to_numpy()) for a, h in hist.items()}
    hits = {a: 0 for a in hist}
    for _ in range(n_perm):
        pt, pv = rng.permutation(yt), rng.permutation(yv)
        for a, h in hist.items():
            r = evaluate_attr(pt, masks[a][0], pv, masks[a][1], h["mean_contribution_when_blank_train"])
            hits[a] += int(r["blocked"])
    per = {a: n / n_perm for a, n in hits.items()}
    return {"permutations": n_perm, "candidates": len(hist), "expected_false_positives": float(sum(per.values())),
            "per_attribute": per}


# --------------------------------------------------------------------------- ceiling, cost
def cost_and_ceiling(X_tr, y_tr, X_va, y_va, blocked: list, hist: dict, limit: int = A.MAX_OTHER_BLANKS) -> dict:
    out = {"blocked": blocked}
    crossed = []
    for split, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
        y = np.asarray(y)
        present = ~X[KEY].isna().to_numpy()
        other = X.drop(columns=[KEY]).isna()
        n_other = other.sum(axis=1).to_numpy()
        within = present & (n_other <= limit)
        n_pres, bk_pres = int(present.sum()), int(y[present].sum())
        per = {}
        any_blank = np.zeros(len(X), dtype=bool)
        for a in blocked:
            m = within & X[hist[a]["name"]].isna().to_numpy()
            any_blank |= m
            m_all = present & X[hist[a]["name"]].isna().to_numpy()
            per[a] = {"firms_new": int(m.sum()), "bankrupt_new": int(y[m].sum()), "firms_any": int(m_all.sum()),
                      "bankrupt_any": int(y[m_all].sum())}
        refused_count_limit = present & (n_other > limit)
        out[split] = {"attr21_present_firms": n_pres, "attr21_present_bankrupt": bk_pres, "per_attribute": per,
                      "new_refusals": int(any_blank.sum()), "new_refusals_bankrupt": int(y[any_blank].sum()),
                      "new_refusals_firm_share": float(any_blank.sum() / n_pres),
                      "new_refusals_bankrupt_share": float(y[any_blank].sum() / bk_pres),
                      "count_limit_refusals": int(refused_count_limit.sum()),
                      "cumulative_firm_share": float((any_blank | refused_count_limit).sum() / n_pres),
                      "cumulative_bankrupt_share": float(y[any_blank | refused_count_limit].sum() / bk_pres)}
        if out[split]["new_refusals_firm_share"] > CEIL_FIRMS:
            crossed.append(f"{split}: new refusals are {out[split]['new_refusals_firm_share']:.1%} of Attr21-present firms (ceiling {CEIL_FIRMS:.0%})")
        if out[split]["new_refusals_bankrupt_share"] > CEIL_BANKRUPT:
            crossed.append(f"{split}: new refusals hold {out[split]['new_refusals_bankrupt_share']:.1%} of Attr21-present bankrupt firms (ceiling {CEIL_BANKRUPT:.0%})")
    out["ceiling"] = {"firm_share": CEIL_FIRMS, "bankrupt_share": CEIL_BANKRUPT, "crossed": bool(crossed), "reasons": crossed}
    return out


# -------------------------------------------------------------------------- count limit
def count_groups(X, y, p, bands, limit_unused=None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    present = ~X[KEY].isna().to_numpy()
    n_other = X.drop(columns=[KEY]).isna().sum(axis=1).to_numpy()
    out = {}
    for label, lo, hi in GROUPS:
        m = present & (n_other >= lo) & (n_other <= hi)
        if m.sum() == 0 or y[m].sum() == 0:
            out[label] = {"firms": int(m.sum()), "bankrupt": int(y[m].sum()), "evaluable": False}
            continue
        b = subset_block(y[m], p[m], bands, N_BOOT)
        out[label] = {"firms": b["firms"], "bankrupt": b["bankrupt"], "evaluable": True, "pr_auc": b["pr_auc"], "pr_auc_ci": b["pr_auc_ci"],
                      "high_flagged": b["high"]["flagged"], "high_precision": b["high"]["precision"], "high_precision_ci": b["high"]["precision_ci"]}
    return out


def count_limit_decision(tr: dict, va: dict) -> dict:
    """Condition from the protocol, for the 3 to 6 group against the 0-blank group."""
    g, z = "3 to 6 blanks", "0 blanks"
    ok = tr[g].get("evaluable") and tr[z].get("evaluable") and va[g].get("evaluable") and va[z].get("evaluable")
    if not ok:
        return {"met": False, "evaluable": False}
    pr_below = tr[g]["pr_auc_ci"][1] < tr[z]["pr_auc_ci"][0]
    hp_below = (tr[g]["high_precision_ci"][1] < tr[z]["high_precision_ci"][0]) if None not in (tr[g]["high_precision_ci"][1], tr[z]["high_precision_ci"][0]) \
        and not any(np.isnan(x) for x in (tr[g]["high_precision_ci"][1], tr[z]["high_precision_ci"][0])) else False
    val_lower = bool(va[g]["pr_auc"] < va[z]["pr_auc"] and (va[g]["high_precision"] or 0) < (va[z]["high_precision"] or 0))
    return {"evaluable": True, "pr_auc_ci_below": bool(pr_below), "high_precision_ci_below": bool(hp_below),
            "validation_points_lower": val_lower, "met": bool(pr_below and hp_below and val_lower)}


# ----------------------------------------------------------------------------- stress test
def stress(X, model, calibrator, bands) -> list:
    present = ~X[KEY].isna().to_numpy()
    Xp = X[present].reset_index(drop=True)
    p0 = calibrator.transform(model.predict_proba(Xp)[:, 1])
    b0 = C.assign_bands(p0, bands["t_watch"], bands["t_high"])
    rows = []
    for c in X.columns:
        have = ~Xp[c].isna().to_numpy()
        if have.sum() == 0:
            continue
        X1 = Xp.copy()
        X1.loc[have, c] = np.nan
        p1 = calibrator.transform(model.predict_proba(X1)[:, 1])
        b1 = C.assign_bands(p1, bands["t_watch"], bands["t_high"])
        m = have
        d = logit(p1[m]) - logit(p0[m])
        rows.append({"attr": NAME[c], "firms_with_value": int(m.sum()), "changed_band": int((b0[m] != b1[m]).sum()),
                     "changed_share": float((b0[m] != b1[m]).mean()), "mean_logodds_change": float(d.mean()),
                     "mean_abs_logodds_change": float(np.abs(d).mean())})
    return rows


# ------------------------------------------------------------------------- Attr27 deep dive
def strata_table(X, y, p, med_attr2) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    present = ~X[KEY].isna().to_numpy()
    blank = X[COL["Attr27"]].isna().to_numpy()
    a22, a2 = X[COL["Attr22"]], X[COL["Attr2"]]
    s22 = np.where(a22.isna(), "Attr22 blank", np.where(a22 > 0, "Attr22 > 0", "Attr22 <= 0"))
    s2 = np.where(a2.isna(), "Attr2 blank", np.where(a2 > med_attr2, "Attr2 above train median", "Attr2 at or below train median"))
    out = {}
    for title, s in (("Attr22 sign", s22), ("Attr2 leverage", s2)):
        rows = []
        for lab in sorted(set(s)):
            r = {"stratum": lab}
            for tag, m in (("blank", blank), ("present", ~blank)):
                mm = present & (s == lab) & m
                k, n = int(y[mm].sum()), int(mm.sum())
                r[tag] = {"firms": n, "bankrupt": k, "rate": k / n if n else None, "wilson_95": E.wilson(k, n),
                          "mean_predicted": float(p[mm].mean()) if n else None}
            rows.append(r)
        out[title] = rows
    return out


def co_blank(X, top: int = 8) -> list:
    present = ~X[KEY].isna().to_numpy()
    Xp = X[present]
    blank = Xp[COL["Attr27"]].isna()
    rows = []
    for c in Xp.columns:
        if c in (KEY, COL["Attr27"]):
            continue
        n = int((Xp.loc[blank, c].isna()).sum())
        if n:
            rows.append({"attr": NAME[c], "blank_with_attr27": n, "share_of_attr27_blank": n / int(blank.sum()),
                         "share_blank_when_attr27_present": float(Xp.loc[~blank, c].isna().mean())})
    return sorted(rows, key=lambda r: -r["share_of_attr27_blank"])[:top]


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def ridge_blank(X, y) -> dict:
    present = ~X[KEY].isna().to_numpy()
    Xp, yp = X[present].reset_index(drop=True), np.asarray(y)[present]
    blank = Xp[COL["Attr27"]].isna().to_numpy().astype(float)
    cov = [c for c in Xp.columns if c not in (KEY, COL["Attr27"]) and Xp[c].isna().mean() < REG_MAX_BLANK_RATE]
    keep = ~Xp[cov].isna().any(axis=1).to_numpy()
    out = {"covariates": len(cov), "rows_before": int(len(Xp)), "rows_dropped": int((~keep).sum()),
           "dropped_share": float((~keep).mean())}
    if out["dropped_share"] > REG_MAX_DROP:
        out["evaluable"] = False
        return out
    Z = signed_log(Xp.loc[keep, cov].to_numpy(dtype=float))
    Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) == 0, 1, Z.std(axis=0))
    D = np.column_stack([blank[keep], Z])
    yk = yp[keep]

    def fit(d, yy):
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED).fit(d, yy).coef_[0, 0]

    coef = float(fit(D, yk))
    idx = C.bootstrap_indices(yk, N_BOOT, SEED)
    draws = [fit(D[i], yk[i]) for i in idx]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out.update({"evaluable": True, "rows_used": int(keep.sum()), "blank_firms_used": int(blank[keep].sum()),
                "coef": coef, "coef_ci": [float(lo), float(hi)], "odds_ratio": float(np.exp(coef)),
                "odds_ratio_ci": [float(np.exp(lo)), float(np.exp(hi))]})
    return out


def deep_dive(X_tr, y_tr, X_va, y_va, p_tr, p_va) -> dict:
    ptr = ~X_tr[KEY].isna().to_numpy()
    med = float(X_tr.loc[ptr, COL["Attr2"]].median())
    return {"attr2_train_median": med, "co_blank_train": co_blank(X_tr), "co_blank_val": co_blank(X_va),
            "strata_train": strata_table(X_tr, y_tr, p_tr, med), "strata_val": strata_table(X_va, y_va, p_va, med),
            "ridge_train": ridge_blank(X_tr, y_tr)}


# ----------------------------------------------------------------------------------- rule
def rule_payload(hist: dict, cost: dict, null: dict, count_dec: dict, decision: dict) -> dict:
    blocked = decision["blocked_set"]
    return {"phase": 9, "criteria": {"ratio_up": RATIO_UP, "ratio_down": RATIO_DOWN, "min_abs_contribution": MIN_CONTRIB,
                                     "validation_ratio_up": VAL_RATIO_UP, "validation_ratio_down": VAL_RATIO_DOWN,
                                     "min_train_blank_firms": MIN_TRAIN_BLANK, "min_validation_blank_firms": MIN_VAL_BLANK},
            "candidates": {a: {"direction": h["criteria"]["direction"], "c1": h["criteria"]["c1"], "c2": h["criteria"]["c2"],
                               "c3": h["criteria"]["c3"], "meets_all": h["criteria"]["blocked"]} for a, h in hist.items()},
            "meets_criteria": [a for a, h in hist.items() if h["criteria"]["blocked"]],
            "ceiling_crossed": cost["ceiling"]["crossed"], "count_limit_condition_met": count_dec["met"],
            "applied": decision["applied"], "blocked": blocked, "count_limit": A.MAX_OTHER_BLANKS,
            "status": decision["status"],
            "expected_false_positives_under_null": null["expected_false_positives"]}


def decide(hist: dict, cost: dict, count_dec: dict) -> dict:
    meets = [a for a, h in hist.items() if h["criteria"]["blocked"]]
    if cost["ceiling"]["crossed"]:
        return {"applied": False, "blocked_set": [], "status": "stopped: coverage ceiling crossed, user decides"}
    if count_dec["met"]:
        return {"applied": False, "blocked_set": [], "status": "stopped: count-limit condition met, user decides"}
    return {"applied": True, "blocked_set": meets, "status": "applied" if meets else "no attribute meets the criteria, rule unchanged"}


# --------------------------------------------------------------------------------- report
def ci(v, c, nd=3):
    return f"{f3(v, nd)} ({f3(c[0], nd)} to {f3(c[1], nd)})"


def cell(b, key="rate"):
    return f"{b['bankrupt']} of {b['firms']} ({f3(b['rate'])})"


def make_flags(hist, cost, null, count_dec, decision, ridge) -> list:
    meets = [a for a, h in hist.items() if h["criteria"]["blocked"]]
    flags = []
    flags.append(f"{len(hist)} attributes are blank in at least {MIN_TRAIN_BLANK} Attr21-present train firms. {len(meets)} meet all three pre-registered criteria: "
                 f"{', '.join(meets) if meets else 'none'}. Under the null, {null['expected_false_positives']:.2f} would be expected by chance "
                 f"({null['permutations']} label permutations). Validation is reused, so these are not described as validated.")
    c = cost["ceiling"]
    flags.append("Coverage ceiling " + (f"CROSSED ({'; '.join(c['reasons'])}). The phase stops after this report and the app is unchanged." if c["crossed"]
                                         else "not crossed.") + f" Decision: {decision['status']}.")
    flags.append("Count limit of 6: " + ("condition met, the phase stops for the user." if count_dec["met"] else
                 ("not evaluable." if not count_dec["evaluable"] else "the 3 to 6 group does not meet the pre-registered condition, the limit stays at 6.")))
    if ridge.get("evaluable"):
        flags.append(f"Attr27 blank, ridge regression with {ridge['covariates']} low-blank covariates: odds ratio {ridge['odds_ratio']:.2f} "
                     f"({ridge['odds_ratio_ci'][0]:.2f} to {ridge['odds_ratio_ci'][1]:.2f}), {ridge['rows_dropped']} rows dropped. An association only, it does not feed the rule.")
    else:
        flags.append(f"Attr27 regression not evaluable: {ridge['dropped_share']:.1%} of rows would be dropped.")
    flags.append("A blocked blank is a guard against trusting a score driven by a blank. It is not a claim that the model is wrong on those firms: "
                 "see the observed rate against the mean predicted probability among blank firms in the table.")
    flags.append("The test split is spent, so no rule change can be checked on untouched data. Blanks in real use are probably not random like these.")
    flags.append("Test split untouched in this phase. No model, calibrator or cut was changed.")
    return flags


def render(res: dict) -> str:
    hist, cost, null, cnt, dec = res["history"], res["cost"], res["null"], res["count_groups"], res["decision"]
    L = ["# Phase 9: how blank attributes change the score", "",
         "Generated by scripts/run_phase9_blanks.py. Train and validation only, frozen model, calibrator and cuts unchanged, test split not read. "
         "Criteria, ceiling and deep-dive design were fixed in docs/decisions.md (Phase 9 protocol) before this ran.", "",
         f"Decision: **{dec['status']}**.", "",
         "## 1. History per candidate attribute (Attr21-present firms)", "",
         f"Candidates: attributes blank in at least {MIN_TRAIN_BLANK} Attr21-present train firms. Criteria: (1) train blank rate at least {RATIO_UP:.0f}x the present rate "
         f"with the Wilson bound clear of it (or at most 1/3, mirror); (2) mean contribution when blank at least {MIN_CONTRIB:.1f} in the same direction; "
         f"(3) validation ratio at least {VAL_RATIO_UP:.0f}x (or at most {VAL_RATIO_DOWN:.1f}x) with at least {MIN_VAL_BLANK} blank validation firms (else not evaluable). "
         "Mean predicted is the train out-of-fold or validation calibrated probability among those firms.", "",
         "| Attribute | Train blank: bankrupt | Train present | Train mean predicted (blank) | Val blank: bankrupt | Val present | Val mean predicted (blank) | Mean contribution blank | (1) | (2) | (3) | Meets all |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, h in hist.items():
        t, v, cr = h["train"], h["val"], h["criteria"]
        c3 = "not evaluable" if cr["c1"] and cr["c3"] is None else ("n/a" if not cr["c1"] else str(cr["c3"]))
        L.append(f"| {a} | {cell(t['blank'])} | {f3(t['present']['rate'])} | {f3(t['blank']['mean_predicted'])} | {cell(v['blank'])} | {f3(v['present']['rate'])} | "
                 f"{f3(v['blank']['mean_predicted'])} | {h['mean_contribution_when_blank_train']:+.2f} | {cr['direction'] or 'no'} | {cr['c2']} | {c3} | {cr['blocked']} |")
    L += ["", "## 2. Multiple-testing exposure", "",
          f"{null['candidates']} candidates. Expected number meeting all three criteria under the null (labels permuted within train and within validation, "
          f"{null['permutations']} permutations, seed {SEED}): {null['expected_false_positives']:.3f}. Validation was already used for model choice, band cuts and the Phase 8 flag, "
          "so criterion 3 is not independent. An attribute meeting the criteria is not described as validated.", ""]
    nz = {a: p for a, p in null["per_attribute"].items() if p > 0}
    L.append("Per-attribute chance of passing under the null (non-zero only): " + (", ".join(f"{a} {p:.3f}" for a, p in nz.items()) if nz else "none") + ".")
    L += ["", "## 3. Cost of blocking and the coverage ceiling", "",
          f"New refusals are Attr21-present firms with at most {A.MAX_OTHER_BLANKS} other blanks and at least one attribute meeting the criteria blank. "
          f"Ceiling: more than {CEIL_FIRMS:.0%} of Attr21-present firms or more than {CEIL_BANKRUPT:.0%} of their bankrupt firms, in train or validation.", ""]
    if cost["blocked"]:
        L += ["| Split | Attribute | Firms refused (new) | Bankrupt (new) | Firms blank (any) | Bankrupt (any) |", "|---|---|---|---|---|---|"]
        for s in ("train", "val"):
            for a, r in cost[s]["per_attribute"].items():
                L.append(f"| {s} | {a} | {r['firms_new']} | {r['bankrupt_new']} | {r['firms_any']} | {r['bankrupt_any']} |")
        L += ["", "| Split | Attr21-present firms | Bankrupt | New refusals | Share of firms | Bankrupt refused | Share of bankrupt | Count-limit refusals | Cumulative firm share | Cumulative bankrupt share |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in ("train", "val"):
            c = cost[s]
            L.append(f"| {s} | {c['attr21_present_firms']} | {c['attr21_present_bankrupt']} | {c['new_refusals']} | {c['new_refusals_firm_share']:.1%} | {c['new_refusals_bankrupt']} | "
                     f"{c['new_refusals_bankrupt_share']:.1%} | {c['count_limit_refusals']} | {c['cumulative_firm_share']:.1%} | {c['cumulative_bankrupt_share']:.1%} |")
    else:
        L.append("No attribute meets all three criteria, so no block applies.")
    L += ["", f"Ceiling: {'CROSSED. ' + '; '.join(cost['ceiling']['reasons']) if cost['ceiling']['crossed'] else 'not crossed.'}", ""]
    L += ["## 4. Count limit (more than 6 other blanks refused)", "",
          "Firms with Attr21 present, by number of other blank attributes. PR AUC and High precision with 95% bootstrap CIs.", "",
          "| Scores | Group | Firms | Bankrupt | PR AUC (95% CI) | High flagged | High precision (95% CI) |", "|---|---|---|---|---|---|---|"]
    for s, name in (("train_oof", "train out-of-fold"), ("val", "validation")):
        for g, r in cnt[s].items():
            if r.get("evaluable"):
                L.append(f"| {name} | {g} | {r['firms']} | {r['bankrupt']} | {ci(r['pr_auc'], r['pr_auc_ci'])} | {r['high_flagged']} | {ci(r['high_precision'], r['high_precision_ci'])} |")
            else:
                L.append(f"| {name} | {g} | {r['firms']} | {r['bankrupt']} | not evaluable | | |")
    cd = res["count_decision"]
    L += ["", "Condition for 3 to 6 blanks against 0 blanks: PR AUC CI entirely below "
          f"{cd.get('pr_auc_ci_below')}, High precision CI entirely below {cd.get('high_precision_ci_below')}, validation points lower "
          f"{cd.get('validation_points_lower')}. " + ("Limit stays at 6." if not cd["met"] else "Limit is a candidate for change, user decides."), "",
          "## 5. Single-blank stress test (descriptive, does not feed the rule)", "",
          "Blank one attribute at a time for Attr21-present firms that have it, with the frozen model. Change is in calibrated log-odds. Largest mean absolute change first (top 15 of 64).", "",
          "| Attribute | Firms | Changed band | Share | Mean change | Mean absolute change |", "|---|---|---|---|---|---|"]
    for s, name in (("train", "Train"), ("val", "Validation")):
        L.append(f"| **{name}** | | | | | |")
        for r in sorted(res["stress"][s], key=lambda r: -r["mean_abs_logodds_change"])[:15]:
            L.append(f"| {r['attr']} | {r['firms_with_value']} | {r['changed_band']} | {r['changed_share']:.1%} | {r['mean_logodds_change']:+.2f} | {r['mean_abs_logodds_change']:.2f} |")
    dd = res["deep_dive"]
    L += ["", "## 6. Attr27 deep dive (descriptive, does not feed the rule)", "",
          f"Attr2 (total liabilities / total assets) cut at the train median among Attr21-present firms: {dd['attr2_train_median']:.4f}.", ""]
    for key, name in (("strata_train", "Train (mean predicted is out-of-fold)"), ("strata_val", "Validation")):
        L += [f"### {name}", "", "| Split by | Stratum | Attr27 blank: bankrupt | Mean predicted | Attr27 present: bankrupt | Mean predicted |", "|---|---|---|---|---|---|"]
        for title, rows in dd[key].items():
            for r in rows:
                b, p = r["blank"], r["present"]
                L.append(f"| {title} | {r['stratum']} | {b['bankrupt']} of {b['firms']} ({f3(b['rate'])}; {f3(b['wilson_95'][0])} to {f3(b['wilson_95'][1])}) | {f3(b['mean_predicted'])} | "
                         f"{p['bankrupt']} of {p['firms']} ({f3(p['rate'])}; {f3(p['wilson_95'][0])} to {f3(p['wilson_95'][1])}) | {f3(p['mean_predicted'])} |")
        L.append("")
    L += ["Attributes most often blank together with Attr27 (Attr21-present firms):", "",
          "| Split | Attribute | Blank with Attr27 | Share of Attr27-blank firms | Share blank when Attr27 present |", "|---|---|---|---|---|"]
    for key, name in (("co_blank_train", "train"), ("co_blank_val", "validation")):
        for r in dd[key]:
            L.append(f"| {name} | {r['attr']} | {r['blank_with_attr27']} | {r['share_of_attr27_blank']:.3f} | {r['share_blank_when_attr27_present']:.3f} |")
    rg = dd["ridge_train"]
    L += ["", "Ridge logistic regression (train, C = 1.0): Attr27-blank indicator plus every other attribute with a blank rate under "
              f"{REG_MAX_BLANK_RATE:.0%} among Attr21-present firms, signed-log transformed and standardised, no imputation.", ""]
    if rg.get("evaluable"):
        L.append(f"{rg['covariates']} covariates, {rg['rows_used']} of {rg['rows_before']} rows used ({rg['rows_dropped']} dropped for a blank covariate), "
                 f"{rg['blank_firms_used']} Attr27-blank firms. Indicator coefficient {rg['coef']:+.3f} (bootstrap 95% CI {rg['coef_ci'][0]:+.3f} to {rg['coef_ci'][1]:+.3f}), "
                 f"odds ratio {rg['odds_ratio']:.2f} ({rg['odds_ratio_ci'][0]:.2f} to {rg['odds_ratio_ci'][1]:.2f}). An association only, not a cause.")
    else:
        L.append(f"Not evaluable: {rg['dropped_share']:.1%} of rows would be dropped, above the {REG_MAX_DROP:.0%} limit.")
    L += ["", "## Flags", "", *[f"{i}. {f}" for i, f in enumerate(res["flags"], 1)], ""]
    return NL.join(L)


# ----------------------------------------------------------------------------------- run
def run(X_tr, y_tr, X_va, y_va, model, calibrator, bands, reports=REPORTS, n_perm: int = N_PERM) -> dict:
    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    oof = calibrator.transform(C.oof_predictions(make_lgbm, X_tr, y_tr))
    p_va = calibrator.transform(model.predict_proba(X_va)[:, 1])
    hist = history(X_tr, y_tr, X_va, y_va, model, calibrator, oof, p_va)
    null = null_false_positives(X_tr, y_tr, X_va, y_va, hist, n_perm)
    meets = [a for a, h in hist.items() if h["criteria"]["blocked"]]
    cost = cost_and_ceiling(X_tr, y_tr, X_va, y_va, meets, hist)
    groups = {"train_oof": count_groups(X_tr, y_tr, oof, bands), "val": count_groups(X_va, y_va, p_va, bands)}
    cdec = count_limit_decision(groups["train_oof"], groups["val"])
    decision = decide(hist, cost, cdec)
    st = {"train": stress(X_tr, model, calibrator, bands), "val": stress(X_va, model, calibrator, bands)}
    dd = deep_dive(X_tr, y_tr, X_va, y_va, oof, p_va)
    flags = make_flags(hist, cost, null, cdec, decision, dd["ridge_train"])
    res = {"history": hist, "null": null, "cost": cost, "count_groups": groups, "count_decision": cdec, "decision": decision,
           "stress": st, "deep_dive": dd, "flags": flags, "seed": SEED, "bootstrap_n": N_BOOT,
           "bands": {"t_watch": bands["t_watch"], "t_high": bands["t_high"]}}
    rule = rule_payload(hist, cost, null, cdec, decision)
    (reports / "phase9_blanks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (reports / "phase9_blank_rule.json").write_text(json.dumps(rule, indent=1), encoding="utf-8")
    (reports / "phase9_blanks.md").write_text(render(res), encoding="utf-8")
    return res


def main() -> None:
    np.random.seed(SEED)
    verify_frozen()
    art = C.load_artifacts(ROOT / "artifacts" / "calibration")
    bands = art["bands"]["lightgbm"]
    calibrator = art["calibrators"]["lightgbm"][bands["selected_method"]]
    model = joblib.load(ROOT / "models" / "lightgbm.joblib")
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    res = run(X_tr, y_tr, X_va, y_va, model, calibrator, bands)
    print(f"wrote reports/phase9_blanks.md and phase9_blank_rule.json; decision: {res['decision']['status']}")


if __name__ == "__main__":
    main()
