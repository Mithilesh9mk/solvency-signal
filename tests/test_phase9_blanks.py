"""Phase 9 probe: criteria, ceiling and decision logic on small synthetic data (no model, no test split)."""

import numpy as np
import pandas as pd

from scripts import run_phase9_blanks as P


def test_crit1_up_down_and_none():
    assert P.crit1(30, 100, 40, 1000) == "up"          # 0.30 vs 0.04
    assert P.crit1(0, 90, 50, 1000) == "down"          # 0 of 90, upper bound below 0.05
    assert P.crit1(5, 100, 50, 1000) is None           # about the same
    assert P.crit1(1, 10, 40, 1000) is None            # 0.10 is under 3 times 0.04
    assert P.crit1(0, 0, 40, 1000) is None and P.crit1(3, 10, 0, 100) is None


def test_crit3_floor_and_direction():
    assert P.crit3("up", 1, 9, 20, 500) is None         # fewer than 10 blank firms: not evaluable
    assert P.crit3("up", 6, 20, 20, 500) is True        # 0.30 vs 0.04
    assert P.crit3("up", 1, 20, 20, 500) is False       # 0.05 vs 0.04
    assert P.crit3("down", 0, 22, 25, 500) is True
    assert P.crit3("down", 4, 22, 25, 500) is False


def test_evaluate_attr_needs_all_three():
    rng = np.random.default_rng(42)
    n = 2000
    btr = np.zeros(n, bool)
    btr[:100] = True
    ytr = np.zeros(n, int)
    ytr[:30] = 1
    ytr[100:140] = 1
    bva = np.zeros(500, bool)
    bva[:20] = True
    yva = np.zeros(500, int)
    yva[:6] = 1
    yva[100:110] = 1
    ok = P.evaluate_attr(ytr, btr, yva, bva, 2.0)
    assert ok["blocked"] and ok["direction"] == "up"
    assert not P.evaluate_attr(ytr, btr, yva, bva, 0.5)["blocked"]       # model effect too small
    bva_small = np.zeros(500, bool)
    bva_small[:5] = True
    r = P.evaluate_attr(ytr, btr, yva, bva_small, 2.0)
    assert r["c3"] is None and not r["blocked"]                           # validation not evaluable


def _frame(n=200):
    return pd.DataFrame({"sales_growth": np.ones(n), "a": np.ones(n), "b": np.ones(n)})


def test_cost_and_ceiling_and_decide():
    X = _frame()
    X.loc[:49, "a"] = np.nan                      # 50 firms blank in a
    y = np.zeros(len(X), int)
    y[:40] = 1                                    # 40 bankrupt, all in the blank group
    y[100:110] = 1                                # 10 bankrupt elsewhere
    hist = {"AttrA": {"name": "a"}}
    cost = P.cost_and_ceiling(X, y, X, y, ["AttrA"], hist)
    assert cost["train"]["new_refusals"] == 50 and cost["train"]["new_refusals_bankrupt"] == 40
    assert cost["ceiling"]["crossed"]            # 25% of firms, 80% of bankrupt
    hist2 = {"AttrA": {"criteria": {"blocked": True}}}
    d = P.decide(hist2, cost, {"met": False})
    assert not d["applied"] and d["blocked_set"] == [] and "ceiling" in d["status"]
    none = P.cost_and_ceiling(X, y, X, y, [], hist)
    assert not none["ceiling"]["crossed"]
    assert P.decide(hist2, none, {"met": True})["applied"] is False
    assert P.decide(hist2, none, {"met": False})["blocked_set"] == ["AttrA"]


def test_count_limit_condition_not_met_when_cis_overlap():
    def g(pr, ci_pr, hp, ci_hp):
        return {"evaluable": True, "pr_auc": pr, "pr_auc_ci": ci_pr, "high_precision": hp, "high_precision_ci": ci_hp}
    tr = {"0 blanks": g(0.5, [0.4, 0.6], 0.5, [0.4, 0.6]), "3 to 6 blanks": g(0.2, [0.05, 0.5], 0.2, [0.0, 0.45])}
    va = {"0 blanks": g(0.5, [0.4, 0.6], 0.5, [0.4, 0.6]), "3 to 6 blanks": g(0.2, [0.05, 0.5], 0.2, [0.0, 0.45])}
    assert not P.count_limit_decision(tr, va)["met"]                       # CI upper 0.5 is not below 0.4
    tr["3 to 6 blanks"] = g(0.2, [0.05, 0.3], 0.2, [0.0, 0.3])
    assert P.count_limit_decision(tr, va)["met"]


def test_signed_log_and_stress_shape():
    x = np.array([-3.0, 0.0, 3.0])
    out = P.signed_log(x)
    assert out[1] == 0 and out[0] == -out[2]


# ------------------------------------------------------------- override and generated files
def test_override_applies_the_set_only_when_the_ceiling_is_the_reason_for_stopping():
    hist = {"AttrA": {"criteria": {"blocked": True}}, "AttrB": {"criteria": {"blocked": False}}}
    crossed = {"ceiling": {"crossed": True}}
    clear = {"ceiling": {"crossed": False}}
    d = P.decide(hist, crossed, {"met": False}, override_ceiling=True)
    assert d["applied"] and d["blocked_set"] == ["AttrA"] and d["ceiling_overridden"] and "overridden" in d["status"]
    d = P.decide(hist, crossed, {"met": False})
    assert not d["applied"] and d["blocked_set"] == [] and not d["ceiling_overridden"]
    d = P.decide(hist, clear, {"met": False}, override_ceiling=True)
    assert d["applied"] and not d["ceiling_overridden"] and d["status"] == "applied"
    d = P.decide(hist, crossed, {"met": True}, override_ceiling=True)                    # the count-limit stop is never overridden
    assert not d["applied"] and "count-limit" in d["status"]
    none = P.decide({"AttrB": {"criteria": {"blocked": False}}}, clear, {"met": False})
    assert none["applied"] and none["blocked_set"] == [] and "unchanged" in none["status"]


def test_generated_phase9_files_agree():
    import json

    from src.preprocess import ROOT

    rule_p, res_p = ROOT / "reports" / "phase9_blank_rule.json", ROOT / "reports" / "phase9_blanks.json"
    if not (rule_p.exists() and res_p.exists()):
        return
    rule, res = json.loads(rule_p.read_text()), json.loads(res_p.read_text())
    assert rule["blocked"] == [a for a, h in res["history"].items() if h["criteria"]["blocked"]] if rule["applied"] else rule["blocked"] == []
    assert set(rule["blocked_detail"]) == set(rule["blocked"])
    assert res["cost"]["ceiling"]["crossed"] == rule["ceiling_crossed"]
    if rule["applied"]:
        for key in ("val", "train_oof"):
            b = res["scorable"][key]
            assert b["firms"] + b["refused_firms"] == b["all_firms"]
