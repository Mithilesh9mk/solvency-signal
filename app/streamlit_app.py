"""solvency-signal: a screening signal for corporate financial distress (Streamlit app).

Run from the repo root: streamlit run app/streamlit_app.py

Scores firms with the frozen LightGBM + Platt model. Policy in src/app_logic.py and docs/decisions.md (Phase 8).
Presentation (tokens, CSS, HTML pieces) lives in src/app_ui.py. Every figure shown is read from generated files.
Uploaded data is processed in memory and is not written to disk by this app.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import app_logic as A  # noqa: E402
from src import app_ui as U  # noqa: E402
from src import quick as Q  # noqa: E402

CLAIM = ("A screening signal for firms like those in the Polish bankruptcy data. Not lending advice. "
         "Trained and checked on a random split only, so nothing is claimed about future years.")
CITATION = ("Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5F600")
ATTRIBUTION = (f"Data: {CITATION} Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
               "This project changed the data: exact duplicates dropped, a stratified random 70/15/15 split, and model scores derived from it.")
PRIVACY = "Uploaded data is processed in memory and is not written to disk by this app."
TAB_NAMES = ["Quick check: 4 ratios", "Full model: 64 ratios", "Evidence", "Limits and data"]

st.set_page_config(page_title="solvency-signal", layout="wide")
st.markdown(f"<style>{U.CSS}</style>", unsafe_allow_html=True)


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def get_frozen():
    return A.load_frozen()


@st.cache_resource
def get_quick():
    return Q.load()


@st.cache_data
def get_json(rel: str) -> dict:
    return A.load_json(rel)


def masthead() -> None:
    md(U.masthead("Screening tool, not lending advice",
                  "An early-warning screen for corporate bankruptcy risk. Enter a few ratios and see how often similar "
                  "Polish firms failed within a year.",
                  "Trained and checked on a random split of the Polish bankruptcy data, so nothing is claimed about future years."))


masthead()
try:
    with st.spinner("Loading the frozen model and checking its hashes"):
        frozen = get_frozen()
        numbers = get_json("reports/phase8_app_numbers.json")
except A.FrozenArtifactError as exc:
    st.error(f"Scoring is switched off. The frozen model files do not match their recorded hashes. {exc} "
             "Restore the files with python -m src.train and reload.")
    st.stop()
try:
    rule = A.load_rule()
    BLOCKED = rule["blocked"]
    phase9 = get_json("reports/phase9_blanks.json")
except (A.BlankRuleError, OSError, ValueError) as exc:
    st.error(f"Scoring is switched off. The blank-attribute rule could not be read. {exc} "
             "Rebuild it with python -m scripts.run_phase9_blanks and reload.")
    st.stop()
REFUSED = A.refused_blank_names(BLOCKED)

md(U.chooser([("Have four ratios?", "Use the Quick check. A 1 to 10 score in under a minute, from current ratio, debt to equity, net margin and cash earnings."),
              ("Have all 64 attributes?", "Use the Full model. The headline model, for one demo firm or a CSV of up to "
                                          f"{A.MAX_ROWS} firms, with drivers and the history of each band.")]))
tab_quick, tab_full, tab_evidence, tab_limits = st.tabs(TAB_NAMES)


# ------------------------------------------------------------------------------ quick check
QUICK_BAND_LEGEND = [(name, f"{name} ({lo} to {hi})") for lo, hi, name in Q.BANDS]


def quick_strip(score: int) -> str:
    cells = [str(s) for s in range(1, 11)]
    return (U.strip(cells, score - 1, [Q.band_for(s) for s in range(1, 11)], QUICK_BAND_LEGEND)
            + '<div class="ss-legend"><span>1 is the safest score and 10 the highest risk.</span></div>')


def plain_value(key: str, v: float) -> str:
    """What a typed ratio means in everyday terms, so people see their own number reflected back."""
    if key == "net_profit_margin":
        return f"{v * 100:.1f}% of revenue is left as profit"
    if key == "cash_earnings_ratio":
        return f"{v * 100:.1f}% of total liabilities"
    if key == "debt_to_equity":
        return "no debt" if v == 0 else ("negative equity" if v < 0 else f"{v:.2f} of debt for each 1 of equity")
    return f"{v:.2f} of current assets for each 1 of current liabilities"


def reset_quick() -> None:
    for spec in Q.INPUTS:
        st.session_state["q_" + spec["key"]] = float(spec["default"])


def show_quick(p: dict) -> None:
    md(U.section_head("Quick check", "Four ratios, plain numbers: 1.5 for a current ratio of 1.5, 0.05 for a 5% margin."))
    bounds = Q.input_bounds(p)
    left, right = st.columns([5, 6], gap="large")
    vals, slots = {}, {}
    with left:
        md(U.step(1, "Enter the ratios"))
        grid = st.columns(2) + st.columns(2)  # two rows, so the reading and tab order stays row by row, also when stacked
        for i, spec in enumerate(Q.INPUTS):
            lo, hi = bounds[spec["key"]]
            tip = f"{spec['formula']}. Higher is {spec['higher']}."
            if spec["key"] == "debt_to_equity":
                tip += " Enter a negative number if equity is negative, and 0 if there is no debt."
            with grid[i]:
                vals[spec["key"]] = st.number_input(spec["label"], value=float(spec["default"]), step=0.05, format="%.4f",
                                                    key="q_" + spec["key"], help=tip)
                slots[spec["key"]] = st.empty()
    changed = any(abs(vals[s["key"]] - s["default"]) > 1e-12 for s in Q.INPUTS)
    try:
        out = Q.assess(vals, p)
        err = ""
    except ValueError as exc:
        out, err = None, str(exc)
    flagged = set(out["outside_range"]) if out else set()
    for spec in Q.INPUTS:
        lo, hi = bounds[spec["key"]]
        span = f"Most training firms: {lo:.3g} to {hi:.3g}." if lo is not None and hi is not None else ""
        outside = spec["feature"] in flagged
        text = f"Outside the usual range. {span}" if outside else span
        slots[spec["key"]].markdown((U.hint(plain_value(spec["key"], vals[spec["key"]])) if out else "") + U.hint(text.strip(), outside),
                                    unsafe_allow_html=True)
    with left:
        if changed:
            st.button("Reset to the example values", on_click=reset_quick, key="q_reset")
        else:
            st.caption("These are example values. Replace them with your firm's ratios.")
    with right:
        md(U.step(2, "Read the result"))
        if out is None:
            md(U.callout("err", "Not scored", f"{err}. Enter a number in every field. Use 0 for no debt, and a negative number if equity is negative. "
                                              "Your other entries are kept."))
        else:
            show_quick_result(out, p, changed)
    if out is not None:
        md(U.step(3, "See what drove it and how far to trust it"))
        why, trust = st.columns(2, gap="large")
        with why:
            md(U.eyebrow("What pushes this score up or down"))
            md(U.driver_bars([{"label": Q.LABEL_BY_FEATURE[n].split(" (")[0], "value": c} for n, c in out["contributions"].items()]))
            st.caption("Calibrated log-odds. Positive raises the score. These describe what the model does, not what causes distress.")
        with trust:
            show_quick_trust()
    with st.expander("What each score meant in the Polish data"):
        rows = [{"Score": i + 1, "Band": Q.band_for(i + 1), "Firms": r["n"], "Bankrupt": r["bankrupt"],
                 "Rate (95% interval)": U.pct_ci(r["rate"], r["wilson"])} for i, r in enumerate(p["score_rates"])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Training firms, scored out of fold. The scale ranks firms within the Polish training data in tenths. "
                   "It is not an absolute risk.")
    st.caption("The cash earnings ratio is a proxy, (net profit + depreciation) / total liabilities, not operating cash flow / current "
               "liabilities. Debt is read as total liabilities. Fitted on Polish firms: for other countries it is a rough screen only, "
               "and the range covers sampling noise, not a different country, year or ratio definition. Not lending advice.")


def show_quick_result(out: dict, p: dict, changed: bool) -> None:
    ob = out["observed"]
    md(U.eyebrow("Result for your values" if changed else "Result for the example values"))
    lead = (f"<b>{ob['bankrupt']} of {ob['n']}</b> Polish training firms with this score went bankrupt within a year "
            f"({U.pct(ob['rate'])}).")
    sub = (f"95% range {U.pct(ob['range'][0])} to {U.pct(ob['range'][1])}. "
           f"Model probability {U.pct(out['probability'])}. The 1 to 10 score ranks firms in tenths of the training data.")
    md(U.verdict(str(out["score"]), "of 10", out["band"], lead, sub, kind="quick"))
    md(quick_strip(out["score"]))
    if out["outside_range"]:
        names = ", ".join(Q.LABEL_BY_FEATURE[n] for n in out["outside_range"])
        md(U.callout("warn", "Treat this score with care",
                     f"Outside the range of most training firms: {names}. The model treats extreme values as the nearest value it has seen, "
                     "so the score is least reliable here."))


def show_quick_trust() -> None:
    v = get_json("reports/phase10_quick.json")["val"]
    q, f = v["quick"], v["full_64"]
    md(U.eyebrow("How far to trust four ratios"))
    md(U.tiles([U.tile("Quick check", U.num(q["pr_auc"]), f"PR AUC, 95% CI {U.num(q['pr_auc_ci'][0])} to {U.num(q['pr_auc_ci'][1])}", lead=True),
                U.tile("Full 64 ratios", U.num(f["pr_auc"]), f"PR AUC, 95% CI {U.num(f['pr_auc_ci'][0])} to {U.num(f['pr_auc_ci'][1])}"),
                U.tile("No skill", U.num(q["positive_rate"]), "PR AUC of a model that guesses")]))
    st.caption("Validation firms. Four ratios are much less informative than all 64. Use the Full model tab when all 64 are available.")


with tab_quick:
    try:
        show_quick(get_quick())
    except (Q.QuickModelError, OSError, ValueError) as exc:
        st.error(f"The quick check could not be loaded, so it is switched off. {exc} Rebuild it with python -m scripts.run_phase10_quick "
                 "and reload. The Full model tab is unaffected.")


# ------------------------------------------------------------------------------ full model
def full_strip(band: str) -> str:
    names = ["Low", "Watch", "High"]
    return U.strip(names, names.index(band), names, [(n, n) for n in names])


def show_result(res_row: pd.Series, frame_row: pd.DataFrame, demo_label: str = "") -> None:
    if res_row["status"] != "scored":
        md(U.callout("err", "Not scored", f"{res_row['reason']} No band and no probability are shown for firms that are not scored. "
                                          "Fill the blank attributes and upload again, or use the Quick check, which needs only four ratios."))
        return
    band = res_row["band"]
    bands = frozen["bands"]
    hist = A.band_history(numbers, band)
    label, n, bk, rate, lo, hi = hist[0]
    lead = (f"<b>{bk} of {n}</b> firms scored {band} went bankrupt within a year on {label} scores ({U.pct(rate)}).")
    sub = (f"95% interval {U.pct(lo)} to {U.pct(hi)}. Calibrated probability {U.pct(res_row['probability'], 2)}. "
           f"Cuts: Watch from {U.pct(bands['t_watch'], 2)}, High from {U.pct(bands['t_high'], 2)}.")
    md(U.eyebrow("Result"))
    md(U.verdict(band, "risk band", band, lead, sub, word=True, kind="full"))
    md(full_strip(band))
    if demo_label:
        st.caption(f"Demo only: in the dataset this firm was {demo_label}.")
    if res_row["warning"]:
        md(U.callout("warn", "Some attributes are blank", res_row["warning"]))
        blanks = A.blank_history(numbers, res_row["blank_attrs"])
        if blanks:
            with st.expander("What the training and validation data show for these blanks (descriptive, not a test)"):
                for line in blanks:
                    st.write("- " + line)
    with st.expander(f"What the {band} band meant historically (firms with Attr21 present)"):
        st.dataframe(pd.DataFrame([{"Scores": a, "Firms": b, "Bankrupt": c, "Rate (95% interval)": U.pct_ci(d, (e, f))}
                                   for a, b, c, d, e, f in hist]), hide_index=True, width="stretch")
        if band == "High":
            st.caption(A.floor_statement(numbers))
    with st.expander("About the probability"):
        p = numbers["numbers"]
        st.write(f"The probability is calibrated on all firms. For firms with Attr21 present the mean predicted probability was "
                 f"{U.pct(p['val']['present']['mean_predicted'], 2)} against an observed rate of {U.pct(p['val']['present']['observed_rate'], 2)} on validation, and "
                 f"{U.pct(p['train_oof']['present']['mean_predicted'], 2)} against {U.pct(p['train_oof']['present']['observed_rate'], 2)} on train out-of-fold. "
                 "That check is reported only and nothing was recalibrated.")
    ex = A.explain_native(frozen, frame_row)
    md(U.step(3, "See what drove it"))
    md(U.eyebrow("What pushes this score up or down"))
    md(U.driver_bars([{"label": (d["label"] or d["attr"]) + (" [U]" if d["unverified"] else ""), "sub": d["attr"], "value": d["contribution"]}
                      for d in ex["drivers"]]))
    st.caption("Largest drivers, in calibrated log-odds; positive raises the score. [U] and UNVERIFIED mark a definition the feature "
               "dictionary flags as possibly mistranslated, unclear or inconsistent. Drivers describe what the model does, not what causes distress.")
    with st.expander("Each driver in words"):
        for d in ex["drivers"]:
            st.write("- " + d["sentence"])


def upload_problem(exc: Exception) -> str:
    """A file-level parse error as one short sentence. A long list of missing columns is counted, not printed."""
    msg = str(exc).strip()
    if msg.startswith("missing columns:"):
        names = [c.strip() for c in msg.split(":", 1)[1].split(". ")[0].split(",")]
        shown = ", ".join(names[:4]) + (" and others" if len(names) > 4 else "")
        return f"{len(names)} of the 64 attribute columns are missing, for example {shown}. The file needs all 64, Attr1 to Attr64."
    return (msg[:1].upper() + msg[1:]).rstrip(".") + "."


def band_cell(v: str) -> str:
    return f"background-color:{U.BAND_COLORS[v]}55" if v in U.BAND_COLORS else ""


def show_batch_summary(results: pd.DataFrame) -> None:
    scored = results[results["status"] == "scored"]
    counts = scored["band"].value_counts()
    tl = [U.tile("Firms", str(len(results)), f"{len(scored)} scored, {len(results) - len(scored)} not scored", lead=True)]
    tl += [U.tile(b, str(int(counts.get(b, 0))), "firms") for b in ("High", "Watch", "Low")]
    md(U.tiles(tl))


with tab_full:
    md(U.section_head("Full model", "The headline model: LightGBM with Platt calibration, frozen before the test split was read."))
    md(U.step(1, "Choose a firm"))
    src = st.radio("Input", ["Demo firm", "Upload CSV"], horizontal=True, label_visibility="collapsed")
    frame, problems, labels = None, {}, {}
    if src == "Demo firm":
        demo = A.demo_firms()
        names = [f"{f['id']} ({f['role']})" for f in demo["firms"]]
        pick = st.selectbox("Demo firm", names)
        i = names.index(pick)
        frame = A.demo_frame(demo).iloc[[i]]
        labels = {frame.index[0]: demo["firms"][i]["dataset_label"]}
        with st.expander("How these demo firms were picked"):
            st.write(demo["rule"])
    else:
        st.write("One row per firm, all 64 attributes, columns named Attr1 to Attr64 (or the dictionary names).")
        md(U.callout("info", "What gets scored",
                     "Blank cells are allowed, but a firm with Attr21 blank"
                     + (f" or {REFUSED} blank" if BLOCKED else "")
                     + f" is not scored, and a firm with more than {A.MAX_OTHER_BLANKS} other blank attributes is not scored. "
                     f"At most {A.MAX_ROWS} rows."))
        st.download_button("Download a header-only template", A.template_csv(), "solvency_signal_template.csv", "text/csv")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is None:
            md(U.empty_state("Your results appear here", "Upload a CSV to score up to "
                             f"{A.MAX_ROWS} firms. {PRIVACY}"))
        else:
            try:
                parsed = A.parse_upload(up.getvalue())
                frame, problems = parsed["frame"], parsed["problems"]
                if parsed["ignored_columns"]:
                    st.info(f"Ignored columns: {', '.join(parsed['ignored_columns'][:10])}")
            except ValueError as exc:
                md(U.callout("err", "This file could not be read",
                             f"{upload_problem(exc)} Fix the file and upload it again, or start from the template above."))
            except Exception:  # noqa: BLE001 - a malformed file must never take the page down
                md(U.callout("err", "This file could not be read",
                             "It does not look like a CSV. Save it as comma-separated text and upload it again, or start from the template above."))
    st.caption(PRIVACY)
    if frame is not None:
        results = A.score_frame(frozen, frame, problems, BLOCKED)
        md(U.step(2, "Read the result"))
        if len(results) > 1:
            show_batch_summary(results)
            show = results[["row", "status", "band", "probability", "n_other_blank", "reason"]].copy()
            show["probability"] = show["probability"].map(lambda x: "" if pd.isna(x) else U.pct(x, 2))
            show.columns = ["Row", "Status", "Band", "Probability", "Other blanks", "Reason"]
            st.dataframe(show.style.map(band_cell, subset=["Band"]), hide_index=True, width="stretch")
            row_no = st.selectbox("Show details for row", list(results["row"]))
            sel = results[results["row"] == row_no].iloc[0]
        else:
            sel = results.iloc[0]
        show_result(sel, frame.loc[[sel["index"]]], labels.get(sel["index"], ""))


# ------------------------------------------------------------------------------ evidence
def metric_table(block: dict) -> pd.DataFrame:
    rows = [("Firms", f"{block['firms']}"), ("Bankrupt within the year", f"{block['bankrupt']}"),
            ("No-skill PR AUC (share bankrupt)", U.num(block["no_skill_pr_auc"])),
            ("PR AUC (95% CI)", U.num_ci(block["pr_auc"], block["pr_auc_ci"])),
            ("ROC AUC (95% CI)", U.num_ci(block["roc_auc"], block["roc_auc_ci"])),
            ("High band precision (95% CI)", U.pct_ci(block["high"]["precision"], block["high"]["precision_ci"])),
            ("High band recall (95% CI)", U.pct_ci(block["high"]["recall"], block["high"]["recall_ci"])),
            ("Watch plus High recall (95% CI)", U.pct_ci(block["watch_plus_high"]["recall"], block["watch_plus_high"]["recall_ci"]))]
    return pd.DataFrame(rows, columns=["Measure", "Value"])


def headline_tiles(block: dict, prevalence: float = None) -> str:
    """The three figures that answer 'does it beat guessing, and what do the bands catch?'. All values come from block."""
    skill = block.get("no_skill_pr_auc", prevalence)
    ci = block["pr_auc_ci"]
    return U.tiles([
        U.tile("PR AUC", U.num(block["pr_auc"]), f"95% CI {U.num(ci[0])} to {U.num(ci[1])}. No skill scores {U.num(skill)}.", lead=True),
        U.tile("High band precision", U.pct(block["high"]["precision"], 0),
               f"of firms flagged High went bankrupt. 95% CI {U.pct(block['high']['precision_ci'][0], 0)} to {U.pct(block['high']['precision_ci'][1], 0)}."),
        U.tile("Watch plus High recall", U.pct(block["watch_plus_high"]["recall"], 0),
               f"of bankrupt firms were flagged Watch or High. 95% CI {U.pct(block['watch_plus_high']['recall_ci'][0], 0)} to "
               f"{U.pct(block['watch_plus_high']['recall_ci'][1], 0)}.")])


def show_checked() -> None:
    n = numbers["numbers"]
    md(U.section_head("How the model was checked", "Measured on the Polish data, on a random split. The headline is firms with Attr21 present, "
                                                     "because those are the firms the app scores."))
    md(U.eyebrow("Validation split, firms with Attr21 present"))
    md(headline_tiles(n["val"]["present"]))
    md(U.callout("warn", "Read the all-firms figures with care", A.flattered_sentence(numbers)))
    st.write(A.floor_statement(numbers))
    t = get_json("reports/phase5_test_results.json")
    h = t["models"]["lightgbm"]
    md(U.eyebrow("One-shot test split, all firms, read once"))
    md(U.tiles([U.tile("PR AUC", U.num(h["metrics"]["pr_auc"]),
                       f"95% CI {U.num(h['ci']['pr_auc'][0])} to {U.num(h['ci']['pr_auc'][1])}. No skill scores {U.num(t['prevalence'])}.", lead=True),
                U.tile("High band precision", U.pct(h["band_stats"]["high"]["precision"], 0),
                       f"95% CI {U.pct(h['ci']['high_precision'][0], 0)} to {U.pct(h['ci']['high_precision'][1], 0)}.")]))
    st.caption("These are all-firms figures, and no breakdown by Attr21 is possible on the test split, so the same caution about "
               "missing Attr21 applies.")
    with st.expander("All measures, validation and train out-of-fold"):
        for key, title in (("val", "Validation split"), ("train_oof", "Train, out-of-fold")):
            st.markdown(f"**{title}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("Firms with Attr21 present (what this app scores)")
                st.dataframe(metric_table(n[key]["present"]), hide_index=True, width="stretch")
            with c2:
                st.markdown("All firms, beside it")
                st.dataframe(metric_table(n[key]["all"]), hide_index=True, width="stretch")
    sc = phase9.get("scorable")
    if sc:
        with st.expander(f"Firms this app scores under all its blank rules (Attr21 present, {REFUSED} not blank, at most {A.MAX_OTHER_BLANKS} other blanks)"):
            c1, c2 = st.columns(2)
            for col, key, title in ((c1, "val", "Validation split"), (c2, "train_oof", "Train, out-of-fold")):
                b = sc[key]
                with col:
                    st.markdown(f"**{title}**: {b['refused_firms']} of {b['all_firms']} firms refused, {b['refused_bankrupt']} of them bankrupt")
                    st.dataframe(metric_table(b), hide_index=True, width="stretch")
            st.caption("Report only. The refusal rule was chosen after looking at train and validation, the cuts and calibration were not changed, "
                       "and train out-of-fold scores use a calibrator fitted on those same scores. The High band's precision here is read against the 60% floor "
                       "the cut was built for, and no cut was moved to meet it.")
    with st.expander("Historical bankrupt rate per band, firms with Attr21 present"):
        rows = []
        for label, k in (("Validation", "val"), ("Train, out-of-fold", "train_oof")):
            for r in n[k]["present"]["band_table"]:
                rows.append({"Scores": label, "Band": r["band"], "Firms": r["firms"], "Bankrupt": r["bankrupt"],
                             "Rate (95% interval)": U.pct_ci(r["bankrupt_rate"], r["wilson_95"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Validation cuts were fitted on validation, so validation precision and recall at the cuts are in-sample. "
                   "Train out-of-fold is the independent check. The Watch band is small, so its rate carries little information.")
    with st.expander("Calibration on firms with Attr21 present (reported only, nothing was recalibrated)"):
        cal = pd.DataFrame([{"Scores": lab, "Mean predicted": U.pct(n[k]["present"]["mean_predicted"], 2),
                             "Observed rate": U.pct(n[k]["present"]["observed_rate"], 2),
                             "ECE (95% CI)": U.num_ci(n[k]["present"]["ece"], n[k]["present"]["ece_ci"], 4)}
                            for lab, k in (("Validation", "val"), ("Train, out-of-fold", "train_oof"))])
        st.dataframe(cal, hide_index=True, width="stretch")
        st.image(str(ROOT / "reports" / "figures" / "phase8_reliability_present.png"))


def show_drivers() -> None:
    md(U.section_head("What drives the model", "SHAP values for the frozen model on train and validation, in calibrated log-odds. "
                                                 "They describe what the model does, not what causes distress."))
    st.write("Correlated ratios split credit between them, so a ratio with a close twin looks less important than it is.")
    st.image(str(ROOT / "reports" / "figures" / "phase6_importance.png"))
    st.image(str(ROOT / "reports" / "figures" / "phase6_categories.png"))
    st.caption("[U] marks a definition the feature dictionary flags as possibly mistranslated, unclear or inconsistent (UNVERIFIED). "
               "The five categories are assigned by this project from each formula. The dataset gives none.")
    p7 = get_json("reports/phase7_attr21.json")["counts"]
    md(U.callout("warn", "Attr21 (sales growth) is why some firms are not scored",
                 f"In the training data, {p7['train']['missing']['bankrupt']} of {p7['train']['missing']['firms']} firms with Attr21 blank are bankrupt, "
                 f"against {p7['train']['present']['bankrupt']} of {p7['train']['present']['firms']} with it present. The model treats a blank as a very strong signal, "
                 "so this app does not score firms with Attr21 blank. The cause of the pattern is not known."))
    if BLOCKED:
        st.markdown(f"**Blank {REFUSED}**")
        for reason in BLOCKED.values():
            st.write(f"- {reason}")
        st.caption("These refusals follow criteria fixed before the check and met on train and validation. They are not described as validated, "
                   "and the test split cannot confirm them. A refusal guards against a score driven by a blank. It is not a claim that the model is wrong on those firms.")


with tab_evidence:
    view = st.segmented_control("Evidence", ["How it was checked", "What drives the model"], default="How it was checked",
                                label_visibility="collapsed") or "How it was checked"
    try:
        show_checked() if view == "How it was checked" else show_drivers()
    except (OSError, KeyError, ValueError) as exc:
        md(U.callout("err", "This evidence page could not be loaded",
                     f"{exc} Regenerate it with python -m scripts.run_phase8_prep and python -m scripts.run_phase6, then reload. Scoring is unaffected."))


# ------------------------------------------------------------------------------ limitations and data
with tab_limits:
    md(U.section_head("Limitations and data", "What this tool is, what it is not, and where the data comes from."))
    refused_bullet = (f"- Firms with {REFUSED} blank are not scored. In the training data those blanks were tied to a very different bankrupt rate, "
                      "and the score would reflect the blank. See the Evidence tab for the numbers.") if BLOCKED else ""
    st.markdown("**Scope**")
    st.markdown(f"""
- {CLAIM}
- The Polish data has no company id and no year, so the split is random and cannot show how the model behaves on later years.
- The sector composition of the data is not stated on the UCI page (UNVERIFIED). Do not assume it fits a particular industry or country.
- A band is a screening signal for a closer look. It is not a decision and not lending advice.
""")
    st.markdown("**Blanks**")
    st.markdown(f"""
- Firms with Attr21 (sales growth) blank are not scored. Attr21 needs last year's sales.
{refused_bullet}
- Other blank attributes are scored with LightGBM's native handling. The effect of those blanks on the score has not been tested, and a firm with more than {A.MAX_OTHER_BLANKS} other blank attributes is not scored.
""")
    st.markdown("**Model and bands**")
    st.markdown(f"""
- Bands were set on all firms. On firms with Attr21 present the High band was less precise than the cut aimed for. See the Evidence tab.
- The model, calibration and cuts are frozen. They were not tuned on the test split.
- {PRIVACY}
""")
    st.markdown("**Data and attribution**")
    st.write(ATTRIBUTION)
    st.write("Dataset page: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data")
    with st.expander("Frozen files verified at start-up"):
        st.caption("SHA256 of the model and calibration files matched artifacts/phase5_protocol.json: "
                   + ", ".join(f"{k.split('/')[-1]} {v[:12]}" for k, v in frozen["hashes"].items()))

md(f'<div class="ss-foot">{U.esc(ATTRIBUTION)}</div>')
