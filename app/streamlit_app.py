"""solvency-signal: a screening signal for corporate financial distress (Streamlit app).

Run from the repo root: streamlit run app/streamlit_app.py

Scores firms with the frozen LightGBM + Platt model. Policy in src/app_logic.py and docs/decisions.md (Phase 8).
Every figure shown is read from generated files. Uploaded data is processed in memory and is not written to disk by this app.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import app_logic as A  # noqa: E402

BLUE, VERMILLION = "#0072B2", "#D55E00"
CLAIM = ("A screening signal for firms like those in the Polish bankruptcy data. Not lending advice. "
         "Trained and checked on a random split only, so nothing is claimed about future years.")
CITATION = ("Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5F600")
ATTRIBUTION = (f"Data: {CITATION} Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
               "This project changed the data: exact duplicates dropped, a stratified random 70/15/15 split, and model scores derived from it.")
PRIVACY = "Uploaded data is processed in memory and is not written to disk by this app."

st.set_page_config(page_title="solvency-signal", layout="wide")


@st.cache_resource
def get_frozen():
    return A.load_frozen()


@st.cache_data
def get_json(rel: str) -> dict:
    return A.load_json(rel)


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def with_ci(v, c, nd=3):
    return f"{fmt(v, nd)} ({fmt(c[0], nd)} to {fmt(c[1], nd)})"


try:
    frozen = get_frozen()
    numbers = get_json("reports/phase8_app_numbers.json")
except A.FrozenArtifactError as exc:
    st.title("solvency-signal")
    st.error(f"The frozen model files do not match their recorded hashes, so scoring is disabled. {exc}")
    st.stop()
try:
    rule = A.load_rule()
    BLOCKED = rule["blocked"]
    phase9 = get_json("reports/phase9_blanks.json")
except (A.BlankRuleError, OSError, ValueError) as exc:
    st.title("solvency-signal")
    st.error(f"The blank-attribute rule could not be read, so scoring is disabled. {exc}")
    st.stop()
REFUSED = A.refused_blank_names(BLOCKED)

st.title("solvency-signal")
st.write(CLAIM)
tab_score, tab_valid, tab_drivers, tab_limits = st.tabs(
    ["Score a firm", "How it was checked", "What drives the model", "Limitations and data"])


# ------------------------------------------------------------------------------ score a firm
def show_result(res_row: pd.Series, frame_row: pd.DataFrame, demo_label: str = "") -> None:
    if res_row["status"] != "scored":
        st.error(f"Not scored. {res_row['reason']}")
        st.caption("No band and no probability are shown for firms that are not scored.")
        return
    band = res_row["band"]
    bands = frozen["bands"]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Risk band", band)
        st.metric("Calibrated probability", f"{res_row['probability']:.3f}")
        st.caption(f"Cuts: Watch from {bands['t_watch']:.4f}, High from {bands['t_high']:.4f}.")
        if demo_label:
            st.caption(f"Demo only: in the dataset this firm was {demo_label}.")
    with right:
        if res_row["warning"]:
            st.warning(res_row["warning"])
            hist_lines = A.blank_history(numbers, res_row["blank_attrs"])
            if hist_lines:
                with st.expander("What the training and validation data show for these blanks (descriptive, not a test)"):
                    for line in hist_lines:
                        st.write("- " + line)
        st.markdown(f"**What the {band} band meant historically** (firms with Attr21 present, bankrupt within the year):")
        hist = pd.DataFrame([{"Scores": a, "Firms": b, "Bankrupt": c, "Rate (95% interval)": with_ci(d, (e, f))}
                             for a, b, c, d, e, f in A.band_history(numbers, band)])
        st.dataframe(hist, hide_index=True)
        if band == "High":
            st.caption(A.floor_statement(numbers))
        with st.expander("About the probability"):
            p = numbers["numbers"]
            st.write(f"The probability is calibrated on all firms. For firms with Attr21 present the mean predicted probability was "
                     f"{p['val']['present']['mean_predicted']:.4f} against an observed rate of {p['val']['present']['observed_rate']:.4f} on validation, and "
                     f"{p['train_oof']['present']['mean_predicted']:.4f} against {p['train_oof']['present']['observed_rate']:.4f} on train out-of-fold. "
                     "That check is reported only and nothing was recalibrated.")
    ex = A.explain_native(frozen, frame_row)
    st.markdown("**What pushes this score up or down** (largest drivers, in calibrated log-odds; positive raises the score):")
    df = pd.DataFrame([{"driver": d["attr"] + (" [U]" if d["unverified"] else ""), "contribution": d["contribution"],
                        "direction": d["direction"]} for d in ex["drivers"]])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("contribution:Q", title="Contribution to the score"), y=alt.Y("driver:N", sort=None, title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises", "lowers"], range=[VERMILLION, BLUE]), legend=alt.Legend(title=None)))
    st.altair_chart(chart, use_container_width=True)
    for d in ex["drivers"]:
        st.write("- " + d["sentence"])
    st.caption("[U] and UNVERIFIED mark a definition the feature dictionary flags as possibly mistranslated, unclear or inconsistent. "
               "Drivers describe what the model does, not what causes distress.")


with tab_score:
    st.subheader("Score a firm")
    src = st.radio("Input", ["Demo firm", "Upload CSV"], horizontal=True)
    frame, problems, labels = None, {}, {}
    if src == "Demo firm":
        demo = A.demo_firms()
        names = [f"{f['id']} ({f['role']})" for f in demo["firms"]]
        pick = st.selectbox("Demo firm", names)
        i = names.index(pick)
        frame = A.demo_frame(demo).iloc[[i]]
        labels = {frame.index[0]: demo["firms"][i]["dataset_label"]}
        st.caption(demo["rule"])
    else:
        st.write("Upload a CSV with one row per firm and all 64 attributes, columns named Attr1 to Attr64 (or the dictionary names). "
                 "Blank cells are allowed, but a firm with Attr21 blank"
                 + (f" or {REFUSED} blank" if BLOCKED else "")
                 + f" is not scored, and a firm with more than {A.MAX_OTHER_BLANKS} other blank attributes is not scored. "
                 f"At most {A.MAX_ROWS} rows.")
        st.download_button("Download a header-only template", A.template_csv(), "solvency_signal_template.csv", "text/csv")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                parsed = A.parse_upload(up.getvalue())
                frame, problems = parsed["frame"], parsed["problems"]
                if parsed["ignored_columns"]:
                    st.info(f"Ignored columns: {', '.join(parsed['ignored_columns'][:10])}")
            except ValueError as exc:
                st.error(str(exc))
    st.caption(PRIVACY)
    if frame is not None:
        results = A.score_frame(frozen, frame, problems, BLOCKED)
        if len(results) > 1:
            show = results[["row", "status", "band", "probability", "n_other_blank", "reason"]].copy()
            show["probability"] = show["probability"].round(3)
            st.dataframe(show, hide_index=True)
            row_no = st.selectbox("Show details for row", list(results["row"]))
            sel = results[results["row"] == row_no].iloc[0]
        else:
            sel = results.iloc[0]
        show_result(sel, frame.loc[[sel["index"]]], labels.get(sel["index"], ""))


# ------------------------------------------------------------------------------ how it was checked
def metric_table(block: dict) -> pd.DataFrame:
    rows = [("Firms", f"{block['firms']}"), ("Bankrupt within the year", f"{block['bankrupt']}"),
            ("No-skill PR AUC (share bankrupt)", fmt(block["no_skill_pr_auc"])),
            ("PR AUC (95% CI)", with_ci(block["pr_auc"], block["pr_auc_ci"])),
            ("ROC AUC (95% CI)", with_ci(block["roc_auc"], block["roc_auc_ci"])),
            ("High band precision (95% CI)", with_ci(block["high"]["precision"], block["high"]["precision_ci"])),
            ("High band recall (95% CI)", with_ci(block["high"]["recall"], block["high"]["recall_ci"])),
            ("Watch plus High recall (95% CI)", with_ci(block["watch_plus_high"]["recall"], block["watch_plus_high"]["recall_ci"]))]
    return pd.DataFrame(rows, columns=["Measure", "Value"])


with tab_valid:
    st.subheader("How the model was checked")
    st.write("Measured on the Polish data, on a random split. The headline for this app is firms with Attr21 present, because those are "
             "the firms it scores.")
    for key, title in (("val", "Validation split"), ("train_oof", "Train, out-of-fold")):
        st.markdown(f"**{title}**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Firms with Attr21 present (what this app scores)")
            st.dataframe(metric_table(numbers["numbers"][key]["present"]), hide_index=True)
        with c2:
            st.markdown("All firms, beside it")
            st.dataframe(metric_table(numbers["numbers"][key]["all"]), hide_index=True)
    st.info(A.flattered_sentence(numbers))
    st.write(A.floor_statement(numbers))
    sc = phase9.get("scorable")
    if sc:
        st.markdown(f"**Firms this app scores under all its blank rules** (Attr21 present, {REFUSED} not blank, at most {A.MAX_OTHER_BLANKS} other blanks)")
        c1, c2 = st.columns(2)
        for col, key, title in ((c1, "val", "Validation split"), (c2, "train_oof", "Train, out-of-fold")):
            b = sc[key]
            with col:
                st.markdown(f"{title}: {b['refused_firms']} of {b['all_firms']} firms refused, {b['refused_bankrupt']} of them bankrupt")
                st.dataframe(metric_table(b), hide_index=True)
        st.caption("Report only. The refusal rule was chosen after looking at train and validation, the cuts and calibration were not changed, "
                   "and train out-of-fold scores use a calibrator fitted on those same scores. The High band's precision here is read against the 0.60 floor "
                   "the cut was built for, and no cut was moved to meet it.")
    st.markdown("**Historical bankrupt rate per band, firms with Attr21 present**")
    rows = []
    for label, k in (("Validation", "val"), ("Train, out-of-fold", "train_oof")):
        for r in numbers["numbers"][k]["present"]["band_table"]:
            rows.append({"Scores": label, "Band": r["band"], "Firms": r["firms"], "Bankrupt": r["bankrupt"],
                         "Rate (95% interval)": with_ci(r["bankrupt_rate"], r["wilson_95"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Validation cuts were fitted on validation, so validation precision and recall at the cuts are in-sample. "
               "Train out-of-fold is the independent check. The Watch band is small, so its rate carries little information.")
    st.markdown("**Calibration on firms with Attr21 present (reported only, nothing was recalibrated)**")
    cal = pd.DataFrame([{"Scores": lab, "Mean predicted": fmt(numbers["numbers"][k]["present"]["mean_predicted"], 4),
                         "Observed rate": fmt(numbers["numbers"][k]["present"]["observed_rate"], 4),
                         "ECE (95% CI)": with_ci(numbers["numbers"][k]["present"]["ece"], numbers["numbers"][k]["present"]["ece_ci"], 4)}
                        for lab, k in (("Validation", "val"), ("Train, out-of-fold", "train_oof"))])
    st.dataframe(cal, hide_index=True)
    st.image(str(ROOT / "reports" / "figures" / "phase8_reliability_present.png"))
    t = get_json("reports/phase5_test_results.json")
    h = t["models"]["lightgbm"]
    st.markdown("**One-shot test split (all firms, read once, no breakdown by Attr21 is possible)**")
    st.write(f"PR AUC {with_ci(h['metrics']['pr_auc'], h['ci']['pr_auc'])} against a no-skill {t['prevalence']:.3f}; "
             f"High band precision {with_ci(h['band_stats']['high']['precision'], h['ci']['high_precision'])}. "
             "These are all-firms figures, so the same caution about missing Attr21 applies.")


# ------------------------------------------------------------------------------ what drives the model
with tab_drivers:
    st.subheader("What drives the model")
    st.write("SHAP values for the frozen model on train and validation, in calibrated log-odds. They describe what the model does, "
             "not what causes distress. Correlated ratios split credit between them, so a ratio with a close twin looks less important than it is.")
    st.image(str(ROOT / "reports" / "figures" / "phase6_importance.png"))
    st.image(str(ROOT / "reports" / "figures" / "phase6_categories.png"))
    st.caption("[U] marks a definition the feature dictionary flags as possibly mistranslated, unclear or inconsistent (UNVERIFIED). "
               "The five categories are assigned by this project from each formula. The dataset gives none.")
    p7 = get_json("reports/phase7_attr21.json")["counts"]
    st.markdown("**Attr21 (sales growth)**")
    st.write(f"In the training data, {p7['train']['missing']['bankrupt']} of {p7['train']['missing']['firms']} firms with Attr21 blank are bankrupt, "
             f"against {p7['train']['present']['bankrupt']} of {p7['train']['present']['firms']} with it present. The model treats a blank as a very strong signal, "
             "so this app does not score firms with Attr21 blank. The cause of the pattern is not known.")
    if BLOCKED:
        st.markdown(f"**Blank {REFUSED}**")
        for a_, reason in BLOCKED.items():
            st.write(f"- {reason}")
        st.caption("These refusals follow criteria fixed before the check and met on train and validation. They are not described as validated, "
                   "and the test split cannot confirm them. A refusal guards against a score driven by a blank. It is not a claim that the model is wrong on those firms.")


# ------------------------------------------------------------------------------ limitations and data
with tab_limits:
    st.subheader("Limitations and data")
    refused_bullet = (f"- Firms with {REFUSED} blank are not scored. In the training data those blanks were tied to a very different bankrupt rate, "
                      "and the score would reflect the blank. See the model page for the numbers.") if BLOCKED else ""
    st.markdown(f"""
- {CLAIM}
- The Polish data has no company id and no year, so the split is random and cannot show how the model behaves on later years.
- The sector composition of the data is not stated on the UCI page (UNVERIFIED). Do not assume it fits a particular industry or country.
- Firms with Attr21 (sales growth) blank are not scored. Attr21 needs last year's sales.
{refused_bullet}
- Other blank attributes are scored with LightGBM's native handling. The effect of those blanks on the score has not been tested, and a firm with more than {A.MAX_OTHER_BLANKS} other blank attributes is not scored.
- Bands were set on all firms. On firms with Attr21 present the High band was less precise than the cut aimed for. See the checks page.
- The model, calibration and cuts are frozen. They were not tuned on the test split.
- A band is a screening signal for a closer look. It is not a decision and not lending advice.
- {PRIVACY}
""")
    st.markdown("**Data and attribution**")
    st.write(ATTRIBUTION)
    st.write("Dataset page: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data")
    st.markdown("**Frozen files verified at start-up**")
    st.caption("SHA256 of the model and calibration files matched artifacts/phase5_protocol.json: "
               + ", ".join(f"{k.split('/')[-1]} {v[:12]}" for k, v in frozen["hashes"].items()))

st.divider()
st.caption(ATTRIBUTION)
