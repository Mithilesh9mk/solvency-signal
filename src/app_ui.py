"""Presentation pieces for the Streamlit app: design tokens, CSS and small HTML builders.

No model logic and no numbers live here. Every figure is passed in from generated files by the app, and these
functions only format it. All dynamic text is HTML-escaped. Strings are built without newlines or indentation so
Markdown never turns them into code blocks.
"""

from html import escape

# Design tokens. One radius pair (6, 12), one 1px stroke, no shadows, one serif for display figures and one sans for text.
INK, MUTED, RULE, PAPER, CARD = "#17202A", "#5B6470", "#DAD4C6", "#F6F3EC", "#FFFDF8"
BLUE, VERMILLION = "#0072B2", "#D55E00"  # colour-blind-safe pair for "lowers" and "raises"
# Band colours (Okabe-Ito). Used as fills with dark text, never as text colour on paper.
BAND_COLORS = {"Low": "#56B4E9", "Watch": "#F0E442", "Elevated": "#E69F00", "High": VERMILLION}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');
:root{--ink:#17202A;--muted:#5B6470;--rule:#DAD4C6;--paper:#F6F3EC;--card:#FFFDF8;--soft:#ECE7DA;--r1:6px;--r2:12px;
--serif:'Newsreader',Georgia,'Times New Roman',serif;--sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp,[data-testid="stMarkdownContainer"],.stApp input,.stApp textarea,.stApp button,.stApp label{font-family:var(--sans)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stDecoration"]{display:none}
[data-testid="stMainBlockContainer"],.block-container{max-width:1080px;padding:2.2rem 1.25rem 3rem}
.stApp h1,.stApp h2,.stApp h3{font-family:var(--serif);font-weight:500;letter-spacing:-.01em;color:var(--ink)}
.stApp p,.stApp li{line-height:1.55}
[data-testid="stTabs"] [role="tab"]{font-size:15px;padding:10px 4px;margin-right:18px;color:var(--muted);font-weight:500}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--ink);font-weight:600}
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid var(--rule)}
[data-testid="stExpander"]{border:1px solid var(--rule);border-radius:var(--r2);background:var(--card)}
[data-testid="stExpander"] summary{font-weight:500}
[data-testid="stNumberInput"] input{font-variant-numeric:tabular-nums}
.stButton>button,.stDownloadButton>button{border-radius:var(--r1);border:1px solid var(--ink);font-weight:500;min-height:44px}
[data-testid="stCaptionContainer"]{color:var(--muted)}
.ss-mast{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:6px}
.ss-word{font:600 30px/1.1 var(--serif);letter-spacing:-.02em}
.ss-word i{font-style:normal;color:var(--muted);font-weight:400}
.ss-chip{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);border:1px solid var(--rule);
border-radius:99px;padding:3px 10px;background:var(--card)}
.ss-lede{font:400 22px/1.35 var(--serif);max-width:46em;margin:10px 0 4px;color:var(--ink)}
.ss-scope{font-size:14px;color:var(--muted);max-width:56em;margin:0 0 18px}
.ss-choose{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 20px}
.ss-choose div{border:1px solid var(--rule);background:var(--card);border-radius:var(--r2);padding:12px 14px;font-size:14px;line-height:1.45}
.ss-choose b{display:block;font-size:15px;margin-bottom:2px}
.ss-head{margin:6px 0 14px}
.ss-head h2{font:500 26px/1.2 var(--serif);margin:0 0 4px;padding:0}
.ss-head p{color:var(--muted);font-size:15px;margin:0;max-width:48em}
.ss-step{display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px;margin:18px 0 8px}
.ss-step span{display:inline-flex;width:24px;height:24px;border-radius:99px;background:var(--ink);color:#fff;font-size:13px;
align-items:center;justify-content:center}
.ss-eyebrow{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:2px 0 8px;font-weight:500}
.ss-verdict{display:flex;gap:20px;background:var(--card);border:1px solid var(--rule);border-radius:var(--r2);padding:18px;flex-wrap:wrap}
.ss-score{min-width:132px;border-radius:var(--r1);display:flex;flex-direction:column;align-items:center;justify-content:center;
padding:14px 18px;color:var(--ink);text-align:center}
.ss-score .n{font:600 64px/1 var(--serif);font-variant-numeric:tabular-nums}
.ss-score .n.word{font-size:40px}
.ss-score .of{font-size:13px;margin-top:4px;font-weight:500}
.ss-vbody{flex:1;min-width:220px;display:flex;flex-direction:column;justify-content:center;gap:8px}
.ss-lead{font:400 19px/1.4 var(--serif);margin:0}
.ss-lead b{font-weight:600}
.ss-sub{font-size:13px;color:var(--muted);margin:0}
.ss-pill{display:inline-flex;align-items:center;gap:7px;font-size:14px;font-weight:600;padding:3px 11px 3px 9px;border-radius:99px;
border:1px solid var(--rule);background:var(--paper);width:fit-content}
.ss-pill i{width:10px;height:10px;border-radius:99px;display:inline-block}
.ss-strip{display:flex;gap:3px;margin-top:14px}
.ss-strip div{flex:1;text-align:center;border-radius:var(--r1);color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums;
display:flex;align-items:center;justify-content:center}
.ss-marks{display:flex;gap:3px;height:16px}
.ss-marks div{flex:1;text-align:center;font-size:12px;line-height:16px}
.ss-legend{font-size:13px;color:var(--muted);margin-top:4px;display:flex;flex-wrap:wrap;gap:4px 16px}
.ss-legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.ss-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:8px 0 14px}
.ss-tile{background:var(--card);border:1px solid var(--rule);border-radius:var(--r2);padding:14px 16px}
.ss-tile .k{font-size:13px;color:var(--muted);font-weight:500}
.ss-tile .v{font:500 34px/1.15 var(--serif);margin:2px 0;font-variant-numeric:tabular-nums}
.ss-tile .s{font-size:12.5px;color:var(--muted);line-height:1.4}
.ss-tile.lead{background:var(--ink);border-color:var(--ink)}
.ss-tile.lead .k,.ss-tile.lead .s{color:#D8DEE6}
.ss-tile.lead .v{color:#fff}
.ss-call{border:1px solid var(--rule);border-left-width:4px;border-radius:var(--r1);background:var(--card);padding:12px 14px;
font-size:14.5px;line-height:1.5;margin:10px 0}
.ss-call b.t{display:block;margin-bottom:2px}
.ss-call.warn{border-left-color:#E69F00}
.ss-call.err{border-left-color:#D55E00}
.ss-call.info{border-left-color:#0072B2}
.ss-empty{border:1.5px dashed var(--rule);border-radius:var(--r2);padding:28px 20px;text-align:center;color:var(--muted);font-size:15px;
background:rgba(255,253,248,.5)}
.ss-empty b{display:block;color:var(--ink);font:500 20px var(--serif);margin-bottom:4px}
.ss-hint{font-size:12.5px;color:var(--muted);margin:-8px 0 12px;line-height:1.4}
.ss-hint.out{color:#A64500;font-weight:500}
.ss-bars{display:flex;flex-direction:column;gap:9px;margin:6px 0 4px}
.ss-bar{display:grid;grid-template-columns:minmax(110px,34%) 1fr 86px;gap:10px;align-items:center;font-size:14px}
.ss-bar .lab{line-height:1.25}
.ss-bar .lab small{display:block;color:var(--muted);font-size:12px}
.ss-track{position:relative;height:16px;background:var(--soft);border-radius:4px}
.ss-track:before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--muted)}
.ss-fill{position:absolute;top:0;bottom:0;border-radius:3px}
.ss-val{font-size:13px;font-variant-numeric:tabular-nums;text-align:right;line-height:1.2}
.ss-val small{display:block;color:var(--muted);font-size:12px}
.ss-axis{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin:2px 0 0 calc(34% + 10px);
padding-right:96px}
.ss-foot{font-size:12.5px;color:var(--muted);line-height:1.5;border-top:1px solid var(--rule);padding-top:12px;margin-top:28px}
@media (max-width:640px){
.ss-choose{grid-template-columns:1fr 1fr;gap:8px}.ss-choose .d{display:none}.ss-choose b{margin:0;font-size:14px}
.ss-bar{grid-template-columns:1fr 78px}.ss-bar .lab{grid-column:1/-1}
.ss-axis{margin-left:0;padding-right:88px}
.ss-lede{font-size:19px}.ss-score .n{font-size:52px}
}
"""


def esc(x) -> str:
    return escape(str(x), quote=True)


def pct(x, nd: int = 1) -> str:
    """A proportion as a percentage, for example 0.0402 -> '4.0%'."""
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def pct_ci(v, c, nd: int = 1) -> str:
    return f"{pct(v, nd)} ({pct(c[0], nd)} to {pct(c[1], nd)})"


def num(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def num_ci(v, c, nd: int = 3) -> str:
    return f"{num(v, nd)} ({num(c[0], nd)} to {num(c[1], nd)})"


def masthead(chip: str, lede: str, scope: str) -> str:
    return (f'<div class="ss-mast"><div class="ss-word">solvency<i>-</i>signal</div><span class="ss-chip">{esc(chip)}</span></div>'
            f'<p class="ss-lede">{esc(lede)}</p><p class="ss-scope">{esc(scope)}</p>')


def chooser(items: list) -> str:
    """Two short cards that say which tab fits which situation. items: [(title, text)]."""
    return '<div class="ss-choose">' + "".join(f'<div><b>{esc(t)}</b><span class="d">{esc(s)}</span></div>' for t, s in items) + "</div>"


def section_head(title: str, text: str = "") -> str:
    return f'<div class="ss-head"><h2>{esc(title)}</h2>' + (f"<p>{esc(text)}</p>" if text else "") + "</div>"


def step(n: int, text: str) -> str:
    return f'<div class="ss-step"><span>{n}</span>{esc(text)}</div>'


def eyebrow(text: str) -> str:
    return f'<div class="ss-eyebrow">{esc(text)}</div>'


def band_pill(band: str) -> str:
    return f'<span class="ss-pill"><i style="background:{BAND_COLORS[band]}"></i>{esc(band)}</span>'


def verdict(big: str, small: str, band: str, lead_html: str, sub: str, word: bool = False, kind: str = "") -> str:
    """The result card: a coloured block (score or band), the band pill, one plain sentence and one receding detail line.
    lead_html must already be escaped by the caller. kind ("quick" or "full") and the data attributes let tests read the result."""
    return (f'<div class="ss-verdict" data-kind="{esc(kind)}" data-band="{esc(band)}" data-score="{"" if word else esc(big)}">'
            f'<div class="ss-score" style="background:{BAND_COLORS[band]}">'
            f'<span class="n{" word" if word else ""}">{esc(big)}</span><span class="of">{esc(small)}</span></div>'
            f'<div class="ss-vbody">{band_pill(band)}<p class="ss-lead">{lead_html}</p><p class="ss-sub">{esc(sub)}</p></div></div>')


def strip(cells: list, active: int, bands: list, legend: list) -> str:
    """A row of coloured cells with the active one raised and marked. cells: labels, bands: band name per cell,
    legend: [(band, text)]."""
    c, m = [], []
    for i, (lab, b) in enumerate(zip(cells, bands)):
        on = i == active
        c.append(f'<div style="background:{BAND_COLORS[b]};padding:{14 if on else 10}px 0;font-size:{26 if on else 14}px;'
                 f'opacity:{1 if on else .42}">{esc(lab)}</div>')
        m.append(f'<div>{"&#9650;" if on else ""}</div>')
    lg = "".join(f'<span><i style="background:{BAND_COLORS[b]}"></i>{esc(t)}</span>' for b, t in legend)
    return f'<div class="ss-strip">{"".join(c)}</div><div class="ss-marks">{"".join(m)}</div><div class="ss-legend">{lg}</div>'


def tile(label: str, value: str, sub: str = "", lead: bool = False) -> str:
    return (f'<div class="ss-tile{" lead" if lead else ""}"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div>'
            + (f'<div class="s">{esc(sub)}</div>' if sub else "") + "</div>")


def tiles(items: list) -> str:
    return '<div class="ss-tiles">' + "".join(items) + "</div>"


def callout(kind: str, title: str, text: str) -> str:
    """kind: info, warn or err. Title is bold, text follows. Both are escaped."""
    return f'<div class="ss-call {kind}"><b class="t">{esc(title)}</b>{esc(text)}</div>'


def empty_state(title: str, text: str) -> str:
    return f'<div class="ss-empty"><b>{esc(title)}</b>{esc(text)}</div>'


def hint(text: str, out: bool = False) -> str:
    return f'<div class="ss-hint{" out" if out else ""}">{esc(text)}</div>'


def driver_bars(rows: list) -> str:
    """Diverging bars around zero. rows: [{"label", "sub", "value"}], value in calibrated log-odds. Raises is vermillion and
    to the right, lowers is blue and to the left. The word is always printed too, so colour is never the only cue."""
    top = max((abs(r["value"]) for r in rows), default=0) or 1.0
    out = []
    for r in rows:
        v = r["value"]
        w = abs(v) / top * 50
        left = 50 if v > 0 else 50 - w
        colour = VERMILLION if v > 0 else BLUE
        word = "raises" if v > 0 else "lowers"
        sub = f"<small>{esc(r['sub'])}</small>" if r.get("sub") else ""
        out.append(f'<div class="ss-bar"><div class="lab">{esc(r["label"])}{sub}</div>'
                   f'<div class="ss-track"><div class="ss-fill" style="left:{left:.1f}%;width:{w:.1f}%;background:{colour}"></div></div>'
                   f'<div class="ss-val">{v:+.2f}<small>{word}</small></div></div>')
    return ('<div class="ss-bars">' + "".join(out) + '</div><div class="ss-axis"><span>&larr; lowers the score</span>'
            '<span>raises the score &rarr;</span></div>')
