"""Shared plot style for Early Tremor.

Colors come from the Okabe-Ito colorblind-safe palette, limited to the three
hues that keep at least 3:1 contrast against the white background (checked by
`contrast_ratio`, and by tests/test_plotting.py). Every series also has its own
line style and marker, so the plots stay readable in grayscale and for
readers who cannot tell the hues apart.
"""

SURFACE = "#ffffff"
INK = "#000000"
INK2 = "#4d4d4d"
GRID = "#e0e0e0"

BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"

# key -> (label, color, line style, marker)
MODEL_STYLE = {
    "logistic_regression": ("Logistic regression", VERMILLION, "--", "s"),
    "neural_net": ("Neural network", GREEN, "-.", "^"),
    "lightgbm": ("LightGBM", BLUE, "-", "o"),
}
METHOD_STYLE = {
    "none": ("No calibration", INK, ":", "x"),
    "platt": ("Platt", BLUE, "-", "o"),
    "isotonic": ("Isotonic", VERMILLION, "--", "s"),
}


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(fg: str, bg: str = SURFACE) -> float:
    """WCAG contrast ratio between two #rrggbb colors."""
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def apply_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style_axes(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
