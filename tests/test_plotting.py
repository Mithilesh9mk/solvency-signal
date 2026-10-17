"""Plot palette: at least 3:1 contrast against the background, distinct styles and markers."""

from src import plotting as P


def test_every_series_color_has_3_to_1_contrast():
    colors = {v[1] for v in P.MODEL_STYLE.values()} | {v[1] for v in P.METHOD_STYLE.values()}
    for c in colors:
        assert P.contrast_ratio(c) >= 3.0, c


def test_series_are_distinct_in_color_style_and_marker():
    for table in (P.MODEL_STYLE, P.METHOD_STYLE):
        assert len({v[1] for v in table.values()}) == len(table)
        assert len({v[2] for v in table.values()}) == len(table)
        assert len({v[3] for v in table.values()}) == len(table)
