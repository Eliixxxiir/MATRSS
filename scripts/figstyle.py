"""Shared colours for the paper figures (see svgplot.py).

A validated categorical palette used in fixed order. Scatter-type charts use at most three hues;
everything else is context grey. Text always uses ink colours, never a series colour. Ordered
quantities (e.g. a memory parameter) use one hue, light to dark (BLUE_RAMP).
"""

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3: blue, orange, aqua
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal, light -> dark
CONTEXT = "#c3c2b7"                             # de-emphasised marks, axis rule
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#ffffff"
DIVERGING = ["#1c5cab", "#f0efec", "#e34948"]   # blue pole, neutral grey midpoint, red pole


def diverging(t: float) -> str:
    """Colour for t in [-1, 1] on the blue - grey - red scale."""
    t = max(-1.0, min(1.0, t))
    a, b = (DIVERGING[1], DIVERGING[0]) if t < 0 else (DIVERGING[1], DIVERGING[2])
    f = abs(t)
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(ca, cb))
