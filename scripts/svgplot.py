"""Minimal pure-Python SVG charts for the paper figures (no compiled dependencies).

Supports line, point, reference-line, rectangle (heatmap) and text marks on linear or log
axes, several panels per figure, and a legend. SVG embeds directly in LaTeX (the `svg`
package, e.g. on Overleaf) and renders in any browser. Colours come from figstyle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from figstyle import CONTEXT, GRID, INK, INK_2, MUTED, SURFACE

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    ticks, t = [], start
    while t <= hi + 1e-9 * step:
        ticks.append(round(t, 10))
        t += step
    return ticks


def log_ticks(lo: float, hi: float) -> list[float]:
    return [10.0 ** k for k in range(math.ceil(math.log10(lo) - 1e-9),
                                     math.floor(math.log10(hi) + 1e-9) + 1)]


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-4:
        exp = math.floor(math.log10(abs(v)))
        mant = v / 10 ** exp
        return f"10^{exp}" if abs(mant - 1) < 1e-9 else f"{mant:g}e{exp}"
    return f"{v:g}"


@dataclass
class Axes:
    x0: float
    y0: float
    w: float
    h: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    xscale: str = "linear"
    yscale: str = "linear"
    title: str = ""
    marks: list[str] = field(default_factory=list)
    legend_items: list[tuple[str, str, str]] = field(default_factory=list)   # (kind, color, text)

    def _t(self, v: float, lim: tuple[float, float], scale: str) -> float:
        if scale == "log":
            v, lim = math.log10(v), (math.log10(lim[0]), math.log10(lim[1]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def px(self, x: float, y: float) -> tuple[float, float]:
        return (self.x0 + self._t(x, self.xlim, self.xscale) * self.w,
                self.y0 + self.h - self._t(y, self.ylim, self.yscale) * self.h)

    def line(self, xs, ys, color: str, width: float = 2.0, label: str | None = None,
             dash: str | None = None) -> None:
        pts = " ".join(f"{a:.2f},{b:.2f}" for a, b in (self.px(x, y) for x, y in zip(xs, ys)))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.marks.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                          f'stroke-width="{width}" stroke-linejoin="round" '
                          f'stroke-linecap="round"{extra}/>')
        if label:
            self.legend_items.append(("line", color, label))

    def points(self, xs, ys, color: str, r: float = 4.0, hollow: bool = False,
               label: str | None = None, titles=None) -> None:
        fill = SURFACE if hollow else color
        stroke = color if hollow else SURFACE
        for i, (x, y) in enumerate(zip(xs, ys)):
            a, b = self.px(x, y)
            tip = f"<title>{escape(titles[i])}</title>" if titles else ""
            self.marks.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="{r}" fill="{fill}" '
                              f'stroke="{stroke}" stroke-width="{1.5 if hollow else 2}">'
                              f"{tip}</circle>")
        if label:
            self.legend_items.append(("hollow" if hollow else "dot", color, label))

    def hline(self, y: float, color: str, label: str | None = None) -> None:
        self.line(self.xlim, (y, y), color, label=label)

    def rect(self, xa: float, ya: float, xb: float, yb: float, fill: str,
             title: str | None = None) -> None:
        (a, b), (c, d) = self.px(xa, ya), self.px(xb, yb)
        tip = f"<title>{escape(title)}</title>" if title else ""
        self.marks.append(f'<rect x="{min(a, c) + 1:.2f}" y="{min(b, d) + 1:.2f}" '
                          f'width="{abs(c - a) - 2:.2f}" height="{abs(d - b) - 2:.2f}" '
                          f'fill="{fill}">{tip}</rect>')

    def text(self, x: float, y: float, s: str, anchor: str = "start", color: str = INK,
             size: float = 10, dx: float = 0, dy: float = 0) -> None:
        a, b = self.px(x, y)
        self.marks.append(f'<text x="{a + dx:.2f}" y="{b + dy:.2f}" text-anchor="{anchor}" '
                          f'font-size="{size}" fill="{color}">{escape(s)}</text>')


class Figure:
    def __init__(self, width: float = 360, height: float = 250) -> None:
        self.width, self.height = width, height
        self.panels: list[tuple[Axes, dict]] = []
        self.extra: list[str] = []

    def axes(self, rect: tuple[float, float, float, float], xlim, ylim, *, xscale="linear",
             yscale="linear", xlabel="", ylabel="", title="", xticks=None, yticks=None,
             xticklabels=None, yticklabels=None, grid=True) -> Axes:
        ax = Axes(*rect, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale, title=title)
        self.panels.append((ax, {"xlabel": xlabel, "ylabel": ylabel, "xticks": xticks,
                                 "yticks": yticks, "xticklabels": xticklabels,
                                 "yticklabels": yticklabels, "grid": grid}))
        return ax

    def _frame(self, ax: Axes, o: dict) -> list[str]:
        out = []
        xt = o["xticks"] if o["xticks"] is not None else (
            log_ticks(*ax.xlim) if ax.xscale == "log" else nice_ticks(*ax.xlim))
        yt = o["yticks"] if o["yticks"] is not None else (
            log_ticks(*ax.ylim) if ax.yscale == "log" else nice_ticks(*ax.ylim))
        xl = o["xticklabels"] or [fmt(v) for v in xt]
        yl = o["yticklabels"] or [fmt(v) for v in yt]
        for v, lab in zip(xt, xl):
            a, _ = ax.px(v, ax.ylim[0])
            if o["grid"]:
                out.append(f'<line x1="{a:.2f}" y1="{ax.y0}" x2="{a:.2f}" y2="{ax.y0 + ax.h}" '
                           f'stroke="{GRID}" stroke-width="1"/>')
            out.append(f'<text x="{a:.2f}" y="{ax.y0 + ax.h + 14}" text-anchor="middle" '
                       f'font-size="10" fill="{MUTED}">{escape(lab)}</text>')
        for v, lab in zip(yt, yl):
            _, b = ax.px(ax.xlim[0], v)
            if o["grid"]:
                out.append(f'<line x1="{ax.x0}" y1="{b:.2f}" x2="{ax.x0 + ax.w}" y2="{b:.2f}" '
                           f'stroke="{GRID}" stroke-width="1"/>')
            out.append(f'<text x="{ax.x0 - 6}" y="{b + 3.5:.2f}" text-anchor="end" '
                       f'font-size="10" fill="{MUTED}">{escape(lab)}</text>')
        out.append(f'<line x1="{ax.x0}" y1="{ax.y0 + ax.h}" x2="{ax.x0 + ax.w}" '
                   f'y2="{ax.y0 + ax.h}" stroke="{CONTEXT}" stroke-width="1"/>')
        if o["xlabel"]:
            out.append(f'<text x="{ax.x0 + ax.w / 2:.2f}" y="{ax.y0 + ax.h + 30}" '
                       f'text-anchor="middle" font-size="11" fill="{INK_2}">'
                       f'{escape(o["xlabel"])}</text>')
        if o["ylabel"]:
            cx, cy = ax.x0 - 38, ax.y0 + ax.h / 2
            out.append(f'<text x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle" font-size="11" '
                       f'fill="{INK_2}" transform="rotate(-90 {cx:.2f} {cy:.2f})">'
                       f'{escape(o["ylabel"])}</text>')
        if ax.title:
            out.append(f'<text x="{ax.x0}" y="{ax.y0 - 8}" font-size="11" font-weight="600" '
                       f'fill="{INK}">{escape(ax.title)}</text>')
        return out

    def legend(self, items: list[tuple[str, str, str]], x: float, y: float,
               row: float = 15) -> None:
        for i, (kind, color, text) in enumerate(items):
            cy = y + i * row
            if kind in ("line", "band"):
                width = 7 if kind == "band" else 2
                self.extra.append(f'<line x1="{x}" y1="{cy}" x2="{x + 16}" y2="{cy}" '
                                  f'stroke="{color}" stroke-width="{width}" '
                                  f'stroke-linecap="round"/>')
            else:
                fill = SURFACE if kind == "hollow" else color
                self.extra.append(f'<circle cx="{x + 8}" cy="{cy}" r="4" fill="{fill}" '
                                  f'stroke="{color}" stroke-width="1.5"/>')
            self.extra.append(f'<text x="{x + 22}" y="{cy + 3.5}" font-size="10" '
                              f'fill="{INK_2}">{escape(text)}</text>')

    def save(self, path: str | Path) -> None:
        body = []
        for ax, o in self.panels:
            body += self._frame(ax, o)
            clip = f"clip{id(ax)}"
            body.append(f'<clipPath id="{clip}"><rect x="{ax.x0 - 6}" y="{ax.y0 - 6}" '
                        f'width="{ax.w + 12}" height="{ax.h + 12}"/></clipPath>')
            body.append(f'<g clip-path="url(#{clip})">' + "".join(ax.marks) + "</g>")
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} '
               f'{self.height}" width="{self.width}" height="{self.height}" '
               f'font-family="{FONT}"><rect width="100%" height="100%" fill="{SURFACE}"/>'
               + "".join(body + self.extra) + "</svg>")
        Path(path).write_text(svg, encoding="utf-8")
