"""A minimal, self-contained SVG canvas for CAD-style 2D drawings.

Used by the GemPy implicit-model section plots. SVG rather than a raster:
sections get zoomed and printed, and the linework has to stay sharp. It also
embeds directly in the HTML report with no image encoding.
"""
from __future__ import annotations

import numpy as np

# CAD graphics-window palette: dark ground, thin bright linework
INK = {
    "bg": "#0b0e13",
    "frame": "#3a4658",
    "grid": "#1b2430",
    "text": "#9fb2c6",
    "text_bright": "#dce7f2",
    "hole": "#5b6b7d",
    "orebody": "#00e5c8",
    "pit_crest": "#ff8a3d",
    "pit_toe": "#ffd166",
    "ramp": "#ff4d6d",
    "topo": "#7a8b5a",
    "block": "#2a3444",
}


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


class SvgCanvas:
    """Minimal SVG writer with a world-to-screen transform and a drawing frame."""

    def __init__(self, width: int, height: int, bounds: tuple[float, float, float, float],
                 margin: int = 54, flip_y: bool = True):
        self.w, self.h = width, height
        self.margin = margin
        self.flip_y = flip_y
        x0, y0, x1, y1 = bounds
        if x1 - x0 < 1e-9:
            x0, x1 = x0 - 1, x1 + 1
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - 1, y1 + 1

        # keep the drawing to scale: one scale factor for both axes
        avail_w = width - 2 * margin
        avail_h = height - 2 * margin
        self.scale = min(avail_w / (x1 - x0), avail_h / (y1 - y0))
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.offset_x = margin + (avail_w - (x1 - x0) * self.scale) / 2.0
        self.offset_y = margin + (avail_h - (y1 - y0) * self.scale) / 2.0
        self.parts: list[str] = []

    def sx(self, x):
        return self.offset_x + (np.asarray(x, float) - self.x0) * self.scale

    def sy(self, y):
        v = (np.asarray(y, float) - self.y0) * self.scale
        return (self.h - self.offset_y - v) if self.flip_y else (self.offset_y + v)

    # -- primitives
    def polyline(self, xs, ys, stroke, width=1.0, closed=False, fill="none",
                 opacity=1.0, dash=None):
        px, py = self.sx(xs), self.sy(ys)
        if len(px) < 2:
            return
        pts = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px, py))
        tag = "polygon" if closed else "polyline"
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<{tag} points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}"'
            f' stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def segments(self, x0, y0, x1, y1, colors, width=2.0, opacity=1.0):
        """Many independent coloured segments, one path each."""
        ax, ay, bx, by = self.sx(x0), self.sy(y0), self.sx(x1), self.sy(y1)
        for i in range(len(ax)):
            self.parts.append(
                f'<line x1="{ax[i]:.2f}" y1="{ay[i]:.2f}" x2="{bx[i]:.2f}" y2="{by[i]:.2f}" '
                f'stroke="{colors[i]}" stroke-width="{width}" stroke-opacity="{opacity}"/>')

    def rects(self, cx, cy, w, h, colors, opacity=1.0, stroke="none"):
        px, py = self.sx(cx), self.sy(cy)
        sw, sh = w * self.scale, h * self.scale
        for i in range(len(px)):
            self.parts.append(
                f'<rect x="{px[i] - sw / 2:.2f}" y="{py[i] - sh / 2:.2f}" '
                f'width="{sw:.2f}" height="{sh:.2f}" fill="{colors[i]}" '
                f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="0.3"/>')

    def circles(self, x, y, r, fill, opacity=1.0):
        px, py = self.sx(x), self.sy(y)
        for i in range(len(px)):
            self.parts.append(f'<circle cx="{px[i]:.2f}" cy="{py[i]:.2f}" r="{r}" '
                              f'fill="{fill}" fill-opacity="{opacity}"/>')

    def text(self, x, y, label, fill=INK["text"], size=10, anchor="start",
             screen=False, weight="normal"):
        px = x if screen else float(self.sx(x))
        py = y if screen else float(self.sy(y))
        self.parts.append(
            f'<text x="{px:.1f}" y="{py:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="ui-monospace,Consolas,monospace">{_esc(label)}</text>')

    # -- frame furniture
    def frame(self, x_label: str, y_label: str, n_ticks: int = 6):
        m = self.margin
        self.parts.append(
            f'<rect x="{m}" y="{m}" width="{self.w - 2 * m}" height="{self.h - 2 * m}" '
            f'fill="none" stroke="{INK["frame"]}" stroke-width="1"/>')

        for value in np.linspace(self.x0, self.x1, n_ticks):
            px = float(self.sx(value))
            if not (m <= px <= self.w - m):
                continue
            self.parts.append(f'<line x1="{px:.1f}" y1="{m}" x2="{px:.1f}" '
                              f'y2="{self.h - m}" stroke="{INK["grid"]}" stroke-width="0.6"/>')
            self.text(px, self.h - m + 16, f"{value:,.0f}", size=9, anchor="middle",
                      screen=True)
        for value in np.linspace(self.y0, self.y1, n_ticks):
            py = float(self.sy(value))
            if not (m <= py <= self.h - m):
                continue
            self.parts.append(f'<line x1="{m}" y1="{py:.1f}" x2="{self.w - m}" '
                              f'y2="{py:.1f}" stroke="{INK["grid"]}" stroke-width="0.6"/>')
            self.text(m - 8, py + 3, f"{value:,.0f}", size=9, anchor="end", screen=True)

        self.text(self.w / 2, self.h - 12, x_label, size=10, anchor="middle",
                  screen=True, fill=INK["text"])
        self.parts.append(
            f'<text x="14" y="{self.h / 2:.0f}" fill="{INK["text"]}" font-size="10" '
            f'text-anchor="middle" font-family="ui-monospace,Consolas,monospace" '
            f'transform="rotate(-90 14 {self.h / 2:.0f})">{_esc(y_label)}</text>')

    def title_block(self, title: str, subtitle: str = ""):
        self.text(self.margin, self.margin - 22, title, size=13,
                  fill=INK["text_bright"], screen=True, weight="700")
        if subtitle:
            self.text(self.margin, self.margin - 8, subtitle, size=10, screen=True)

    def scale_bar(self):
        """A round-number scale bar, sized to the current drawing scale."""
        target_px = (self.w - 2 * self.margin) * 0.22
        raw = target_px / self.scale
        magnitude = 10 ** np.floor(np.log10(max(raw, 1e-9)))
        nice = min((1, 2, 5, 10), key=lambda k: abs(k * magnitude - raw)) * magnitude
        length_px = nice * self.scale

        x = self.w - self.margin - length_px
        y = self.h - self.margin - 16
        self.parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + length_px:.1f}" '
                          f'y2="{y:.1f}" stroke="{INK["text_bright"]}" stroke-width="2"/>')
        for tick in (x, x + length_px):
            self.parts.append(f'<line x1="{tick:.1f}" y1="{y - 4:.1f}" x2="{tick:.1f}" '
                              f'y2="{y + 4:.1f}" stroke="{INK["text_bright"]}" stroke-width="2"/>')
        self.text(x + length_px / 2, y - 8, f"{nice:,.0f} m", size=9, anchor="middle",
                  screen=True, fill=INK["text_bright"])

    def north_arrow(self):
        cx, cy, r = self.w - self.margin - 22, self.margin + 30, 15
        self.parts.append(
            f'<path d="M {cx:.1f} {cy - r:.1f} L {cx + r * 0.45:.1f} {cy + r * 0.7:.1f} '
            f'L {cx:.1f} {cy + r * 0.3:.1f} L {cx - r * 0.45:.1f} {cy + r * 0.7:.1f} Z" '
            f'fill="{INK["text_bright"]}" fill-opacity="0.85"/>')
        self.text(cx, cy + r + 13, "N", size=10, anchor="middle", screen=True,
                  fill=INK["text_bright"], weight="700")

    def legend(self, entries: list[tuple[str, str]], x=None, y=None):
        x = self.margin + 10 if x is None else x
        y = self.margin + 16 if y is None else y
        for i, (colour, label) in enumerate(entries):
            yy = y + i * 15
            self.parts.append(f'<rect x="{x}" y="{yy - 8}" width="14" height="9" '
                              f'fill="{colour}"/>')
            self.text(x + 20, yy, label, size=9, screen=True)

    def render(self) -> str:
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" '
                f'xmlns="http://www.w3.org/2000/svg" role="img" '
                f'style="background:{INK["bg"]};display:block">'
                f'<rect width="{self.w}" height="{self.h}" fill="{INK["bg"]}"/>'
                + "".join(self.parts) + "</svg>")
