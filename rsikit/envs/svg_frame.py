"""Small SVG drawing vocabulary shared by environment video renderers."""

from contextlib import contextmanager
from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from PIL import Image

from .render_theme import FACES, FONTS, HEIGHT, INK, MUTED, PAPER, RULE, WIDTH, font, font_css


def number(value):
    return f"{value:.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


class SvgFrame:
    """A vector frame; stable group IDs identify scene elements for future animation."""

    def __init__(self, title):
        self.root = Element(
            "svg",
            xmlns="http://www.w3.org/2000/svg",
            width=str(WIDTH),
            height=str(HEIGHT),
            viewBox=f"0 0 {WIDTH} {HEIGHT}",
            role="img",
        )
        SubElement(self.root, "title").text = title
        self.style = SubElement(self.root, "style")
        self.parent = self.root
        self.rect((0, 0, WIDTH, HEIGHT), fill=PAPER, id="background")

    def add(self, tag, **attributes):
        return SubElement(
            self.parent,
            tag,
            {k.replace("_", "-"): number(v) for k, v in attributes.items() if v is not None},
        )

    @contextmanager
    def group(self, id, **attributes):
        parent = self.parent
        self.parent = self.add("g", id=id, **attributes)
        try:
            yield
        finally:
            self.parent = parent

    def rect(self, bounds, fill="none", stroke=None, width=1, radius=0, **attributes):
        x1, y1, x2, y2 = bounds
        self.add(
            "rect",
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            fill=fill,
            stroke=stroke,
            stroke_width=width,
            rx=radius,
            **attributes,
        )

    def line(self, points, fill=INK, width=1, dash=None, **attributes):
        self.add(
            "polyline",
            points=" ".join(f"{number(x)},{number(y)}" for x, y in points),
            fill="none",
            stroke=fill,
            stroke_width=width,
            stroke_dasharray=dash,
            stroke_linejoin="round",
            **attributes,
        )

    def polygon(self, points, fill=INK, **attributes):
        self.add(
            "polygon",
            points=" ".join(f"{number(x)},{number(y)}" for x, y in points),
            fill=fill,
            **attributes,
        )

    def circle(self, x, y, radius, fill=INK, **attributes):
        self.add("circle", cx=x, cy=y, r=radius, fill=fill, **attributes)

    def text(
        self, x, y, value, size=16, fill=INK, anchor="lt", face="sans", max_width=None, **attributes
    ):
        value = str(value)
        typeface = font(size, face)
        if max_width is not None and typeface.getlength(value) > max_width:
            low, high = 0, len(value)
            while low < high:
                middle = (low + high + 1) // 2
                if typeface.getlength(value[:middle] + "…") <= max_width:
                    low = middle
                else:
                    high = middle - 1
            value = value[:low] + "…" if typeface.getlength("…") <= max_width else ""
        _, top, _, bottom = typeface.getbbox(value, anchor="ls")
        baseline = y - ((top + bottom) / 2 if anchor[1] == "m" else top)
        node = self.add(
            "text",
            x=x,
            y=baseline,
            fill=fill,
            font_size=size,
            font_family=FACES[face][1],
            font_weight=FACES[face][2],
            text_anchor={"l": "start", "m": "middle", "r": "end"}[anchor[0]],
            **attributes,
        )
        node.text = value

    def header(self, title, policy_name, detail, value, caption, color=INK):
        with self.group("header"):
            self.text(32, 18, title, 32, face="serif", max_width=850)
            self.text(32, 57, policy_name, 18, max_width=760, id="policy-name")
            self.text(1248, 18, value, 32, color, "rt", face="serif", max_width=340, id="score")
            self.text(1248, 57, caption, 16, MUTED, "rt", max_width=440)
            self.line([(32, 86), (1248, 86)], RULE)
            self.text(32, 99, detail, 16, MUTED, max_width=1216, id="episode-context")

    def svg(self, *, embed_fonts=True):
        self.style.text = font_css(embed_fonts)
        return tostring(self.root, encoding="unicode")


def rasterize(svg):
    """Rasterize the canonical SVG only at the Gymnasium/video boundary."""
    import resvg_py

    png = resvg_py.svg_to_bytes(
        svg_string=svg,
        skip_system_fonts=True,
        font_dirs=[str(FONTS)],
        font_family="Latin Modern Sans",
    )
    with Image.open(BytesIO(png)) as image:
        return np.asarray(image.convert("RGB")).copy()
