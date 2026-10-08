"""Shared typography and colors for SVG environment performance frames."""

from base64 import b64encode
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

WIDTH, HEIGHT, FPS = 1280, 720, 30
# Bump when fonts, drawing, or layouts change; action traces remain reusable.
RENDER_VERSION = "research-svg-1"
PAPER, SURFACE = "#FAFAF8", "#FFFFFF"
INK, MUTED = "#1F2933", "#52616B"
BLUE, GOLD = "#284B63", "#A87014"
GREEN, RED = "#18705B", "#B44949"
RULE, GRID = "#CAD3D5", "#E1E6E8"
HIGHLIGHT = "#EAF0F4"
POSITIVE_FILL, NEGATIVE_FILL = "#E0EEE8", "#F5E4E4"
FONTS = Path(__file__).with_name("fonts")


FACES = {
    "sans": ("lmsans10-regular.otf", "Latin Modern Sans", "normal"),
    "bold": ("lmsans10-bold.otf", "Latin Modern Sans", "bold"),
    "serif": ("lmroman10-regular.otf", "Latin Modern Roman", "normal"),
    "mono": ("lmmono10-regular.otf", "Latin Modern Mono", "normal"),
}


@lru_cache(maxsize=64)
def font(size, face="sans"):
    """Font metrics only; glyphs remain SVG text in the canonical frame."""
    return ImageFont.truetype(str(FONTS / FACES[face][0]), size)


@lru_cache(maxsize=2)
def font_css(embed_fonts):
    """Standalone frames embed fonts; archived frames share ../fonts assets."""
    rules = []
    for filename, family, weight in FACES.values():
        source = (
            "data:font/otf;base64," + b64encode((FONTS / filename).read_bytes()).decode()
            if embed_fonts
            else "../fonts/" + filename
        )
        rules.append(
            f"@font-face{{font-family:'{family}';font-weight:{weight};"
            f"src:url('{source}') format('opentype');}}"
        )
    return "\n".join(rules)
