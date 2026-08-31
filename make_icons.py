"""Generate the app icons.

The icon is the tug-of-war bar, which is the one element unique to this
dashboard. A monogram would say nothing; three split bars at different
points read as "probabilities" even at 48 pixels, and use the same blue
side / red side language as the app itself.

Android masks icons into circles, squircles and so on depending on the
launcher, so the background is full-bleed and the artwork stays inside the
centre 60%. Anything closer to the edge risks being cropped off.

    python tools/make_icons.py
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

INK = (11, 21, 25)
BLUE = (79, 168, 216)
RED = (216, 96, 79)
GOLD = (201, 162, 39)

# (split point, has the gold reference tick)
BARS = [(0.68, False), (0.42, True), (0.79, False)]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "icons")


def draw(size: int, safe: float = 0.62, rounded: bool = False) -> Image.Image:
    """One icon. `safe` is the fraction of the canvas the artwork occupies."""
    scale = 4                                   # supersample, then downsample
    px = size * scale
    img = Image.new("RGBA", (px, px), INK + (255,))
    d = ImageDraw.Draw(img)

    if rounded:                                 # for favicons, which aren't masked
        mask = Image.new("L", (px, px), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, px - 1, px - 1], radius=int(px * 0.22), fill=255)
        img.putalpha(mask)

    width = px * safe
    bar_h = width * 0.155
    gap = width * 0.105
    total = len(BARS) * bar_h + (len(BARS) - 1) * gap

    # Android launchers crop icons to a circle, a squircle or a rounded
    # square depending on the phone. Everything must therefore sit inside
    # the centre circle of diameter 80%, so the corners of the artwork --
    # the far ends of the top and bottom bars -- are the binding
    # constraint. Shrink until they fit rather than guessing a safe zone.
    limit = px * 0.40
    reach = ((width / 2) ** 2 + (total / 2) ** 2) ** 0.5
    if reach > limit:
        shrink = limit / reach
        width *= shrink
        bar_h *= shrink
        gap *= shrink
        total *= shrink

    left = (px - width) / 2
    top = (px - total) / 2
    radius = bar_h / 2

    for split, tick in BARS:
        bottom = top + bar_h
        mid = left + width * split
        # Red underneath the full width, blue drawn over the left portion:
        # simpler than two shapes meeting, and the join stays clean.
        d.rounded_rectangle([left, top, left + width, bottom],
                            radius=radius, fill=RED)
        d.rounded_rectangle([left, top, mid, bottom], radius=radius, fill=BLUE)
        # Square off the blue's right end so the split reads as a hard edge.
        d.rectangle([mid - radius, top, mid, bottom], fill=BLUE)
        # The divider, in the page background colour.
        d.rectangle([mid - px * 0.008, top, mid + px * 0.008, bottom], fill=INK)

        if tick and size >= 64:                 # too fine to survive at 32px
            tick_x = left + width * 0.55
            d.rectangle([tick_x - px * 0.007, bottom + gap * 0.22,
                         tick_x + px * 0.007, bottom + gap * 0.62], fill=GOLD)
        top = bottom + gap

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for size in (192, 512):                     # Android, from the manifest
        draw(size).save(os.path.join(OUT, f"icon-{size}.png"))
    draw(180).save(os.path.join(OUT, "apple-touch-icon.png"))   # iOS
    for size in (32, 16):                       # browser tab
        draw(size, safe=0.78, rounded=True).save(
            os.path.join(OUT, f"favicon-{size}.png"))

    # Multi-resolution .ico for older browsers.
    draw(64, safe=0.78, rounded=True).save(
        os.path.join(OUT, "favicon.ico"),
        sizes=[(16, 16), (32, 32), (48, 48)])

    for name in sorted(os.listdir(OUT)):
        path = os.path.join(OUT, name)
        print(f"  {name:<26} {os.path.getsize(path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
