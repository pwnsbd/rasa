"""Generates Rasa's app icon: a stylized distillation flask with a glowing
gold liquid, matching the existing in-app visual language exactly rather
than inventing a new one — same silhouette family as
ui/src/components/DistillationBottle.tsx's SVG bottle path, same palette
as tailwind.config.js (charcoal #241c2b background, gold #c9a35c liquid).

Built with Pillow directly (no SVG renderer needed/available) at high
resolution, then downsampled into every size Windows' .ico format wants.
Run via the sidecar's own Python (Pillow's already a dependency there):

    sidecar/venv/Scripts/python.exe scripts/make-icon.py

Outputs electron/icon.ico (referenced by package.json's build.win.icon)
and electron/icon.png (1024x1024, for anywhere a plain raster is more
convenient than an .ico).
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
OUT_DIR = Path(__file__).resolve().parent.parent / "electron"

CHARCOAL = (36, 28, 43, 255)
CHARCOAL_DEEP = (24, 18, 29, 255)
GOLD = (201, 163, 92, 255)
GOLD_BRIGHT = (232, 199, 133, 255)
GOLD_DEEP = (163, 122, 58, 255)


def rounded_square_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    grad = Image.new("RGBA", (1, size))
    for y in range(size):
        t = y / (size - 1)
        px = tuple(int(top[c] + (bottom[c] - top[c]) * t) for c in range(4))
        grad.putpixel((0, y), px)
    return grad.resize((size, size))


def flask_polygon(cx: float, cy: float, scale: float) -> list[tuple[float, float]]:
    """A neck-and-bulb flask outline, in a 0..100 design grid centered
    roughly at (cx, cy) and scaled by `scale`. Built as rectangle (neck) +
    trapezoid (shoulder) + a real rounded-rectangle body (not a hand-rolled
    arc — an earlier version of this tried sampling an ellipse directly and
    the seams/proportions were fragile to get right by reasoning alone; a
    rounded rect with a large corner radius gives a reliably smooth,
    correctly-proportioned rounded bottom instead) — outline traced as one
    polygon so the whole thing renders as a single seamless silhouette.
    """
    neck_hw = 9  # neck half-width
    neck_top_y = -44
    neck_bottom_y = -20
    shoulder_bottom_y = -8
    body_hw = 27  # body half-width
    body_top_y = -8
    body_bottom_y = 40
    corner_r = 22  # bottom-corner rounding radius — most of body_hw, for a proper rounded bottom, not a boxy one

    def p(x, y):
        return (cx + x * scale, cy + y * scale)

    pts = [p(-neck_hw, neck_top_y), p(neck_hw, neck_top_y), p(neck_hw, neck_bottom_y)]
    pts.append(p(body_hw, shoulder_bottom_y))  # right shoulder
    pts.append(p(body_hw, body_bottom_y - corner_r))
    # Bottom-right rounded corner, sampled as a quarter-circle arc (0 -> 90 degrees).
    for i in range(17):
        rad = math.radians(90 * i / 16)
        pts.append(p(body_hw - corner_r + corner_r * math.cos(rad), body_bottom_y - corner_r + corner_r * math.sin(rad)))
    pts.append(p(-body_hw + corner_r, body_bottom_y))
    # Bottom-left rounded corner (90 -> 180 degrees).
    for i in range(17):
        rad = math.radians(90 + 90 * i / 16)
        pts.append(p(-body_hw + corner_r + corner_r * math.cos(rad), body_bottom_y - corner_r + corner_r * math.sin(rad)))
    pts.append(p(-body_hw, shoulder_bottom_y + (body_top_y - shoulder_bottom_y)))
    pts.append(p(-body_hw, shoulder_bottom_y))  # left shoulder
    pts.append(p(-neck_hw, neck_bottom_y))
    return pts


def draw_flask(canvas: Image.Image, cx: float, cy: float, scale: float):
    draw = ImageDraw.Draw(canvas)
    outline = flask_polygon(cx, cy, scale)

    # Soft gold glow behind the flask — same "something is glowing here"
    # language as DistillationBottle/CauldronVessel's drop-shadow glow on
    # the liquid, just baked into a raster instead of a CSS filter.
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon(outline, fill=(*GOLD[:3], 140))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=scale * 1.1))
    canvas.alpha_composite(glow)

    # The flask body itself: a vertical gold gradient (deep at the neck,
    # bright toward the pooled "liquid" at the bottom), clipped to the
    # flask silhouette via a mask so the gradient never spills outside it.
    body_mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(body_mask).polygon(outline, fill=255)
    body_grad = vertical_gradient(canvas.size[1], GOLD_DEEP, GOLD_BRIGHT)
    canvas.paste(body_grad, (0, 0), body_mask)

    # A thin darker rim so the silhouette reads clearly against light
    # backgrounds too (Windows taskbars aren't always dark).
    draw.line(outline + [outline[0]], fill=(*GOLD_DEEP[:3], 255), width=max(2, int(scale * 0.12)), joint="curve")

    # A small stopper/cap at the neck — the one added "character" detail,
    # kept simple since it has to survive being shrunk to 16px.
    cap_w, cap_h = scale * 12, scale * 7
    draw.rounded_rectangle(
        [cx - cap_w / 2, cy - 46 * scale - cap_h * 0.6, cx + cap_w / 2, cy - 46 * scale + cap_h * 0.4],
        radius=cap_h * 0.35,
        fill=CHARCOAL_DEEP,
    )

    # A soft diagonal highlight streak for a glass/liquid shine, same
    # restrained touch DistillationBottle's gradient stop softening gives.
    highlight = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    hl_draw = ImageDraw.Draw(highlight)
    hl_draw.ellipse(
        [cx - 22 * scale, cy - 10 * scale, cx - 6 * scale, cy + 30 * scale],
        fill=(255, 255, 255, 60),
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(radius=scale * 0.6))
    mask_img = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(mask_img).polygon(outline, fill=255)
    canvas.paste(Image.alpha_composite(canvas, highlight), (0, 0), mask_img)


def main():
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    bg = vertical_gradient(SIZE, CHARCOAL, CHARCOAL_DEEP)
    mask = rounded_square_mask(SIZE, radius=int(SIZE * 0.22))
    canvas.paste(bg, (0, 0), mask)

    draw_flask(canvas, cx=SIZE / 2, cy=SIZE / 2 + SIZE * 0.02, scale=SIZE / 100)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_DIR / "icon.png")

    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    canvas.save(OUT_DIR / "icon.ico", format="ICO", sizes=ico_sizes)
    print(f"Wrote {OUT_DIR / 'icon.png'} and {OUT_DIR / 'icon.ico'}")


if __name__ == "__main__":
    main()
