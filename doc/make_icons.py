"""Generate Stratum brand images from the source icon.

Outputs:
  repository-card.png      — 1280x640 GitHub social card (white background)
  repository-card-dark.png — Transparent card with white text (for dark themes)
  stratum_logo.png         — Transparent logo, cropped tight, light pixels cleaned
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIR = os.path.dirname(__file__)
ICON_PATH = os.path.join(DIR, "stratum_icon.png")

CARD_OUTPUT = os.path.join(DIR, "repository-card.png")
CARD_DARK_OUTPUT = os.path.join(DIR, "repository-card-dark.png")
LOGO_OUTPUT = os.path.join(DIR, "stratum_logo.png")

CARD_W, CARD_H = 1280, 640
TEXT = "stratum"
TEXT_COLOR = (30, 41, 59)       # Dark navy to match icon's bottom layer
TEXT_COLOR_DARK = (255, 255, 255)  # White text for dark theme
FONT_SIZE = 96 * 2
GAP = 40

# Post-processing constants
COMPOSITE_BRIGHT_THRESH = 200
NEIGHBOR_RADIUS = 8
OPAQUE_ALPHA = 180


def load_font(size, bold=False):
    candidates = (
        ["/System/Library/Fonts/SFPro-Bold.otf",
         "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["/System/Library/Fonts/SFPro-Regular.otf",
         "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def load_icon():
    """Load and scale the source icon."""
    icon = Image.open(ICON_PATH).convert("RGBA")
    icon_target_h = int(CARD_H * 0.45)
    scale = icon_target_h / icon.height
    return icon.resize((int(icon.width * scale), icon_target_h), Image.LANCZOS)


def measure_text(font):
    """Return (width, height, y_offset) for the label text."""
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = tmp.textbbox((0, 0), TEXT, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[1]


def _draw_card(icon, font, text_w, text_h, text_y_off, *, bg, text_color):
    """Compose icon + text on a canvas of the given background."""
    if bg is None:
        img = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    else:
        img = Image.new("RGB", (CARD_W, CARD_H), bg)
    draw = ImageDraw.Draw(img)

    total_w = icon.width + GAP + text_w
    start_x = (CARD_W - total_w) // 2

    icon_y = (CARD_H - icon.height) // 2
    img.paste(icon, (start_x, icon_y), icon)

    text_y = (CARD_H - text_h) // 2 - text_y_off
    draw.text((start_x + icon.width + GAP, text_y), TEXT,
              fill=text_color, font=font)
    return img


def make_card(icon, font, text_w, text_h, text_y_off):
    """1280x640 white-background GitHub social card."""
    img = _draw_card(icon, font, text_w, text_h, text_y_off,
                     bg=(255, 255, 255), text_color=TEXT_COLOR)
    img.save(CARD_OUTPUT, "PNG")
    print(f"Saved {CARD_OUTPUT} ({CARD_W}x{CARD_H})")


def make_card_dark(icon, font, text_w, text_h, text_y_off):
    """Transparent card with white text for dark themes."""
    img = _draw_card(icon, font, text_w, text_h, text_y_off,
                     bg=None, text_color=TEXT_COLOR_DARK)
    img.save(CARD_DARK_OUTPUT, "PNG")
    print(f"Saved {CARD_DARK_OUTPUT} ({CARD_W}x{CARD_H})")


def make_logo(icon, font, text_w, text_h, text_y_off):
    """Transparent logo, cropped tight, with light-pixel cleanup."""
    canvas_w = icon.width + GAP + text_w + 20
    canvas_h = max(icon.height, text_h) + 20
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    icon_y = (canvas_h - icon.height) // 2
    img.paste(icon, (0, icon_y), icon)

    text_y = (canvas_h - text_h) // 2 - text_y_off
    draw.text((icon.width + GAP, text_y), TEXT, fill=TEXT_COLOR, font=font)

    img = _clean_light_pixels(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    img.save(LOGO_OUTPUT, "PNG")
    print(f"Saved {LOGO_OUTPUT} ({img.width}x{img.height})")


def _clean_light_pixels(img):
    """Replace semi-transparent light pixels with darkened neighbor colors."""
    data = np.array(img, dtype=np.int32)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    af = a / 255.0
    comp_bright = (r * af + 255 * (1 - af)
                   + g * af + 255 * (1 - af)
                   + b * af + 255 * (1 - af)) / 3.0

    visible = a > 5
    light_on_white = visible & (comp_bright > COMPOSITE_BRIGHT_THRESH)
    opaque = a >= OPAQUE_ALPHA

    h, w = data.shape[:2]
    out = data.copy()
    fixed = 0

    for y, x in zip(*np.where(light_on_white)):
        y0, y1 = max(0, y - NEIGHBOR_RADIUS), min(h, y + NEIGHBOR_RADIUS + 1)
        x0, x1 = max(0, x - NEIGHBOR_RADIUS), min(w, x + NEIGHBOR_RADIUS + 1)
        patch_opaque = opaque[y0:y1, x0:x1]

        if patch_opaque.any():
            avg_rgb = data[y0:y1, x0:x1, :3][patch_opaque].mean(axis=0)
            blend = 0.3 * (a[y, x] / 255.0)
            out[y, x, 0] = int(r[y, x] * blend + avg_rgb[0] * (1 - blend))
            out[y, x, 1] = int(g[y, x] * blend + avg_rgb[1] * (1 - blend))
            out[y, x, 2] = int(b[y, x] * blend + avg_rgb[2] * (1 - blend))
            out[y, x, 3] = min(255, int(a[y, x] * 1.8))
        else:
            out[y, x, 3] = 0
        fixed += 1

    print(f"  Fixed {fixed} light pixels")
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def main():
    icon = load_icon()
    font = load_font(FONT_SIZE, bold=True)
    text_w, text_h, text_y_off = measure_text(font)

    make_card(icon, font, text_w, text_h, text_y_off)
    make_card_dark(icon, font, text_w, text_h, text_y_off)
    make_logo(icon, font, text_w, text_h, text_y_off)


if __name__ == "__main__":
    main()
