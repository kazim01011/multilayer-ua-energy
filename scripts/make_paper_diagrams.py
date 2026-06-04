from __future__ import annotations

import math
import random
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "overleaf" / "figures"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_HEAD = font(25, True)
F_TEXT = font(22)
F_SMALL = font(18)
F_MATH = font(21)
F_TINY = font(16)


def rounded_box(draw: ImageDraw.ImageDraw, xy, fill, outline, width=3, radius=18):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill=(20, 20, 20), spacing=5):
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=fnt)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=fnt, fill=fill)
        y += h + spacing


def wrapped_text(draw: ImageDraw.ImageDraw, xy, title: str, body: str, color, body_width=26):
    x0, y0, x1, y1 = xy
    title_box = draw.textbbox((0, 0), title, font=F_HEAD)
    draw.text((x0 + (x1 - x0 - (title_box[2] - title_box[0])) / 2, y0 + 18), title, font=F_HEAD, fill=color)
    lines: list[str] = []
    for part in body.split("\n"):
        lines.extend(wrap(part, width=body_width) if part else [""])
    y = y0 + 58
    for line in lines:
        box = draw.textbbox((0, 0), line, font=F_TEXT)
        draw.text((x0 + (x1 - x0 - (box[2] - box[0])) / 2, y), line, font=F_TEXT, fill=(25, 25, 25))
        y += 28


def arrow(draw: ImageDraw.ImageDraw, start, end, color=(90, 90, 90), width=5):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        pts = [(ex, ey), (ex - sign * 22, ey - 11), (ex - sign * 22, ey + 11)]
    else:
        sign = 1 if ey > sy else -1
        pts = [(ex, ey), (ex - 11, ey - sign * 22), (ex + 11, ey - sign * 22)]
    draw.polygon(pts, fill=color)


def draw_table(draw, xy, cols=5, rows=5):
    x0, y0, x1, y1 = xy
    draw.rectangle(xy, outline=(120, 120, 120), width=2, fill=(255, 255, 255))
    for c in range(1, cols):
        x = x0 + c * (x1 - x0) / cols
        draw.line([(x, y0), (x, y1)], fill=(170, 170, 170), width=1)
    for r in range(1, rows):
        y = y0 + r * (y1 - y0) / rows
        draw.line([(x0, y), (x1, y)], fill=(170, 170, 170), width=1)


def make_flowchart():
    img = Image.new("RGB", (1900, 1180), "white")
    draw = ImageDraw.Draw(img)
    boxes = {
        "sim": (35, 35, 350, 190),
        "snap": (430, 35, 755, 190),
        "multi": (835, 35, 1215, 190),
        "oracle": (1295, 35, 1665, 190),
        "layers": (835, 270, 1215, 485),
        "base": (430, 600, 755, 785),
        "models": (835, 600, 1215, 785),
        "repair": (1295, 600, 1665, 785),
        "eval": (890, 875, 1460, 1065),
    }
    styles = {
        "sim": ((247, 247, 247), (70, 70, 70), (20, 20, 20)),
        "snap": ((255, 247, 236), (220, 92, 26), (150, 55, 10)),
        "multi": ((250, 242, 248), (178, 45, 117), (120, 20, 75)),
        "oracle": ((255, 250, 232), (200, 130, 25), (130, 80, 0)),
        "layers": ((239, 246, 255), (70, 110, 210), (35, 75, 160)),
        "base": ((237, 252, 248), (0, 130, 125), (0, 90, 90)),
        "models": ((246, 243, 255), (105, 75, 185), (70, 45, 145)),
        "repair": ((238, 251, 239), (40, 150, 60), (10, 95, 35)),
        "eval": ((255, 245, 248), (185, 45, 90), (135, 25, 65)),
    }
    content = {
        "sim": ("1. Wireless simulation", "dense BS layout\nUE demand snapshots\nchannel gain / SINR"),
        "snap": ("2. Snapshot features", "UE feature matrix X\ntraffic vector d\nchannel matrix G"),
        "multi": ("3. Multilayer graph", "same UE nodes\nrelation-specific\nadjacency matrices"),
        "oracle": ("4. Reference decisions", "active BS subsets\nenergy-aware assignment\nQoS/load checks"),
        "layers": ("5. Four relation layers", "association competition\ninterference similarity\nload pressure\ntemporal traffic similarity"),
        "base": ("6A. Baselines", "RSRP, SINR, load-aware\nFlat MLP\nAggregated GCN"),
        "models": ("6B. Proposed models", "ML-GCN fixed fusion\nAttn-ML-GCN\nactive-BS selector"),
        "repair": ("7. Feasible control", "select active BS set\nQoS/load repair\nfinal association"),
        "eval": ("8. Comparative evaluation", "energy saving, oracle gap, served ratio,\nactive BSs, load, ablation, attention entropy"),
    }
    for key, xy in boxes.items():
        fill, outline, color = styles[key]
        rounded_box(draw, xy, fill, outline, width=3)
        wrapped_text(draw, xy, *content[key], color=color, body_width=30)

    draw_table(draw, (515, 123, 690, 173), cols=5, rows=3)

    arrow(draw, (350, 112), (430, 112))
    arrow(draw, (755, 112), (835, 112))
    arrow(draw, (1215, 112), (1295, 112))
    arrow(draw, (1025, 190), (1025, 270))
    arrow(draw, (1025, 485), (1025, 600))
    arrow(draw, (835, 695), (755, 695))
    arrow(draw, (1215, 695), (1295, 695))
    arrow(draw, (1465, 785), (1280, 875))
    arrow(draw, (1025, 785), (1115, 875))
    arrow(draw, (592, 785), (1010, 875))

    legend = (40, 875, 390, 1065)
    rounded_box(draw, legend, (255, 255, 255), (160, 160, 160), width=2, radius=14)
    text_center(draw, (legend[0] + 20, legend[1] + 18, legend[2] - 20, legend[1] + 54), "Legend", F_HEAD)
    arrow(draw, (80, 960), (150, 960))
    draw.text((170, 947), "Data / decision flow", font=F_SMALL, fill=(30, 30, 30))
    draw.line([(80, 1015), (150, 1015)], fill=(70, 110, 210), width=5)
    draw.text((170, 1002), "Multilayer branch", font=F_SMALL, fill=(30, 30, 30))

    out = FIG_DIR / "fig_paper_flow.png"
    img.save(out, dpi=(300, 300))


def draw_nodes(draw, cx, cy, color):
    coords = [(cx - 60, cy - 30), (cx - 10, cy - 70), (cx + 55, cy - 35), (cx + 25, cy + 30), (cx - 45, cy + 45)]
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 0), (0, 3)]
    for a, b in edges:
        draw.line([coords[a], coords[b]], fill=color, width=3)
    for x, y in coords:
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(245, 245, 245), outline=(20, 20, 20), width=2)


def make_model():
    img = Image.new("RGB", (1900, 1120), "white")
    draw = ImageDraw.Draw(img)
    title = "Proposed Multilayer Graph Learning Model for User Association"
    draw.text((950 - draw.textbbox((0, 0), title, font=F_TITLE)[2] / 2, 30), title, font=F_TITLE, fill=(20, 20, 20))

    main = (325, 90, 1790, 1060)
    rounded_box(draw, main, (255, 255, 255), (80, 80, 80), width=2, radius=18)
    for dash_x in range(main[0] + 8, main[2] - 8, 28):
        draw.line([(dash_x, main[1]), (dash_x + 14, main[1])], fill=(120, 120, 120), width=2)
        draw.line([(dash_x, main[3]), (dash_x + 14, main[3])], fill=(120, 120, 120), width=2)
    for dash_y in range(main[1] + 8, main[3] - 8, 28):
        draw.line([(main[0], dash_y), (main[0], dash_y + 14)], fill=(120, 120, 120), width=2)
        draw.line([(main[2], dash_y), (main[2], dash_y + 14)], fill=(120, 120, 120), width=2)

    feature = (35, 145, 285, 760)
    rounded_box(draw, feature, (252, 252, 252), (80, 80, 80), width=2, radius=18)
    text_center(draw, (feature[0], 165, feature[2], 235), "UE Feature\nMatrix X", F_HEAD)
    draw_table(draw, (78, 270, 242, 415), cols=4, rows=5)
    for i in range(5):
        for j in range(4):
            x = 98 + j * 40
            y = 290 + i * 28
            shade = 90 + ((i + j) * 23) % 120
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(70, shade, 80), outline=(40, 120, 50))
    bullets = ["position", "demand", "RSRP/SINR", "traffic hour", "BS radio features"]
    y = 470
    for b in bullets:
        draw.ellipse((62, y + 9, 70, y + 17), fill=(45, 130, 55))
        draw.text((85, y), b, font=F_TEXT, fill=(20, 20, 20))
        y += 45

    arrow(draw, (285, 450), (340, 450), color=(30, 90, 170), width=8)

    layers = [
        ((360, 140, 785, 300), "Association Layer", "strongest-BS competition", (255, 238, 225), (220, 105, 35)),
        ((360, 335, 785, 495), "Interference Layer", "channel-profile similarity", (232, 243, 255), (55, 110, 205)),
        ((360, 530, 785, 690), "Load Layer", "demand and load pressure", (255, 248, 218), (218, 160, 40)),
        ((360, 725, 785, 885), "Temporal Layer", "spatial-traffic similarity", (235, 250, 235), (55, 150, 65)),
    ]
    for xy, head, body, fill, outline in layers:
        rounded_box(draw, xy, fill, outline, width=3)
        draw.text((xy[0] + 25, xy[1] + 28), head, font=F_HEAD, fill=outline)
        draw.text((xy[0] + 25, xy[1] + 70), body, font=F_TEXT, fill=(35, 35, 35))
        draw_nodes(draw, xy[0] + 315, xy[1] + 88, outline)
        draw.text((xy[2] - 78, xy[3] - 35), "N x N", font=F_SMALL, fill=(30, 30, 30))
        arrow(draw, (xy[2], (xy[1] + xy[3]) / 2), (850, (xy[1] + xy[3]) / 2))

    block_positions = [(880, 165), (880, 360), (880, 555), (880, 750)]
    colors = [(220, 105, 35), (55, 110, 205), (218, 160, 40), (55, 150, 65)]
    names = ["Assoc.", "Interf.", "Load", "Temp."]
    for (x, y), color, name in zip(block_positions, colors, names):
        rounded_box(draw, (x, y, x + 130, y + 80), (248, 248, 248), color, width=2, radius=12)
        text_center(draw, (x, y, x + 130, y + 80), name + "\nGCN", F_TEXT)
        arrow(draw, (x + 130, y + 40), (1080, y + 40))
        rounded_box(draw, (1080, y - 5, 1145, y + 85), (255, 255, 255), (40, 40, 40), width=2, radius=12)
        for k in range(3):
            draw.ellipse((1102, y + 10 + 24 * k, 1124, y + 32 + 24 * k), fill=tuple(min(255, c + 100) for c in color), outline=(20, 20, 20))
        arrow(draw, (1145, y + 40), (1240, 470), color=color, width=4)

    fusion = (1235, 350, 1440, 610)
    rounded_box(draw, fusion, (247, 243, 255), (105, 75, 185), width=3, radius=14)
    text_center(
        draw,
        fusion,
        "Cross-layer\nAttention Fusion\n\nalpha_a  alpha_i\nalpha_l  alpha_t",
        F_TEXT,
    )
    arrow(draw, (1440, 480), (1505, 480))
    rounded_box(draw, (1505, 420, 1575, 560), (255, 255, 255), (40, 40, 40), width=2, radius=12)
    for k in range(4):
        draw.ellipse((1530, 438 + 28 * k, 1552, 460 + 28 * k), fill=(145, 110, 205), outline=(20, 20, 20))
    draw.text((1518, 575), "E", font=F_HEAD, fill=(40, 40, 40))

    arrow(draw, (1575, 480), (1635, 480))
    rounded_box(draw, (1635, 370, 1760, 455), (245, 245, 245), (70, 70, 70), width=2, radius=12)
    text_center(draw, (1635, 370, 1760, 455), "UE-BS\nsoftmax P", F_TEXT)
    rounded_box(draw, (1635, 515, 1760, 600), (245, 245, 245), (70, 70, 70), width=2, radius=12)
    text_center(draw, (1635, 515, 1760, 600), "Active-BS\nselector", F_TEXT)
    arrow(draw, (1698, 455), (1698, 515))
    arrow(draw, (1698, 600), (1698, 670))
    rounded_box(draw, (1608, 670, 1790, 770), (238, 250, 240), (45, 150, 65), width=3, radius=12)
    text_center(draw, (1608, 670, 1790, 770), "QoS / Load\nRepair", F_TEXT)
    arrow(draw, (1698, 770), (1698, 840))
    rounded_box(draw, (1608, 840, 1790, 925), (235, 244, 255), (60, 115, 205), width=3, radius=12)
    text_center(draw, (1608, 840, 1790, 925), "Energy +\nFeasibility Metrics", F_TEXT)

    notes = (820, 880, 1415, 1035)
    rounded_box(draw, notes, (250, 247, 255), (115, 95, 170), width=2, radius=14)
    draw.text((850, 902), "Model Notes", font=F_HEAD, fill=(45, 40, 80))
    note_lines = [
        "Each relation layer has its own support matrix and GCN filter.",
        "Attention learns layer weights before association and BS control.",
        "Repair enforces QoS/load feasibility after learned prediction.",
    ]
    y = 945
    for line in note_lines:
        draw.ellipse((850, y + 8, 858, y + 16), fill=(80, 70, 140))
        draw.text((872, y), line, font=F_TINY, fill=(20, 20, 20))
        y += 26

    out = FIG_DIR / "fig_multilayer_ua_model.png"
    img.save(out, dpi=(300, 300))


def make_simulation_scenario():
    img = Image.new("RGB", (1900, 1000), "white")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    odraw = ImageDraw.Draw(overlay)

    title = "Dense Wireless Simulation Scenario"
    draw.text((950 - draw.textbbox((0, 0), title, font=F_TITLE)[2] / 2, 28), title, font=F_TITLE, fill=(20, 20, 20))

    map_box = (55, 95, 1045, 920)
    rounded_box(draw, map_box, (251, 253, 255), (75, 95, 125), width=3, radius=18)
    draw.text((92, 125), "Circular deployment snapshot", font=F_HEAD, fill=(35, 60, 95))
    draw.text((92, 162), "R = 650 m, B = 7, N = 50", font=F_TEXT, fill=(45, 45, 45))

    cx, cy, rr = 545, 520, 350
    odraw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(240, 247, 255, 190), outline=(60, 90, 130, 255), width=4)
    for frac in (0.33, 0.66):
        r = rr * frac
        odraw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(130, 150, 175, 150), width=2)

    hotspot = (cx + 70, cy + 40)
    odraw.ellipse((hotspot[0] - 145, hotspot[1] - 120, hotspot[0] + 145, hotspot[1] + 120), fill=(255, 210, 80, 90), outline=(220, 145, 20, 170), width=3)
    draw.text((hotspot[0] + 108, hotspot[1] - 110), "traffic\nhotspot", font=F_SMALL, fill=(135, 85, 0), spacing=2)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return cx + x / 650.0 * rr, cy - y / 650.0 * rr

    bs_world = [(0.0, 0.0)]
    ring = 0.55 * 650.0
    for idx in range(6):
        th = 2.0 * math.pi * idx / 6
        bs_world.append((ring * math.cos(th), ring * math.sin(th)))
    bs_px = [to_px(x, y) for x, y in bs_world]
    active_bs = {0, 1, 3}

    for idx, (bx, by) in enumerate(bs_px):
        radius = 116 if idx in active_bs else 88
        color = (95, 155, 230, 80) if idx in active_bs else (190, 190, 190, 45)
        outline = (80, 130, 205, 120) if idx in active_bs else (145, 145, 145, 100)
        odraw.ellipse((bx - radius, by - radius, bx + radius, by + radius), fill=color, outline=outline, width=2)

    rng = random.Random(11)
    ues: list[tuple[float, float, float]] = []
    for _ in range(64):
        th = rng.uniform(0.0, 2.0 * math.pi)
        rad = 650.0 * math.sqrt(rng.random())
        x, y = rad * math.cos(th), rad * math.sin(th)
        hx, hy = 120.0, -80.0
        demand = 0.25 + 0.75 * math.exp(-math.hypot(x - hx, y - hy) / 360.0)
        demand += rng.uniform(0.0, 0.25)
        ues.append((x, y, min(demand, 1.0)))

    # Draw a subset of association links before nodes.
    for idx, (x, y, demand) in enumerate(ues):
        if idx % 3 != 0:
            continue
        ux, uy = to_px(x, y)
        nearest = min(active_bs, key=lambda b: (ux - bs_px[b][0]) ** 2 + (uy - bs_px[b][1]) ** 2)
        bx, by = bs_px[nearest]
        odraw.line((ux, uy, bx, by), fill=(85, 115, 150, 70), width=2)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    for x, y, demand in ues:
        ux, uy = to_px(x, y)
        size = 5 + int(7 * demand)
        red = int(80 + 170 * demand)
        green = int(150 - 55 * demand)
        draw.ellipse((ux - size, uy - size, ux + size, uy + size), fill=(red, green, 70), outline=(50, 80, 50), width=1)

    for idx, (bx, by) in enumerate(bs_px):
        if idx in active_bs:
            fill, outline, text = (35, 105, 190), (20, 65, 130), "ON"
        else:
            fill, outline, text = (175, 175, 175), (105, 105, 105), "SLEEP"
        draw.rounded_rectangle((bx - 24, by - 24, bx + 24, by + 24), radius=8, fill=fill, outline=outline, width=3)
        draw.text((bx - 12, by - 11), "BS", font=F_SMALL, fill="white")
        draw.text((bx - 25, by + 30), f"{idx}: {text}", font=F_TINY, fill=(30, 30, 30))

    # Scale bar.
    draw.line((205, 865, 420, 865), fill=(30, 30, 30), width=4)
    draw.line((205, 855, 205, 875), fill=(30, 30, 30), width=3)
    draw.line((420, 855, 420, 875), fill=(30, 30, 30), width=3)
    draw.text((262, 875), "400 m", font=F_SMALL, fill=(30, 30, 30))

    legend = (700, 790, 1010, 895)
    rounded_box(draw, legend, (255, 255, 255), (145, 145, 145), width=2, radius=12)
    draw.text((725, 810), "Legend", font=F_HEAD, fill=(35, 35, 35))
    draw.rounded_rectangle((725, 850, 755, 880), radius=6, fill=(35, 105, 190))
    draw.text((765, 852), "base station", font=F_SMALL, fill=(35, 35, 35))
    draw.ellipse((875, 853, 895, 873), fill=(220, 105, 70), outline=(50, 80, 50))
    draw.text((905, 852), "UE demand", font=F_SMALL, fill=(35, 35, 35))

    # Right side explanatory panels.
    right_x0, right_x1 = 1100, 1815
    panels = [
        ((right_x0, 125, right_x1, 265), "Snapshot Inputs",
         "UE positions, time-varying traffic demands, BS locations, and channel-gain matrix."),
        ((right_x0, 310, right_x1, 450), "Radio Channel",
         "Path loss, lognormal shadowing, Rayleigh fading, RSRP, SINR, and achievable rate."),
        ((right_x0, 495, right_x1, 635), "Multilayer Graph",
         "Association, interference, load-pressure, and temporal similarity layers over UE nodes."),
        ((right_x0, 680, right_x1, 820), "Control Decision",
         "Select active/sleep BS states, assign UEs to BSs, then check QoS, load, and energy."),
    ]
    fills = [(248, 248, 248), (255, 248, 235), (242, 246, 255), (238, 250, 240)]
    outlines = [(85, 85, 85), (210, 135, 35), (75, 115, 210), (45, 150, 65)]
    for (xy, head, body), fill, outline in zip(panels, fills, outlines):
        rounded_box(draw, xy, fill, outline, width=3, radius=16)
        draw.text((xy[0] + 28, xy[1] + 24), head, font=F_HEAD, fill=outline)
        lines: list[str] = []
        for part in wrap(body, width=54):
            lines.append(part)
        y = xy[1] + 67
        for line in lines:
            draw.text((xy[0] + 28, y), line, font=F_TEXT, fill=(30, 30, 30))
            y += 28

    for y0, y1 in [(265, 310), (450, 495), (635, 680)]:
        arrow(draw, ((right_x0 + right_x1) / 2, y0 + 6), ((right_x0 + right_x1) / 2, y1 - 6), color=(90, 90, 90), width=5)

    equation = "Energy = active fixed power + load-dependent power + sleep power"
    draw.text((1130, 870), equation, font=F_MATH, fill=(45, 45, 45))

    out = FIG_DIR / "fig_simulation_scenario.png"
    img.save(out, dpi=(300, 300))


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_flowchart()
    make_simulation_scenario()
    make_model()
    print(f"Wrote figures to {FIG_DIR}")
