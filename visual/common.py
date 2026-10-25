"""Shared look and helpers for the compact talk (16 scenes, ~5 min)."""

from manim import *
import numpy as np
import os, pathlib, re

# Video export uses plain Scenes. Set MANIM_DECK=slides when running
# `manim-slides render` to get click-stepped presentations instead.
_Slide, _ThreeDSlide = Scene, ThreeDScene
if os.environ.get("MANIM_DECK") == "slides":
    from manim_slides import Slide as _Slide, ThreeDSlide as _ThreeDSlide

# Manim 0.19 pastes image mobjects onto Image.fromarray(zeros(h, w), mode="RGBA"), which newer
# Pillow rejects; building the blank canvas with Image.new is equivalent.
import PIL.Image as _PI
_fromarray = _PI.fromarray


def _fromarray_compat(obj, mode=None):
    if mode == "RGBA" and getattr(obj, "ndim", 0) == 2:
        return _PI.new("RGBA", (obj.shape[1], obj.shape[0]))
    return _fromarray(obj, mode=mode)


_PI.fromarray = _fromarray_compat

# ------------------------------------------------------------------ palette
BG0, BG1 = "#1B2338", "#090B12"          # centre / edge of the backdrop
RESID = "#F4F6FA"                         # residual stream, main text
TOKEN = "#CFD6E4"
Q_COL = "#35D6D0"
K_COL = "#B48CFF"
V_COL = "#FFCB57"
ATTN = "#5B9BFF"                          # Theo
MAMBA = "#FF8A65"                         # Manu
MAMBA2 = "#5EE0B0"                        # Max
MOE = "#FFC857"
FIN = "#FF6FA8"                           # "bank" = money
RIVER = "#4DB6FF"                         # "bank" = river
FADE = "#7C879B"
SOFT = "#26304A"                          # card fill / dim

config.background_color = BG1
FONT = "Inter"
TOTAL_SCENES = 16


def T(text, size=28, color=RESID, bold=False, **kw):
    return Text(text, font=FONT, font_size=size, color=color, weight=BOLD if bold else NORMAL, **kw)


def M(*tex, size=40, color=RESID, **kw):
    return MathTex(*tex, font_size=size, color=color, **kw)


# --------------------------------------------------------------- background
def _backdrop():
    path = pathlib.Path(__file__).resolve().parents[1] / "media" / "backdrop.png"
    if not path.exists():
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        W, H = 1920, 1080
        y, x = np.mgrid[0:H, 0:W]
        r = np.sqrt(((x / W - 0.5) * 1.5) ** 2 + (y / H - 0.5) ** 2)
        r = np.clip(r / 0.85, 0, 1)[..., None]
        c0 = np.array([int(BG0[i:i + 2], 16) for i in (1, 3, 5)])
        c1 = np.array([int(BG1[i:i + 2], 16) for i in (1, 3, 5)])
        img = c0 * (1 - r) + c1 * r
        dots = ((x % 64 == 32) & (y % 64 == 32))[..., None]       # faint dot grid
        img = np.where(dots, img + 14, img)
        Image.fromarray(np.clip(img, 0, 255).astype("uint8")).convert("RGBA").save(path)
    im = ImageMobject(str(path))
    im.scale_to_fit_width(config.frame_width).set_z_index(-100)
    return im


class _DeckMixin:
    """`pause()` = one presenter click in the live deck, a short hold in the video."""

    def setup(self):
        super().setup()
        self._chrome = Group(_backdrop())
        m = re.match(r"S(\d+)", type(self).__name__)
        if m:
            i = int(m.group(1))
            dots = VGroup(*[Circle(radius=0.07 if k + 1 == i else 0.04, stroke_width=0,
                                   fill_color=RESID if k + 1 == i else FADE,
                                   fill_opacity=1 if k + 1 <= i else 0.35) for k in range(TOTAL_SCENES)])
            dots.arrange(RIGHT, buff=0.2).to_edge(DOWN, buff=0.18).set_z_index(50)
            self._chrome.add(dots)
        self.add(self._chrome)

    def check_layout(self, tag=""):
        """DECK_CHECK=1: report text boxes that overlap each other or leave the frame."""
        if not os.environ.get("DECK_CHECK"):
            return
        texts = [m for m in self.mobjects for m in m.get_family() if isinstance(m, (Text, MathTex, Tex))]
        texts = [m for m in texts if max(max(c.get_fill_opacity(), c.get_stroke_opacity()) for c in m.get_family()) > 0.35 and m.width > 0.01]
        inner = set()
        for m in texts:
            for c in m.get_family()[1:]:
                inner.add(id(c))
        texts = [m for m in texts if id(m) not in inner]
        W, H = config.frame_width / 2, config.frame_height / 2
        name = type(self).__name__
        for m in texts:
            l, r, u, d = m.get_left()[0], m.get_right()[0], m.get_top()[1], m.get_bottom()[1]
            if l < -W + 0.05 or r > W - 0.05 or u > H - 0.05 or d < -H + 0.05:
                print(f"CHECK {name} {tag} OFF-FRAME  {self._label(m)}")
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                ox = min(a.get_right()[0], b.get_right()[0]) - max(a.get_left()[0], b.get_left()[0])
                oy = min(a.get_top()[1], b.get_top()[1]) - max(a.get_bottom()[1], b.get_bottom()[1])
                if ox > 0.04 and oy > 0.04:
                    print(f"CHECK {name} {tag} OVERLAP    {self._label(a)}  <>  {self._label(b)}")

    @staticmethod
    def _label(m):
        t = getattr(m, "text", None) or getattr(m, "tex_string", None) or type(m).__name__
        return f"'{str(t)[:28]}'@({m.get_center()[0]:.1f},{m.get_center()[1]:.1f})"

    def pause(self, t=0.9):
        self.wait(t)
        self.check_layout("pause")
        if hasattr(self, "next_slide"):
            self.next_slide()

    def title(self, text, color=RESID, accent=None):
        t = T(text, 42, color, bold=True).to_corner(UL, buff=0.5)
        bar = RoundedRectangle(width=1.0, height=0.08, corner_radius=0.04, stroke_width=0,
                               fill_color=accent or color, fill_opacity=1).next_to(t, DOWN, buff=0.14, aligned_edge=LEFT)
        g = VGroup(t, bar)
        self.play(FadeIn(g, shift=RIGHT * 0.3), run_time=0.7)
        return g

    def clear_all(self, run_time=0.6):
        self.check_layout("end")
        gone = [m for m in self.mobjects if m not in self._chrome.submobjects and m is not self._chrome
                and not any(m in c.get_family() for c in self._chrome.submobjects)]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)


class Deck(_DeckMixin, _Slide):
    pass


class Deck3D(_DeckMixin, _ThreeDSlide):
    pass


# ------------------------------------------------------------------ shapes
def glow(mob, color=None, layers=4, base=2.0, spread=3.0, opacity=0.13):
    """Soft halo behind a shape (stroke-only copies, widest first)."""
    g = VGroup()
    for i in range(layers, 0, -1):
        c = mob.copy().set_fill(opacity=0).set_stroke(color or mob.get_color(), width=base + spread * i, opacity=opacity)
        g.add(c)
    return g


def card(w, h, color=ATTN, fill=0.16, radius=0.2, sw=2.5):
    return RoundedRectangle(width=w, height=h, corner_radius=radius, stroke_color=color, stroke_width=sw,
                            fill_color=color, fill_opacity=fill)


def pill(text, color=ATTN, size=20, filled=False, pad=0.28):
    t = T(text, size, BG1 if filled else color, bold=True)
    r = RoundedRectangle(width=t.width + pad * 2, height=t.height + 0.3, corner_radius=0.25,
                         stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=1 if filled else 0.14)
    return VGroup(r, t.move_to(r))


def chip(text, color=TOKEN, size=20):
    t = T(text, size, color)
    r = RoundedRectangle(width=t.width + 0.4, height=0.55, corner_radius=0.16, stroke_color=color,
                         stroke_width=2, fill_color=color, fill_opacity=0.1)
    return VGroup(r, t.move_to(r))


def token_card(text, w=1.2, h=1.35, key=K_COL, val=V_COL, size=20):
    body = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=TOKEN, stroke_width=2,
                            fill_color=val, fill_opacity=0.16)
    strip = Rectangle(width=w - 0.04, height=0.3, stroke_width=0, fill_color=key, fill_opacity=0.9)
    strip.move_to(body.get_top() + DOWN * 0.17)
    label = T(text, size, RESID).move_to(body).shift(DOWN * 0.12)
    return VGroup(body, strip, label)


def vec(n=4, color=ATTN, cell=0.17, vals=None, horizontal=False):
    vals = np.linspace(0.3, 1, n) if vals is None else vals
    g = VGroup(*[Square(cell, stroke_color=color, stroke_width=1.2, fill_color=color, fill_opacity=float(v)) for v in vals])
    return g.arrange(RIGHT if horizontal else DOWN, buff=0.02)


def arrow(start, end, color=RESID, sw=5, tip=0.18):
    return Arrow(start, end, buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=0.35)


def heat(data, cell=0.34, pos=MAMBA2, zero="#1E2740", stroke="#0A0D15", numbers=False, nsize=14):
    data = np.asarray(data, dtype=float)
    n, m = data.shape
    vmax = np.abs(data).max() or 1.0
    g = VGroup()
    for i in range(n):
        for j in range(m):
            v = abs(data[i, j]) / vmax
            col = interpolate_color(ManimColor(zero), ManimColor(pos), min(1, v))
            sq = Square(cell, stroke_width=1, stroke_color=stroke, fill_color=col, fill_opacity=1)
            sq.move_to([j * cell, -i * cell, 0])
            g.add(sq)
            if numbers and data[i, j] != 0:
                g.add(T(f"{data[i, j]:.3g}".lstrip("0") if abs(data[i, j]) < 1 else f"{data[i, j]:.3g}", nsize,
                        BG1 if v > 0.55 else RESID, bold=True).move_to(sq))
    return g.move_to(ORIGIN)


def causal_softmax(n, seed=0, temp=1.5):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, n)) * temp
    s[np.triu_indices(n, 1)] = -np.inf
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------- characters
def brother(name, color, prop, scale=1.0):
    head = Circle(radius=0.36, stroke_width=0, fill_color="#F6E3D2", fill_opacity=1)
    hair = Arc(radius=0.36, start_angle=0.15, angle=PI - 0.3, stroke_color=color, stroke_width=9)
    eyes = VGroup(*[Dot(radius=0.045, color=BG1).move_to(head.get_center() + RIGHT * dx + UP * 0.03) for dx in (-0.13, 0.13)])
    smile = Arc(radius=0.14, start_angle=PI * 1.15, angle=PI * 0.7, stroke_color=BG1, stroke_width=3)
    smile.move_to(head.get_center() + DOWN * 0.1)
    body = RoundedRectangle(width=1.05, height=0.95, corner_radius=0.32, stroke_width=0, fill_color=color, fill_opacity=0.95)
    body.next_to(head, DOWN, buff=-0.06)
    body.set_z_index(head.z_index - 1)
    if prop == "cards":
        p = VGroup(*[RoundedRectangle(width=0.3, height=0.42, corner_radius=0.05, stroke_color=BG1, stroke_width=1.5,
                                      fill_color=c, fill_opacity=1).rotate(a)
                     for c, a in zip((K_COL, V_COL, Q_COL, K_COL), (0.45, 0.15, -0.15, -0.45))]).arrange(RIGHT, buff=-0.12)
    elif prop == "notebook":
        p = VGroup(RoundedRectangle(width=0.7, height=0.5, corner_radius=0.06, stroke_color=BG1, stroke_width=1.5,
                                    fill_color=RESID, fill_opacity=1),
                   *[Line(LEFT * 0.22, RIGHT * 0.22, stroke_width=2, color=MAMBA).shift(UP * (0.12 - 0.12 * i)) for i in range(3)])
        p[1:].move_to(p[0])
    else:
        p = VGroup(*[Square(0.17, stroke_color=BG1, stroke_width=1, fill_color=RESID, fill_opacity=0.95)
                     for _ in range(6)]).arrange_in_grid(2, 3, buff=0.04)
    p.move_to(body.get_center() + DOWN * 0.1).set_z_index(5)
    lab = T(name, 24, color, bold=True).next_to(body, DOWN, buff=0.18)
    g = VGroup(VGroup(body, head, hair, eyes, smile), p, lab)
    return g.scale(scale)
