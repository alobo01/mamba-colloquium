"""The `Deck` scene base class: backdrop, progress dots, title, presenter clicks, layout check."""

import os
import re

import numpy as np
from manim import (BOLD, DOWN, RIGHT, UL, Circle, FadeIn, FadeOut, Group, ImageMobject, MathTex, RoundedRectangle,
                   Scene, Tex, Text, ThreeDScene, VGroup, config)

from .theme import BG0, BG1, FADE, RESID, ROOT, TOTAL_SCENES, T

# Video export uses plain Scenes. Set MANIM_DECK=slides when running
# `manim-slides render` to get click-stepped presentations instead.
_Slide, _ThreeDSlide = Scene, ThreeDScene
if os.environ.get("MANIM_DECK") == "slides":
    from manim_slides import Slide as _Slide, ThreeDSlide as _ThreeDSlide


def _backdrop() -> ImageMobject:
    """Radial gradient with a faint dot grid, generated once into media/backdrop.png."""
    path = ROOT / "media" / "backdrop.png"
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
        dots = ((x % 64 == 32) & (y % 64 == 32))[..., None]
        img = np.where(dots, img + 14, img)
        Image.fromarray(np.clip(img, 0, 255).astype("uint8")).convert("RGBA").save(path)
    im = ImageMobject(str(path))
    im.scale_to_fit_width(config.frame_width).set_z_index(-100)
    return im


class _DeckMixin:
    """Shared behaviour. `pause()` = one presenter click in the live deck, a short hold in the video."""

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

    # ---------------------------------------------------------------- beats
    def title(self, text: str, color=RESID, accent=None) -> VGroup:
        """Slide heading in the top-left corner with a short accent bar."""
        t = T(text, 42, color, bold=True).to_corner(UL, buff=0.5)
        bar = RoundedRectangle(width=1.0, height=0.08, corner_radius=0.04, stroke_width=0,
                               fill_color=accent or color, fill_opacity=1).next_to(t, DOWN, buff=0.14, aligned_edge=[-1, 0, 0])
        g = VGroup(t, bar)
        self.play(FadeIn(g, shift=RIGHT * 0.3), run_time=0.7)
        self._title = g
        return g

    def pause(self, t: float = 0.9):
        """One presenter click (a ▶ in TALK_SCRIPT.md)."""
        self.wait(t)
        self.check_layout("pause")
        if hasattr(self, "next_slide"):
            self.next_slide()

    def content(self) -> list:
        """Everything on screen except the backdrop and progress dots."""
        chrome = set(id(m) for c in self._chrome.submobjects for m in c.get_family()) | {id(self._chrome)}
        return [m for m in self.mobjects if id(m) not in chrome]

    def clear_all(self, run_time: float = 0.6):
        """Fade out everything but the chrome."""
        self.check_layout("end")
        gone = self.content()
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)

    def clear_body(self, run_time: float = 0.6):
        """Fade out everything but the chrome and the slide title (start a new beat on the same slide)."""
        keep = getattr(self, "_title", None)
        gone = [m for m in self.content() if m is not keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)

    # ------------------------------------------------------------------ QA
    def check_layout(self, tag: str = ""):
        """DECK_CHECK=1: report text boxes that overlap each other or leave the frame."""
        if not os.environ.get("DECK_CHECK"):
            return
        texts = [m for m in self.mobjects for m in m.get_family() if isinstance(m, (Text, MathTex, Tex))]
        texts = [m for m in texts if max(max(c.get_fill_opacity(), c.get_stroke_opacity()) for c in m.get_family()) > 0.35
                 and m.width > 0.01]
        inner = {id(c) for m in texts for c in m.get_family()[1:]}
        texts = [m for m in texts if id(m) not in inner]
        W, H = config.frame_width / 2, config.frame_height / 2
        name = type(self).__name__
        for m in texts:
            l, r, u, d = m.get_left()[0], m.get_right()[0], m.get_top()[1], m.get_bottom()[1]
            if l < -W + 0.05 or r > W - 0.05 or u > H - 0.05 or d < -H + 0.3:
                print(f"CHECK {name} {tag} OFF-FRAME  {self._label(m)}")
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                ox = min(a.get_right()[0], b.get_right()[0]) - max(a.get_left()[0], b.get_left()[0])
                oy = min(a.get_top()[1], b.get_top()[1]) - max(a.get_bottom()[1], b.get_bottom()[1])
                if ox > 0.04 and oy > 0.04:
                    print(f"CHECK {name} {tag} OVERLAP    {self._label(a)}  <>  {self._label(b)}")

    @staticmethod
    def _label(m) -> str:
        t = getattr(m, "text", None) or getattr(m, "tex_string", None) or type(m).__name__
        return f"'{str(t)[:28]}'@({m.get_center()[0]:.1f},{m.get_center()[1]:.1f})"


class Deck(_DeckMixin, _Slide):
    """Base class of every slide."""


class Deck3D(_DeckMixin, _ThreeDSlide):
    """Base class for slides that need a 3D camera."""
