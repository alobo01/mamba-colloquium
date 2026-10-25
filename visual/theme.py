"""Palette, fonts and text helpers shared by every slide."""

import pathlib

import numpy as np
import PIL.Image as _PI
from manim import BOLD, NORMAL, MathTex, Text, config

# Manim 0.19 pastes image mobjects onto Image.fromarray(zeros(h, w), mode="RGBA"), which newer
# Pillow rejects; building the blank canvas with Image.new is equivalent.
_fromarray = _PI.fromarray


def _fromarray_compat(obj, mode=None):
    if mode == "RGBA" and getattr(obj, "ndim", 0) == 2:
        return _PI.new("RGBA", (obj.shape[1], obj.shape[0]))
    return _fromarray(obj, mode=mode)


_PI.fromarray = _fromarray_compat

# ------------------------------------------------------------------ palette
BG0, BG1 = "#1B2338", "#090B12"          # centre / edge of the backdrop
RESID = "#F4F6FA"                         # residual stream, main text, outputs
TOKEN = "#CFD6E4"
Q_COL = "#35D6D0"                         # query  q, Q  (Mamba's C)
K_COL = "#B48CFF"                         # key    k, K  (Mamba's B)
V_COL = "#FFCB57"                         # value  v, V  (Mamba's x)
ATTN = "#5B9BFF"                          # Theo, softmax, attention windows
MAMBA = "#FF8A65"                         # Manu, and every recurrent state S, h
MAMBA2 = "#5EE0B0"                        # Max, and every decay a, L, gamma
MOE = "#FFC857"                           # experts and the router
FIN = "#FF6FA8"                           # "bank" = money
RIVER = "#4DB6FF"                         # "bank" = river
PROB = "#FF6B81"                          # a limitation
FADE = "#7C879B"
SOFT = "#26304A"                          # card fill / dim

STATE = MAMBA
DECAY = MAMBA2
ROOM = [ATTN, Q_COL, MAMBA2, V_COL, MAMBA, FIN, K_COL, MOE]   # expert colours

config.background_color = BG1
FONT = "Inter"
ROOT = pathlib.Path(__file__).resolve().parents[1]
TOTAL_SCENES = len(list((ROOT / "slides").glob("s[0-9][0-9]_*.py")))


def T(text: str, size: float = 28, color=RESID, bold: bool = False, **kw) -> Text:
    """Body text in the deck font."""
    return Text(text, font=FONT, font_size=size, color=color, weight=BOLD if bold else NORMAL, **kw)


def M(*tex: str, size: float = 40, color=RESID, **kw) -> MathTex:
    """Plain, single-colour maths. Use `ceq` from visual.equations for anything with symbols."""
    return MathTex(*tex, font_size=size, color=color, **kw)


def fmt_weight(w: float) -> str:
    """0.375 -> '.375', 1 -> '1': how decay weights are printed on screen."""
    return "1" if np.isclose(w, 1) else f"{w:.3g}".lstrip("0")
