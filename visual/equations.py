"""Colour-coded equations and the two derivation moves used across the deck.

Every symbol keeps one colour on every slide:

    q Q c  query (cyan)      k K b  key (purple)     v V x  value (yellow)
    S h    recurrent state (coral)                  a L γ Δ  decay / mask (mint)
    softmax, W (window)  attention (blue)           y Y O r  outputs, residual (white)

Write equations as a list of atomic parts, one symbol per part, and `ceq` colours them.
"""

import re

from manim import (DOWN, UP, FadeIn, FadeOut, LaggedStart, MathTex, ReplacementTransform,
                   SurroundingRectangle, TransformFromCopy, VGroup)

from .theme import ATTN, DECAY, K_COL, Q_COL, RESID, STATE, V_COL

_FIRST = {"q": Q_COL, "Q": Q_COL, "c": Q_COL, "k": K_COL, "K": K_COL, "b": K_COL,
          "v": V_COL, "V": V_COL, "x": V_COL, "a": DECAY, "L": DECAY, "S": STATE, "h": STATE,
          "s": STATE, "W": ATTN}
_SIZING = re.compile(r"\\(?:big|Big|bigg|Bigg|left|right|!|,|;|quad|qquad)|[\s(){}\[\]]")


def role(part: str):
    """Colour for one equation part, from its leading symbol (None = default white)."""
    s = _SIZING.sub("", part)
    if "softmax" in s:
        return ATTN
    if s.startswith((r"\gamma", r"\Delta")):
        return DECAY
    if not s or s[0] == "\\":
        return None
    return _FIRST.get(s[0])


def ceq(*parts, size: float = 40, color=RESID) -> MathTex:
    """Colour-coded MathTex. Each part is a TeX string or (tex, colour) to override the rule."""
    texs, cols = [], []
    for p in parts:
        tex, col = (p if isinstance(p, tuple) else (p, None))
        texs.append(tex)
        cols.append(col or role(tex) or color)
    m = MathTex(*texs, font_size=size, color=color)
    for sub, c in zip(m, cols):
        sub.set_color(c)
    return m


def residual_eq(size: float = 34) -> MathTex:
    """r⁽ˡ⁺¹⁾ = r⁽ˡ⁾ + block(norm r⁽ˡ⁾): the residual update, the block coloured as a writer (gold)."""
    from .theme import MOE
    return ceq(r"r^{(\ell+1)}", "=", r"r^{(\ell)}", "+", (r"\mathrm{block}_\ell", MOE), r"\big(\mathrm{norm}\,", r"r^{(\ell)}",
               r"\big)", size=size)


def boxed(eq: MathTex, color=DECAY, buff: float = 0.25) -> VGroup:
    """An equation framed as a result."""
    box = SurroundingRectangle(eq, color=color, buff=buff, corner_radius=0.2, stroke_width=4)
    return VGroup(box, eq)


def morph(scene, src: MathTex, dst: MathTex, bring: dict | None = None, run_time: float = 1.2) -> MathTex:
    """Rewrite `src` into `dst` in place, one substitution step.

    Parts with identical TeX glide to their new place (first unused match, left to right).
    `bring` maps a dst index to an on-screen mobject that is copied into that slot (the
    substituted expression flies in). Every other src part fades out, every other dst part
    fades in. Returns `dst`, which replaces `src` on screen.
    """
    bring = bring or {}
    used_src, pairs = set(), {}
    for j, part in enumerate(dst):
        if j in bring:
            continue
        for i, sp in enumerate(src):
            if i not in used_src and sp.get_tex_string() == part.get_tex_string():
                used_src.add(i)
                pairs[j] = i
                break
    anims = [ReplacementTransform(src[i], dst[j]) for j, i in pairs.items()]
    anims += [FadeOut(sp, shift=UP * 0.15) for i, sp in enumerate(src) if i not in used_src]
    anims += [TransformFromCopy(m, dst[j]) for j, m in bring.items()]
    anims += [FadeIn(part, shift=DOWN * 0.1) for j, part in enumerate(dst) if j not in pairs and j not in bring]
    scene.play(*anims, run_time=run_time)
    scene.remove(src, *src.submobjects)
    return settle(scene, dst)


def settle(scene, mob):
    """After part-wise animations, keep `mob` on screen as one object (not as loose parts)."""
    scene.remove(*mob.submobjects)
    scene.add(mob)
    return mob


def collapse(scene, steps: list, result, run_time: float = 1.3):
    """Substitute a whole derivation by its result: every step slides into `result` and vanishes."""
    target = result.get_center()
    scene.play(LaggedStart(*[s.animate.move_to(target).scale(0.6).set_opacity(0) for s in steps], lag_ratio=0.12),
               FadeIn(result, scale=0.85), run_time=run_time)
    scene.remove(*[m for s in steps for m in s.get_family()])
    return result
