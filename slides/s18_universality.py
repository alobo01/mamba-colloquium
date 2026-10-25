"""Slide 18 · Universality: all three families can approximate anything — the question is the size.

Yun et al. 2020 (Transformers are universal seq-to-seq approximators), Orvieto et al. 2023 (linear recurrence +
position-wise MLP is universal), Cirone et al. 2024 (selective SSMs through rough-path signatures).

Clicks:  ▶1 the statement   ▶2 three proofs, three mechanisms   ▶3 the price: a lossless state is a KV cache
         ▶4 what the lower bounds say
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

XS = [-4.75, 0, 4.75]


def statement() -> MathTex:
    """∀ε ∃θ: sup ‖f − f_θ‖ < ε."""
    return ceq(r"\forall\varepsilon>0\ \ \exists\,\theta:\quad", r"\sup_{x}\,\big\|\,f(x)-", (r"f_\theta(x)", MAMBA2), r"\,\big\|<\varepsilon",
               size=40)


def build_card(x: float, title: str, who: str, color, visual: Mobject, note: str) -> VGroup:
    fr = card(4.4, 3.9, color, 0.07).move_to([x, -0.75, 0])
    head = VGroup(T(title, 22, color, bold=True), T(who, 15, FADE)).arrange(DOWN, buff=0.08).move_to(fr.get_top() + DOWN * 0.5)
    visual.move_to(fr.get_center() + DOWN * 0.05)
    sub = T(note, 16, TOKEN).move_to(fr.get_bottom() + UP * 0.4)
    if sub.width > 4.1:
        sub.scale_to_fit_width(4.1)
    return VGroup(fr, head, visual, sub)


def contexts_lookup() -> VGroup:
    """Tokens get unique context tags (attention), then a table maps tags to outputs (FFN)."""
    toks = VGroup(*[chip(t, TOKEN, 18) for t in ("a", "b", "a")]).arrange(DOWN, buff=0.12)
    tags = VGroup(*[Dot(radius=0.12, color=c) for c in (Q_COL, K_COL, V_COL)]).arrange(DOWN, buff=0.38)
    table = VGroup(*[Square(0.28, stroke_color=MOE, stroke_width=1.5, fill_color=MOE, fill_opacity=o)
                     for o in (0.1, 0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.1, 0.6)]).arrange_in_grid(3, 3, buff=0.04)
    g = VGroup(toks, tags, table).arrange(RIGHT, buff=0.45)
    links = VGroup(*[Arrow(toks[i].get_right(), tags[i].get_left(), buff=0.06, stroke_width=2.5, tip_length=0.1, color=ATTN) for i in range(3)])
    return VGroup(g, links)


def compress_decode() -> VGroup:
    """A sequence squeezed into one state vector, then decoded by an MLP hourglass."""
    seq = VGroup(*[Square(0.22, stroke_color=TOKEN, stroke_width=1.5, fill_color=TOKEN, fill_opacity=0.3) for _ in range(5)]).arrange(DOWN, buff=0.05)
    st = vec(5, STATE, 0.2, vals=[0.9, 0.4, 0.7, 0.3, 0.8])
    mlp = hourglass(ORIGIN, MOE, 0.35, 1.1).rotate(-PI / 2)
    g = VGroup(seq, st, mlp).arrange(RIGHT, buff=0.45)
    links = VGroup(Arrow(seq.get_right(), st.get_left(), buff=0.06, stroke_width=2.5, tip_length=0.1, color=STATE),
                   Arrow(st.get_right(), mlp.get_left(), buff=0.06, stroke_width=2.5, tip_length=0.1, color=MOE))
    return VGroup(g, links)


def gated_path() -> VGroup:
    """A wiggly input path read through input-dependent gates (dials)."""
    path = VMobject(stroke_color=MAMBA2, stroke_width=4).set_points_smoothly(
        [np.array([x, 0.35 * np.sin(2.3 * x) + 0.15 * np.cos(5 * x), 0]) for x in np.linspace(-1.4, 1.4, 12)])
    dials = VGroup(*[dial(DECAY, 0.18).rotate(-a, about_point=ORIGIN) for a in (0.3, 1.2, -0.6, 0.8)]).arrange(RIGHT, buff=0.35)
    return VGroup(path, dials.next_to(path, DOWN, buff=0.35))


def animate_proofs(scene: Deck) -> VGroup:
    cards = VGroup(
        build_card(XS[0], "Transformers", "Yun et al. 2020", ATTN, contexts_lookup(), "attention tags each context, FFN looks it up"),
        build_card(XS[1], "linear RNN + MLP", "Orvieto et al. 2023", STATE, compress_decode(), "the state compresses, the MLP decodes"),
        build_card(XS[2], "selective SSMs", "Cirone et al. 2024", MAMBA2, gated_path(), "input-dependent gates read the path"),
    )
    scene.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.3), run_time=2.0)
    return cards


def animate_price(scene: Deck):
    """To compress losslessly, the notebook has to grow until it is as big as Theo's wall."""
    wall = VGroup(*[token_card("", w=0.62, h=0.85) for _ in range(8)]).arrange(RIGHT, buff=0.08).move_to(LEFT * 2.2 + UP * 1.2)
    lab_w = T("Theo: one card per token", 20, ATTN, bold=True).next_to(wall, UP, buff=0.25).align_to(wall, LEFT)
    note = sheet(2, 2, 0.42, STATE, 0.6).move_to(wall, aligned_edge=LEFT).shift(DOWN * 1.75)
    lab_n = T("a notebook that loses nothing", 20, STATE, bold=True).next_to(note, UP, buff=0.2).align_to(wall, LEFT)
    scene.play(FadeIn(wall, lag_ratio=0.05), FadeIn(lab_w), FadeIn(note), FadeIn(lab_n))
    big = sheet(2, 11, 0.42, STATE, 0.6).move_to(note, aligned_edge=UL)
    big.stretch_to_fit_width(wall.width).move_to(note, aligned_edge=UL)
    scene.play(Transform(note, big), run_time=1.6)
    eq = ceq((r"\text{state}", STATE), r"\ \propto\ ", "T", r"\cdot d", r"\ =\ ", (r"\text{a KV cache}", V_COL), size=38)
    scene.play(Write(eq.next_to(note, DOWN, buff=0.55).align_to(wall, LEFT)))
    return VGroup(wall, lab_w, note, lab_n, eq)


def animate_bounds(scene: Deck):
    """Recap of the three lower bounds from the previous slides, and the verdict."""
    rows = VGroup(
        VGroup(pill("recall", ATTN, 17), ceq((r"\text{bits of state}", STATE), r"\ge", r"n\log|\mathcal V|", size=26)),
        VGroup(pill("selective copy", V_COL, 17), ceq(r"\sum_i", (r"\log|S_i|", STATE), r"\ge", r"N\log M", size=26)),
        VGroup(pill("state tracking", PROB, 17), T("outside TC⁰", 18, TOKEN)),
    )
    for r in rows:
        r.arrange(RIGHT, buff=0.2)
    rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(RIGHT * 4.15 + UP * 0.6)
    scene.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows], lag_ratio=0.25), run_time=1.3)
    verdict = pill("universal ≠ efficient", MAMBA2, 22, filled=True).move_to(RIGHT * 4.35 + DOWN * 1.4)
    scene.play(FadeIn(verdict, scale=0.9), FadeIn(glow(verdict[0], MAMBA2, 3)))


class S18Universality(Deck):
    def construct(self):
        self.title("Universal… but at what size?", MAMBA2)
        st = statement().move_to(UP * 2.05)
        self.play(Write(st))
        self.pause()
        animate_proofs(self)
        self.pause()
        self.clear_body()
        animate_price(self)
        self.pause()
        animate_bounds(self)
        self.pause(1.4)
        self.clear_all()
