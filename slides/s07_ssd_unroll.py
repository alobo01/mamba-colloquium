"""Slide 07 · Unroll the notebook one step at a time, then replace the steps by the closed form.

Clicks:  ▶1 S₁, S₂, S₃ by substitution (token weights appear)   ▶2 collapse to Σ (a⋯a) k vᵀ
         ▶3 read it: y_t, then all rows at once, collapse to Y = (L ⊙ QKᵀ)V   ▶4 the matrices   ▶5 vs softmax
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *
from math_contracts.test_ssd_equivalence import decay_mask

A = np.array([0.9, 0.8, 0.5, 0.75])      # decays used for every number on this slide
EQ_SIZE = 44
LEFT_EDGE = -6.3


def at_row(m, y: float):
    """Left-align an equation in the derivation column at height y."""
    return m.move_to([0, y, 0]).align_to([LEFT_EDGE, 0, 0], LEFT)


def animate_unroll(scene: Deck) -> list:
    """S₁ → S₂ → S₃, each new line substitutes the previous one. Returns the three final lines."""
    e1 = at_row(ceq("S_1", "=", "k_1", r"v_1^\top", size=EQ_SIZE), 1.8)
    scene.play(Write(e1))

    e2a = at_row(ceq("S_2", "=", "a_2", "S_1", "+", "k_2", r"v_2^\top", size=EQ_SIZE), 0.65)
    scene.play(Write(e2a))
    e2 = at_row(ceq("S_2", "=", "a_2", "k_1", r"v_1^\top", "+", "k_2", r"v_2^\top", size=EQ_SIZE), 0.65)
    morph(scene, e2a, e2, bring={3: e1[2], 4: e1[3]})

    e3a = at_row(ceq("S_3", "=", "a_3", "S_2", "+", "k_3", r"v_3^\top", size=EQ_SIZE), -0.5)
    scene.play(Write(e3a))
    e3 = at_row(ceq("S_3", "=", "a_3", "a_2", "k_1", r"v_1^\top", "+", "a_3", "k_2", r"v_2^\top", "+", "k_3", r"v_3^\top",
                    size=EQ_SIZE), -0.5)
    morph(scene, e3a, e3, bring={3: e2[2], 4: e2[3], 5: e2[4], 7: e3a[2], 8: e2[6], 9: e2[7]})
    return [e1, e2, e3]


def animate_token_weights(scene: Deck) -> VGroup:
    """Three tokens dimmed by how much of them survives in S₃: a₃a₂, a₃, 1."""
    names = ["Leo", "hid", "key"]
    weights = [A[2] * A[1], A[2], 1.0]
    syms = [("a_3", "a_2"), ("a_3",), ("1",)]
    col = VGroup()
    for n, w, s in zip(names, weights, syms):
        c = chip(n, TOKEN, 24)
        c[0].set_stroke(opacity=0.25 + 0.75 * w)
        c[1].set_opacity(0.25 + 0.75 * w)
        sym = ceq(*s, size=30)
        num = T(fmt_weight(w), 22, DECAY, bold=True)
        col.add(VGroup(c, sym, num).arrange(DOWN, buff=0.18))
    col.arrange(RIGHT, buff=0.6).move_to(RIGHT * 4.6 + UP * 0.6)
    now = T("now", 18, FADE).next_to(col[2], UP, buff=0.2)
    scene.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in col], lag_ratio=0.25), FadeIn(now), run_time=1.4)
    return VGroup(col, now)


def closed_form_state() -> VGroup:
    """Boxed S_t = Σ_{s≤t} (a_{s+1}⋯a_t) k_s v_sᵀ; the decay product is part [1][3]."""
    return boxed(ceq("S_t", "=", r"\sum_{s\le t}", r"a_{s+1}\cdots a_t", r"\,k_s", r"v_s^\top", size=42))


def animate_read(scene: Deck, state_box: VGroup) -> VGroup:
    """y_t = S_tᵀq_t → Σ L_ts (q_tᵀk_s) v_s → Y = (L ⊙ QKᵀ)V, then all collapse into the boxed result."""
    e4a = at_row(ceq("y_t", "=", r"S_t^\top", "q_t", size=EQ_SIZE + 2), 0.2)
    scene.play(Write(e4a))
    e4 = at_row(ceq("y_t", "=", r"\sum_{s\le t}", "L_{ts}", r"\,(", r"q_t^\top", "k_s", ")", "v_s", size=EQ_SIZE + 2), 0.2)
    sb = state_box[1]
    morph(scene, e4a, e4, bring={2: sb[2], 3: sb[3], 5: e4a[3], 6: sb[4], 8: sb[5]})
    lts = ceq("L_{ts}", "=", r"a_{s+1}\cdots a_t", size=36).next_to(e4, RIGHT, buff=0.8)
    scene.play(FadeIn(lts, shift=LEFT * 0.2))
    scene.pause()
    e5 = at_row(ceq("Y", "=", r"\big(", "L", r"\odot", "Q", r"K^\top", r"\big)", "V", size=EQ_SIZE + 2), -1.2)
    scene.play(*[TransformFromCopy(e4[i], e5[j]) for i, j in ((3, 3), (5, 5), (6, 6), (8, 8))],
               FadeIn(VGroup(e5[0], e5[1], e5[2], e5[4], e5[7])), run_time=1.3)
    settle(scene, e5)
    final = boxed(ceq("Y", "=", r"\big(", "L", r"\odot", "Q", r"K^\top", r"\big)", "V", size=48)).move_to(UP * 2.25)
    collapse(scene, [state_box, e4, lts, e5], final)
    return final


def animate_matrices(scene: Deck) -> VGroup:
    """Heatmaps: L (with numbers) ⊙ QKᵀ = the masked score matrix."""
    L = decay_mask(A)
    rng = np.random.default_rng(7)
    QK = np.tril(rng.uniform(0.2, 1, (4, 4)))
    Lh = heat(L, 0.62, pos=DECAY, numbers=True, nsize=18).move_to(LEFT * 3.9 + DOWN * 0.75)
    QKh = heat(QK, 0.62, pos=ATTN).move_to(RIGHT * 0.1 + DOWN * 0.75)
    res = heat(L * QK, 0.62, pos=V_COL).move_to(RIGHT * 4.3 + DOWN * 0.75)
    labs = VGroup(ceq("L", size=40).next_to(Lh, UP, buff=0.25), ceq("Q", r"K^\top", size=40).next_to(QKh, UP, buff=0.25),
                  ceq(r"L\odot QK^\top", size=36, color=V_COL).next_to(res, UP, buff=0.25))
    ops = VGroup(M(r"\odot", size=56).move_to((Lh.get_right() + QKh.get_left()) / 2),
                 M("=", size=56).move_to((QKh.get_right() + res.get_left()) / 2))
    scene.play(FadeIn(Lh, lag_ratio=0.03), FadeIn(labs[0]), run_time=1.2)
    scene.play(FadeIn(ops[0]), FadeIn(QKh), FadeIn(labs[1]))
    scene.play(FadeIn(ops[1]), FadeIn(res), FadeIn(labs[2]))
    hint = ceq(r".3", "=", ".8", r"\cdot", ".5", r"\cdot", ".75", size=30, color=DECAY).next_to(Lh, DOWN, buff=0.3)
    scene.play(Indicate(VGroup(Lh[-8], Lh[-7]), color=DECAY), Write(hint))   # cell L[3,0]
    return VGroup(Lh, QKh, res, labs, ops, hint)


def animate_softmax_contrast(scene: Deck):
    """A small softmax map beside it: same recipe, different mask."""
    sm = heat(causal_softmax(4, 2), 0.5, pos=ATTN).scale(0.9)
    ss = VGroup(T("softmax attention", 20, ATTN, bold=True), sm).arrange(DOWN, buff=0.15).scale(0.75).move_to(RIGHT * 5.8 + UP * 2.4)
    scene.play(FadeIn(ss, shift=LEFT * 0.3))


class S07SSDUnroll(Deck):
    def construct(self):
        self.title("Unroll it: attention with a decay mask", MAMBA2)
        lines = animate_unroll(self)
        weights = animate_token_weights(self)
        self.pause()
        state_box = closed_form_state().move_to(UP * 1.75)
        self.play(FadeOut(weights))
        collapse(self, lines, state_box)
        self.pause()
        final = animate_read(self, state_box)
        self.pause()
        animate_matrices(self)
        self.pause()
        animate_softmax_contrast(self)
        self.pause(1.4)
        self.clear_all()
