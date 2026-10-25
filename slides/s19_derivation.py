"""Slide 19 · Deriving Mamba-2's attention form, starting from softmax attention.

Route: softmax attention → drop the softmax (linear attention, Katharopoulos et al. 2020) → one row, regroup
(associativity) → the sum is a state (an RNN) → add a fade a_t → collapse into Y = (L ⊙ QKᵀ)V,
the same formula slide 07 reached from the notebook. Numbers are checked by math_contracts.

Clicks:  ▶1 softmax → linear   ▶2 one row, regroup   ▶3 it is an RNN   ▶4 add a fade
         ▶5 collapse, both routes meet   ▶6 numeric check + the family
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *
from math_contracts.test_ssd_equivalence import ssd_matrix, ssd_recurrent

SIZE = 38
EQ_X = -3.0
ROWS = [1.95, 1.05, 0.15, -0.75, -1.65]
STEP_NAMES = [("start", ATTN), ("drop softmax", ATTN), ("one row", Q_COL), ("name the sum", STATE), ("add a fade", DECAY)]


def at_row(m, k: int):
    """Left-align an equation on derivation row k."""
    return m.move_to([0, ROWS[k], 0]).align_to([EQ_X, 0, 0], LEFT)


def step_label(k: int) -> VGroup:
    name, c = STEP_NAMES[k]
    return pill(name, c, 17).move_to([-5.2, ROWS[k], 0])


def animate_linear(scene: Deck) -> list:
    """softmax(QKᵀ)V, then the softmax is replaced by a plain causal mask."""
    e1 = at_row(ceq("O", "=", r"\mathrm{softmax}\big(", "Q", r"K^\top", r"\big)", "V", size=SIZE), 0)
    scene.play(FadeIn(step_label(0)), Write(e1))
    e2 = at_row(ceq("O", "=", r"\big(", "Q", r"K^\top", r"\odot", (r"M_{\mathrm{causal}}", FADE), r"\big)", "V", size=SIZE), 1)
    scene.play(FadeIn(step_label(1)), *[TransformFromCopy(e1[i], e2[j]) for i, j in ((0, 0), (1, 1), (3, 3), (4, 4), (6, 8))],
               FadeIn(VGroup(e2[2], e2[5], e2[6], e2[7])), run_time=1.2)
    settle(scene, e2)
    return [e1, e2]


def animate_regroup(scene: Deck) -> MathTex:
    """Row t: Σ (q_tᵀk_s) v_s, regrouped as (Σ k_s v_sᵀ)ᵀ q_t."""
    e3a = at_row(ceq("o_t", "=", r"\sum_{s\le t}", r"(", r"q_t^\top", "k_s", ")", "v_s", size=SIZE), 2)
    scene.play(FadeIn(step_label(2)), Write(e3a))
    scene.wait(0.4)
    e3 = at_row(ceq("o_t", "=", r"\Big(", r"\sum_{s\le t}", "k_s", r"v_s^\top", r"\Big)^{\!\top}", r"q_t", size=SIZE), 2)
    morph(scene, e3a, e3, bring={5: e3a[7], 7: e3a[4]})
    return e3


def animate_rnn(scene: Deck, e3: MathTex) -> MathTex:
    """Name the bracket S_t: it grows by one outer product per token, and o_t = S_tᵀ q_t."""
    e4 = at_row(ceq("S_t", "=", "S_{t-1}", "+", "k_t", r"v_t^\top", r",\quad", "o_t", "=", r"S_t^\top", "q_t", size=SIZE), 3)
    br = Brace(e3[3:6], DOWN, buff=0.05, color=STATE)
    scene.play(FadeIn(step_label(3)), GrowFromCenter(br))
    scene.play(TransformFromCopy(e3[3:6], e4[0:6]), FadeIn(e4[6:]), FadeOut(br), run_time=1.2)
    settle(scene, e4)
    return e4


def animate_fade(scene: Deck, e4: MathTex) -> MathTex:
    """Insert a_t in front of the old state: the RNN forgets smoothly — this is SSD."""
    e5 = at_row(ceq("S_t", "=", "a_t", "S_{t-1}", "+", "k_t", r"v_t^\top", r",\quad", "o_t", "=", r"S_t^\top", "q_t", size=SIZE), 4)
    scene.play(FadeIn(step_label(4)), *[TransformFromCopy(e4[i], e5[i + (i >= 2)]) for i in range(len(e4))],
               FadeIn(e5[2], scale=1.6), run_time=1.2)
    settle(scene, e5)
    scene.play(Circumscribe(e5[2], color=DECAY, fade_out=True))
    return e5


def animate_meet(scene: Deck, steps: list) -> VGroup:
    """All five lines collapse into the boxed result; a chip from slide 07 joins it."""
    final = boxed(ceq("Y", "=", r"\big(", "L", r"\odot", "Q", r"K^\top", r"\big)", "V", size=56)).move_to(UP * 0.9)
    collapse(scene, steps, final, run_time=1.6)
    left = pill("from attention (this slide)", ATTN, 18).move_to(LEFT * 4.4 + UP * 2.3)
    right = pill("from the notebook (slide 7)", STATE, 18).move_to(RIGHT * 4.4 + UP * 2.3)
    scene.play(FadeIn(left), FadeIn(right),
               GrowArrow(Arrow(left.get_bottom(), final.get_left(), buff=0.15, color=ATTN, stroke_width=5)),
               GrowArrow(Arrow(right.get_bottom(), final.get_right(), buff=0.15, color=STATE, stroke_width=5)))
    return final


def animate_check(scene: Deck):
    """Both routes give the same numbers (computed by the tested reference code)."""
    rng = np.random.default_rng(11)
    a = np.array([0.9, 0.8, 0.5, 0.75])
    K, Q, V = rng.normal(size=(4, 2)), rng.normal(size=(4, 2)), rng.normal(size=(4, 1))
    fmt = lambda y: "   ".join(f"{v:+.2f}" for v in y)
    c1 = VGroup(T("recurrence", 22, STATE, bold=True), T(fmt(ssd_recurrent(a, K, Q, V)[:, 0]), 26, TOKEN)).arrange(DOWN, buff=0.15)
    c2 = VGroup(T("matrix form", 22, ATTN, bold=True), T(fmt(ssd_matrix(a, K, Q, V)[:, 0]), 26, TOKEN)).arrange(DOWN, buff=0.15)
    c1.move_to(LEFT * 3.7 + DOWN * 1.05)
    c2.move_to(RIGHT * 3.7 + DOWN * 1.05)
    scene.play(FadeIn(c1, shift=UP * 0.2), FadeIn(c2, shift=UP * 0.2), FadeIn(M("=", size=64, color=MAMBA2).move_to(DOWN * 1.2), scale=1.5))


def animate_family(scene: Deck):
    """One template, different fades: the linear-attention family."""
    fam = VGroup(
        VGroup(ceq("a_t", "=", "1", size=30), T("linear attention", 16, TOKEN)),
        VGroup(ceq("a_t", "=", r"\gamma", size=30), T("RetNet", 16, TOKEN)),
        VGroup(ceq("a_t", "=", r"a(x_t)", size=30), T("Mamba-2", 16, TOKEN)),
        VGroup(ceq(r"a_t", r"\to", r"\mathrm{diag}(", r"a_t", ")", size=30), T("Mamba-1", 16, TOKEN)),
    )
    for f in fam:
        f.arrange(DOWN, buff=0.1)
    fam.arrange(RIGHT, buff=0.9).move_to(DOWN * 2.75)
    scene.play(LaggedStart(*[FadeIn(f, shift=UP * 0.2) for f in fam], lag_ratio=0.2), run_time=1.3)


class S19Derivation(Deck):
    def construct(self):
        self.title("Deriving Mamba-2's attention form", MAMBA2)
        lines = animate_linear(self)
        self.pause()
        lines.append(animate_regroup(self))
        self.pause()
        lines.append(animate_rnn(self, lines[-1]))
        self.pause()
        lines.append(animate_fade(self, lines[-1]))
        self.pause()
        labels = [m for m in self.content() if m is not self._title and m not in lines]
        self.play(*[FadeOut(m) for m in labels], run_time=0.4)
        animate_meet(self, lines)
        self.pause()
        animate_check(self)
        animate_family(self)
        self.pause(1.4)
        self.clear_all()
