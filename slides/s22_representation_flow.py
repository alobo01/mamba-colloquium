"""Slide 22 · Where the representations differ: token diversity across depth, and what attention looks like.

Hoang et al. 2025 (arXiv 2510.06640): Transformers homogenise token representations early and recover diversity
late; SSMs keep tokens distinct early and homogenise deep (training dynamics, not architecture, for SSMs).
Ali, Zimerman & Wolf 2024 "The Hidden Attention of Mamba Models": Mamba layers define implicit attention maps.
Curves are schematic shapes of the reported trend, not data.

Clicks:  ▶1 the two curves   ▶2 why   ▶3 hidden attention maps
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *
from math_contracts.test_ssd_equivalence import decay_mask


def tf_curve(x: float) -> float:
    """Transformer: fast drop, plateau, late recovery."""
    return 8.6 - 6.3 * (1 - np.exp(-x / 0.9)) + 4.6 / (1 + np.exp(-(x - 8.4) / 0.55))


def ssm_curve(x: float) -> float:
    """SSM: stays diverse, then smooths out deep in the stack."""
    return 9.0 - 6.4 / (1 + np.exp(-(x - 6.0) / 0.9))


def animate_curves(scene: Deck) -> Axes:
    ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=7.0, y_length=4.2, tips=False,
              axis_config={"color": FADE, "include_ticks": False, "stroke_width": 2}).move_to(LEFT * 2.4 + DOWN * 0.4)
    scene.play(Create(ax), FadeIn(T("depth →", 20, FADE).next_to(ax, DOWN, buff=0.15)),
               FadeIn(T("how different the tokens are", 20, FADE).rotate(PI / 2).next_to(ax, LEFT, buff=0.15)))
    tf = ax.plot(tf_curve, x_range=[0, 10], color=ATTN, stroke_width=6)
    ss = ax.plot(ssm_curve, x_range=[0, 10], color=STATE, stroke_width=6)
    scene.play(Create(tf), run_time=1.5)
    scene.play(FadeIn(T("Transformer: blur early, recover late", 18, ATTN, bold=True).next_to(ax.c2p(3.5, tf_curve(3.5)), DOWN, buff=0.25)))
    scene.play(Create(ss), run_time=1.5)
    scene.play(FadeIn(T("SSM: sharp early, blur deep", 18, STATE, bold=True).next_to(ax.c2p(2.0, ssm_curve(2.0)), UR, buff=0.15)))
    return ax


def animate_causes(scene: Deck):
    """Two short reasons on the right."""
    rows = VGroup(VGroup(pill("Transformer", ATTN, 18), T("comes from the architecture", 16, TOKEN)).arrange(DOWN, buff=0.12),
                  VGroup(pill("SSM", STATE, 18), T("comes from training", 16, TOKEN), T("(an untrained Mamba stays diverse)", 14, FADE))
                  .arrange(DOWN, buff=0.1)).arrange(DOWN, buff=0.5).move_to(RIGHT * 4.9 + DOWN * 0.3)
    scene.play(FadeIn(T("why?", 24, RESID, bold=True).next_to(rows, UP, buff=0.35)), LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows], lag_ratio=0.3))


def animate_hidden_attention(scene: Deck):
    """A sharp softmax map next to Mamba's smooth hidden-attention map, with its formula."""
    n = 6
    sm = heat(causal_softmax(n, 4, temp=2.4), 0.42, pos=ATTN).move_to(LEFT * 3.3 + DOWN * 0.15)
    rng = np.random.default_rng(2)
    a = rng.uniform(0.7, 0.95, n)
    hid = heat(decay_mask(a) * np.tril(np.abs(rng.normal(1, 0.25, (n, n)))), 0.42, pos=STATE).move_to(RIGHT * 1.0 + DOWN * 0.15)
    labs = VGroup(T("softmax attention", 20, ATTN, bold=True).next_to(sm, UP, buff=0.25),
                  T("Mamba, read as attention", 20, STATE, bold=True).next_to(hid, UP, buff=0.25))
    scene.play(FadeIn(sm, lag_ratio=0.02), FadeIn(hid, lag_ratio=0.02), FadeIn(labs), run_time=1.3)
    eq = ceq(r"\tilde\alpha_{ts}", "=", "c_t^\\top", r"\Big(", (r"\prod_{u=s+1}^{t}\bar A_u", DECAY), r"\Big)", (r"\bar b_s", K_COL), size=36)
    eq.move_to(DOWN * 2.45 + LEFT * 1.2)
    scene.play(Write(eq))
    note = VGroup(T("one map per channel", 18, TOKEN), T("≈ 1000× more maps than heads", 16, FADE)).arrange(DOWN, buff=0.1)
    scene.play(FadeIn(note.move_to(RIGHT * 5.0 + DOWN * 0.15), shift=LEFT * 0.2))


class S22RepresentationFlow(Deck):
    def construct(self):
        self.title("Where the representations differ", STATE)
        animate_curves(self)
        self.pause()
        animate_causes(self)
        self.pause()
        self.clear_body()
        animate_hidden_attention(self)
        self.pause(1.4)
        self.clear_all()
