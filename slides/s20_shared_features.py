"""Slide 20 · Same features, stretched depth (Wang et al., ICLR 2025, "Towards Universality").

SAEs on Pythia-160M (12 layers) and Mamba-130M (24 layers), both trained on the Pile. Mean max-pairwise Pearson
correlation (MPPC): SAE-seed skyline 0.806, model-seed skyline 0.725, cross-architecture 0.681. A Pythia feature
in layer l is usually matched near Mamba layer 2l.

Clicks:  ▶1 two stacks, matched depths   ▶2 the similarity score + numbers   ▶3 one matched pair
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

TOP, BOTTOM = 2.1, -3.1
PYTHIA_X, MAMBA_X = -5.3, -2.4
LINKS = [1, 3, 6, 9, 11]                      # Pythia layer l  ->  Mamba layer 2l (1-based)
MPPC = [("same model\nnew SAE", 0.806, FADE), ("same arch\nnew seed", 0.725, ATTN), ("Pythia\n↔ Mamba", 0.681, K_COL)]


def build_stack(n: int, x: float, color) -> VGroup:
    """n layer bars filling the same height: deeper models get thinner layers."""
    gap = 0.05
    h = (TOP - BOTTOM - gap * (n - 1)) / n
    return VGroup(*[RoundedRectangle(width=1.1, height=h, corner_radius=min(0.06, h / 2), stroke_width=0, fill_color=color,
                                     fill_opacity=0.55).move_to([x, TOP - h / 2 - k * (h + gap), 0]) for k in range(n)])


def animate_stacks(scene: Deck) -> tuple:
    """Pythia (12) and Mamba (24) side by side; matched features link layer l to layer 2l."""
    py, mb = build_stack(12, PYTHIA_X, ATTN), build_stack(24, MAMBA_X, STATE)
    labs = VGroup(T("Pythia · 12", 18, ATTN, bold=True).next_to(py, UP, buff=0.15),
                  T("Mamba · 24", 18, STATE, bold=True).next_to(mb, UP, buff=0.15))
    scene.play(FadeIn(py, lag_ratio=0.05), FadeIn(mb, lag_ratio=0.03), FadeIn(labs), run_time=1.3)
    links = VGroup(*[Line(py[l - 1].get_right(), mb[2 * l - 1].get_left(), color=K_COL, stroke_width=3) for l in LINKS])
    scene.play(LaggedStart(*[Create(l) for l in links], lag_ratio=0.2),
               *[py[l - 1].animate.set_fill(opacity=0.95) for l in LINKS], *[mb[2 * l - 1].animate.set_fill(opacity=0.95) for l in LINKS],
               run_time=1.6)
    rule = ceq(r"\ell", r"\ \mapsto\ ", r"2\ell", size=34, color=K_COL).move_to([(PYTHIA_X + MAMBA_X) / 2, BOTTOM - 0.35, 0])
    scene.play(FadeIn(rule))
    return py, mb


def animate_scores(scene: Deck):
    """MPPC formula and the three averages as bars."""
    eq = ceq(r"\rho_i", "=", r"\max_j\ \mathrm{corr}\big(", (r"f^{\,\mathrm{Pythia}}_i", ATTN), ",", (r"f^{\,\mathrm{Mamba}}_j", STATE),
             r"\big)", size=32).move_to(RIGHT * 3.0 + UP * 1.95)
    scene.play(Write(eq))
    base_y, hscale = -1.5, 3.0
    xs = [1.0, 3.0, 5.0]
    grp = VGroup()
    for (name, v, c), x in zip(MPPC, xs):
        b = Rectangle(width=1.1, height=v * hscale, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to([x, base_y + v * hscale / 2, 0])
        grp.add(VGroup(b, T(f"{v:.3f}", 20, c, bold=True).next_to(b, UP, buff=0.1), T(name, 15, c, bold=True).move_to([x, base_y - 0.4, 0])))
    scene.play(LaggedStart(*[GrowFromEdge(g[0], DOWN) for g in grp], lag_ratio=0.2),
               LaggedStart(*[FadeIn(VGroup(g[1], g[2])) for g in grp], lag_ratio=0.2), run_time=1.5)
    notes = VGroup(T("over 25% of Pythia features have a Mamba twin with ρ > 0.95", 16, TOKEN),
                   T("raw neurons: under 15% reach ρ > 0.4", 16, FADE)).arrange(DOWN, buff=0.12).move_to(RIGHT * 3.0 + DOWN * 2.75)
    scene.play(FadeIn(notes, shift=UP * 0.15))


def animate_example(scene: Deck, py: VGroup, mb: VGroup):
    """One matched pair: the phrase 'more or less' — Pythia layer 6, Mamba layer 12, ρ = 0.83."""
    txt = VGroup(T("“more or less”", 15, K_COL, bold=True), T("ρ = 0.83", 15, TOKEN)).arrange(DOWN, buff=0.06)
    box = card(txt.width + 0.25, txt.height + 0.22, K_COL, 0.85, radius=0.1).set_fill(BG1, 0.92)
    ex = VGroup(box, txt).move_to([(PYTHIA_X + MAMBA_X) / 2, (py[5].get_y() + mb[11].get_y()) / 2 + 0.6, 0])
    scene.play(Indicate(py[5], color=K_COL), Indicate(mb[11], color=K_COL), FadeIn(ex, scale=0.9))


class S20SharedFeatures(Deck):
    def construct(self):
        self.title("Same features, stretched depth", K_COL)
        py, mb = animate_stacks(self)
        self.pause()
        animate_scores(self)
        self.pause()
        animate_example(self, py, mb)
        self.pause(1.4)
        self.clear_all()
