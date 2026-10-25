"""Slide 02 · Attention in one picture: query, key, value, softmax, heads.

Clicks:  ▶1 three projections   ▶2 scores q·k   ▶3 softmax   ▶4 blend values   ▶5 many heads + formula
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from visual import *

WORDS = ["Leo", "hid", "the key", "under", "the bridge", "..."]
SCORES = np.array([1.5, 0.5, 2.6, 0.2, 2.0, 0.3])
HEADS = [
    ("head 1", "who did what", ATTN, [[1, 0, 0, 0], [.8, .2, 0, 0], [.1, .8, .1, 0], [0, .1, .2, .7]]),
    ("head 2", "nearby words", MAMBA2, [[1, 0, 0, 0], [.1, .9, 0, 0], [0, .1, .9, 0], [0, 0, .1, .9]]),
    ("head 3", "far back", MOE, [[1, 0, 0, 0], [1, 0, 0, 0], [.8, .1, .1, 0], [.9, 0, .05, .05]]),
]


def animate_projections(scene: Deck):
    """One token passes through three lenses and becomes query, key and value."""
    tok = chip("bank", TOKEN, 28).move_to(LEFT * 5.3 + UP * 0.6)
    lens = VGroup(*[Ellipse(width=0.4, height=1.0, stroke_color=c, stroke_width=3, fill_color=c, fill_opacity=0.25)
                    for c in (Q_COL, K_COL, V_COL)]).arrange(DOWN, buff=0.25).next_to(tok, RIGHT, buff=1.0)
    names = [("query", "what am I looking for?", Q_COL), ("key", "how can I be found?", K_COL),
             ("value", "what do I hand over?", V_COL)]
    outs = VGroup()
    for (n, d, c), e in zip(names, lens):
        o = VGroup(T(n, 26, c, bold=True), T(d, 18, FADE)).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        outs.add(o.next_to(e, RIGHT, buff=0.5))
    wires = VGroup(*[Line(tok.get_right(), e.get_left(), color=e.get_color(), stroke_width=2.5) for e in lens])
    scene.play(FadeIn(tok), FadeIn(lens), Create(wires))
    scene.play(LaggedStart(*[FadeIn(o, shift=RIGHT * 0.3) for o in outs], lag_ratio=0.3), run_time=1.5)
    scene.pause()
    scene.play(FadeOut(VGroup(tok, lens, wires, outs)))


def animate_scores(scene: Deck) -> tuple:
    """The query beam sweeps the key strips; a bar grows per card. Returns (cards, bars, query, label)."""
    cards = VGroup(*[token_card(w, w=1.7, h=1.5, size=18) for w in WORDS]).arrange(RIGHT, buff=0.25).shift(DOWN * 1.6)
    query = chip("query  q", Q_COL, 26).move_to(UP * 1.5 + LEFT * 4)
    scene.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.1), FadeIn(query))
    hs = VGroup(*[Rectangle(width=0.55, height=0.3 * s, stroke_width=0, fill_color=K_COL, fill_opacity=0.9)
                  .next_to(c, UP, buff=0.12) for c, s in zip(cards, SCORES)])
    beam = Line(query.get_bottom(), cards[0].get_top(), color=Q_COL, stroke_width=4)
    scene.add(beam)
    for i in range(len(cards)):
        scene.play(beam.animate.put_start_and_end_on(query.get_bottom(), cards[i].get_top()),
                   GrowFromEdge(hs[i], DOWN), run_time=0.35)
    scene.play(FadeOut(beam))
    lab = ceq("q", r"\cdot", "k_s", size=40).next_to(query, RIGHT, buff=0.8)
    scene.play(Write(lab))
    return cards, hs, query, lab


def animate_softmax(scene: Deck, cards: VGroup, hs: VGroup, lab) -> tuple:
    """Scores become weights that sum to one; the label becomes softmax(q·k). Returns (weights, percent labels)."""
    w = np.exp(SCORES) / np.exp(SCORES).sum()
    nb = VGroup(*[Rectangle(width=0.55, height=max(0.04, 2.4 * wi), stroke_width=0, fill_color=K_COL, fill_opacity=0.9)
                  .next_to(c, UP, buff=0.12) for c, wi in zip(cards, w)])
    pct = VGroup(*[T(f"{100 * wi:.0f}%", 16, RESID, bold=True).next_to(b, UP, buff=0.05) for b, wi in zip(nb, w)])
    sm = ceq(r"\mathrm{softmax}", r"(", "q", r"\cdot", "k_s", ")", size=40).move_to(lab, aligned_edge=LEFT)
    scene.play(Transform(hs, nb), FadeIn(pct), Transform(lab, sm), run_time=1.3)
    return w, pct


def animate_blend(scene: Deck, cards: VGroup, w) -> VGroup:
    """Value bodies flow into one output, weighted by attention."""
    out = card(2.0, 1.0, V_COL, 0.4).move_to(RIGHT * 4 + UP * 1.5)
    out_t = T("output", 22, V_COL, bold=True).move_to(out)
    fl = [c[0].copy().set_fill(V_COL, opacity=0.3 + 0.6 * wi).set_stroke(V_COL, opacity=0.6) for c, wi in zip(cards, w)]
    scene.play(LaggedStart(*[f.animate.scale(0.3).move_to(out) for f in fl], lag_ratio=0.08), run_time=1.5)
    scene.play(FadeIn(out), FadeIn(out_t), *[FadeOut(f) for f in fl])
    return VGroup(out, out_t)


def animate_heads(scene: Deck):
    """Three heads with different patterns merge through W_O; the full formula appears."""
    heads = VGroup()
    for n, d, c, m in HEADS:
        h = heat(np.array(m), 0.42, pos=c)
        fr = card(2.4, 2.9, c, 0.1)
        h.move_to(fr).shift(UP * 0.15)
        heads.add(VGroup(fr, h, T(n, 22, c, bold=True).move_to(fr.get_top() + DOWN * 0.3),
                         T(d, 16, FADE).move_to(fr.get_bottom() + UP * 0.3)))
    heads.arrange(RIGHT, buff=0.5).shift(UP * 0.1)
    scene.play(LaggedStart(*[FadeIn(h, shift=UP * 0.3) for h in heads], lag_ratio=0.25), run_time=1.6)
    merge = card(3.4, 0.7, RESID, 0.1).shift(DOWN * 2.6)
    merge_t = M(r"\mathrm{concat}\ \to\ W_O", size=30).move_to(merge)
    scene.play(Create(merge), Write(merge_t),
               *[Create(Line(h.get_bottom(), merge.get_top(), color=h[0].get_color(), stroke_width=3)) for h in heads])
    eq = ceq("O", "=", r"\mathrm{softmax}\!\big(", "Q", "K^\\top", r"/\sqrt{d}\,\big)", "V", size=40)
    eq.to_edge(UP, buff=0.55).to_edge(RIGHT, buff=0.7)
    scene.play(Write(eq))


class S02QKV(Deck):
    def construct(self):
        self.title("Attention in one picture", ATTN)
        animate_projections(self)
        cards, hs, query, lab = animate_scores(self)
        self.pause()
        w, pct = animate_softmax(self, cards, hs, lab)
        self.pause()
        out = animate_blend(self, cards, w)
        self.pause()
        self.play(FadeOut(VGroup(cards, hs, pct, lab, query, out)))
        animate_heads(self)
        self.pause(1.4)
        self.clear_all()
