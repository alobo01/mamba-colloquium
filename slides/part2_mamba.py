"""Part 2 — Notebooks: long context -> RNN -> Mamba -> Mamba-2, and the Mamba-2 maths (S05–S08)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *
from math_contracts.test_ssd_equivalence import decay_mask, ssd_recurrent, ssd_matrix

PROB = "#FF6B81"


class S05Ladder(Deck):
    def construct(self):
        # --- the wall keeps growing
        theo = brother("Theo", ATTN, "cards", 0.9).move_to(LEFT * 5.2 + DOWN * 1.8)
        self.play(FadeIn(theo))
        wall = VGroup()
        ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=5.2, y_length=3.6, tips=False,
                  axis_config={"color": FADE, "include_ticks": False, "stroke_width": 2}).move_to(RIGHT * 2.8 + DOWN * 0.1)
        xl = T("context length", 20, FADE).next_to(ax, DOWN, buff=0.15)
        quad = ax.plot(lambda x: min(10, 0.1 * x * x), x_range=[0, 10], color=ATTN, stroke_width=6)
        flat = ax.plot(lambda x: 1.2, x_range=[0, 10], color=MAMBA, stroke_width=6)
        self.play(Create(ax), FadeIn(xl))
        for k in range(10):
            c = token_card("", w=0.5, h=0.65).move_to(theo.get_top() + UP * 0.6 + RIGHT * (0.45 * (k % 5)) + UP * (0.7 * (k // 5)) + LEFT * 0.6)
            wall.add(c)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in wall], lag_ratio=0.1), Create(quad), run_time=2.2)
        qlab = T("attention", 22, ATTN, bold=True).next_to(quad.point_from_proportion(0.9), LEFT, buff=0.15)
        self.play(FadeIn(qlab))
        self.pause()
        fl = T("fixed-size state", 22, MAMBA, bold=True).next_to(flat.point_from_proportion(0.8), UP, buff=0.2)
        self.play(Create(flat), FadeIn(fl))
        self.pause()
        self.clear_all()

        # --- the ladder: RNN -> Mamba -> Mamba-2
        cols = [(-4.75, "RNN", FADE), (0, "Mamba", MAMBA), (4.75, "Mamba-2", MAMBA2)]
        frames = VGroup(*[card(4.3, 6.2, c, 0.07).move_to([x, -0.1, 0]) for x, _, c in cols])
        heads = VGroup(*[T(n, 34, c, bold=True).move_to([x, 2.55, 0]) for x, n, c in cols])
        arrows = VGroup(*[arrow([x - 0.3, 2.55, 0], [x + 0.3, 2.55, 0], RESID, 5, 0.16) for x in (-2.375, 2.375)])
        self.play(FadeIn(frames, lag_ratio=0.2), FadeIn(heads, lag_ratio=0.2), FadeIn(arrows), run_time=1.2)

        # RNN: a domino chain that must fall in order, and a fading gradient
        x0 = -4.75
        dom = VGroup(*[RoundedRectangle(width=0.24, height=0.8, corner_radius=0.05, stroke_width=0,
                                        fill_color=FADE, fill_opacity=0.9) for _ in range(6)]).arrange(RIGHT, buff=0.3).move_to([x0, 1.2, 0])
        self.play(FadeIn(dom))
        self.play(LaggedStart(*[Rotate(d, -0.9, about_point=d.get_bottom() + RIGHT * 0.14) for d in dom], lag_ratio=0.25), run_time=1.5)
        pulse = VGroup(*[Arrow(RIGHT * 0.28, LEFT * 0.28, buff=0, stroke_width=6, tip_length=0.14, color=PROB).set_opacity(1 - 0.17 * i)
                         for i in range(6)]).arrange(LEFT, buff=0.1).move_to([x0, 0.15, 0])
        self.play(LaggedStart(*[GrowArrow(a) for a in pulse], lag_ratio=0.2), run_time=1.2)
        p1 = pill("one token at a time", PROB, 19).move_to([x0, -1.0, 0])
        p2 = pill("gradients fade away", PROB, 19).move_to([x0, -1.8, 0])
        self.play(FadeIn(p1, shift=UP * 0.15), FadeIn(p2, shift=UP * 0.15))
        self.pause()

        # Mamba: input decides what to keep + a parallel scan
        x1 = 0
        f1 = pill("input decides what to keep", MAMBA2, 19).move_to([x1, -0.4, 0])
        f2 = pill("parallel scan", MAMBA2, 19).move_to([x1, -1.1, 0])
        dial = VGroup(Circle(radius=0.32, stroke_color=MAMBA, stroke_width=4), Line(ORIGIN, UP * 0.26, color=MAMBA, stroke_width=4))
        dial.move_to([x1 - 1.0, 1.2, 0])
        delta = M(r"\Delta_t", size=34, color=MAMBA).next_to(dial, DOWN, buff=0.1)
        leaves = VGroup(*[Dot(radius=0.07, color=MAMBA) for _ in range(4)]).arrange(RIGHT, buff=0.35).move_to([x1 + 0.9, 0.8, 0])
        mid = VGroup(*[Dot(radius=0.07, color=MAMBA) for _ in range(2)]).arrange(RIGHT, buff=0.7).move_to([x1 + 0.9, 1.3, 0])
        top = Dot(radius=0.07, color=MAMBA).move_to([x1 + 0.9, 1.8, 0])
        tree = VGroup(*[Line(l.get_center(), m.get_center(), color=MAMBA, stroke_width=2.5)
                        for l, m in zip(leaves, [mid[0], mid[0], mid[1], mid[1]])],
                      *[Line(m.get_center(), top.get_center(), color=MAMBA, stroke_width=2.5) for m in mid])
        self.play(FadeIn(f1, shift=UP * 0.15), FadeIn(dial), FadeIn(delta))
        self.play(Rotate(dial[1], 1.2, about_point=dial[0].get_center()), FadeIn(f2, shift=UP * 0.15),
                  FadeIn(VGroup(tree, leaves, mid, top)), run_time=1.3)
        q1 = pill("scan can't use matmul chips", PROB, 19).move_to([x1, -1.9, 0])
        self.play(FadeIn(q1, shift=UP * 0.15))
        self.pause()

        # Mamba-2: one decay per head -> matmuls
        x2 = 4.75
        g1 = pill("one decay per head", MAMBA2, 19).move_to([x2, -0.4, 0])
        g2 = pill("everything becomes matmul", MAMBA2, 19).move_to([x2, -1.1, 0])
        tiles = VGroup(*[Square(0.42, stroke_color=MAMBA2, stroke_width=2, fill_color=MAMBA2, fill_opacity=0.15) for _ in range(12)])
        tiles.arrange_in_grid(3, 4, buff=0.08).move_to([x2, 1.3, 0])
        self.play(FadeIn(g1, shift=UP * 0.15), FadeIn(tiles))
        self.play(LaggedStart(*[t.animate.set_fill(opacity=0.85) for t in tiles], lag_ratio=0.06), FadeIn(g2, shift=UP * 0.15), run_time=1.4)
        res = pill("bigger state · 2–8× faster", MAMBA2, 19, filled=True).move_to([x2, -1.9, 0])
        self.play(FadeIn(res, scale=0.9), FadeIn(glow(res[0], MAMBA2, 3)))
        self.pause(1.4)
        self.clear_all()


def sheet(N, P, cell=0.55, color=MAMBA2, opacity=0.35):
    g = VGroup(*[Square(cell, stroke_color=color, stroke_width=1.5, fill_color=color, fill_opacity=opacity)
                 .move_to([p * cell, -n * cell, 0]) for n in range(N) for p in range(P)])
    return g.move_to(ORIGIN)


class S06SSDState(Deck):
    def construct(self):
        self.title("Mamba-2 in one picture", MAMBA2)
        N, P, cell = 4, 6, 0.6
        rng = np.random.default_rng(3)
        g = sheet(N, P, cell)
        for sq in g:
            sq.set_fill(opacity=float(rng.uniform(0.15, 0.7)))
        g.move_to(LEFT * 1.2 + UP * 0.2)
        halo = glow(SurroundingRectangle(g, buff=0.05, corner_radius=0.1), MAMBA2, 3)
        bN = Brace(g, LEFT, color=FADE); bP = Brace(g, UP, color=FADE)
        lN = M("N", size=30, color=FADE).next_to(bN, LEFT, buff=0.1); lP = M("P", size=30, color=FADE).next_to(bP, UP, buff=0.1)
        nb = T("the notebook: a fixed N × P sheet", 22, MAMBA2, bold=True).next_to(g, DOWN, buff=0.35)
        self.play(FadeIn(g), FadeIn(halo), GrowFromCenter(bN), GrowFromCenter(bP), FadeIn(lN), FadeIn(lP), FadeIn(nb))
        eq = M(r"S_t", "=", r"a_t", r"S_{t-1}", "+", r"k_t", r"v_t^\top", r",\quad", r"y_t", "=", r"S_t^\top", r"q_t", size=44).to_edge(DOWN, buff=0.75)
        eq[2].set_color(FADE); eq[5].set_color(K_COL); eq[6].set_color(V_COL); eq[11].set_color(Q_COL)
        self.play(Write(eq))
        steps = VGroup(pill("1  fade", FADE, 22), pill("2  write", V_COL, 22), pill("3  read", Q_COL, 22)).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        steps.move_to(RIGHT * 5.0 + UP * 0.3)
        self.play(FadeIn(steps, lag_ratio=0.2))
        self.pause()

        # 1 fade
        self.play(Indicate(eq[2:4], color=FADE), steps[0].animate.scale(1.12),
                  *[sq.animate.set_fill(opacity=sq.get_fill_opacity() * 0.55) for sq in g], run_time=1.3)
        self.play(steps[0].animate.scale(1 / 1.12))
        self.pause(0.4)
        # 2 write: k (column) x v (row) -> rank-1 layer
        kcol = VGroup(*[Square(cell, stroke_color=K_COL, stroke_width=2, fill_color=K_COL, fill_opacity=float(o))
                        for o in (0.9, 0.3, 0.7, 0.5)]).arrange(DOWN, buff=0).next_to(g, LEFT, buff=0.35)
        vrow = VGroup(*[Square(cell, stroke_color=V_COL, stroke_width=2, fill_color=V_COL, fill_opacity=float(o))
                        for o in (0.4, 0.9, 0.3, 0.8, 0.6, 0.5)]).arrange(RIGHT, buff=0).next_to(g, UP, buff=0.62)
        self.play(FadeOut(VGroup(bN, bP, lN, lP)), FadeIn(kcol, shift=RIGHT * 0.3), FadeIn(vrow, shift=DOWN * 0.3), Indicate(eq[5:7], color=V_COL),
                  steps[1].animate.scale(1.12))
        ko = [0.9, 0.3, 0.7, 0.5]; vo = [0.4, 0.9, 0.3, 0.8, 0.6, 0.5]
        layer = VGroup(*[Square(cell, stroke_width=0, fill_color=V_COL, fill_opacity=0.75 * ko[n] * vo[p]).move_to(g[n * P + p])
                         for n in range(N) for p in range(P)])
        self.play(FadeIn(layer), run_time=1.2)
        self.play(*[sq.animate.set_fill(opacity=min(0.95, sq.get_fill_opacity() + l.get_fill_opacity())) for sq, l in zip(g, layer)],
                  FadeOut(layer), steps[1].animate.scale(1 / 1.12), run_time=0.9)
        self.pause(0.4)
        # 3 read
        qcol = VGroup(*[Square(cell, stroke_color=Q_COL, stroke_width=2, fill_color=Q_COL, fill_opacity=float(o))
                        for o in (0.8, 0.4, 0.9, 0.3)]).arrange(DOWN, buff=0).move_to(kcol)
        self.play(Transform(kcol, qcol), steps[2].animate.scale(1.12), Indicate(eq[10:12], color=Q_COL))
        yrow = VGroup(*[Square(cell, stroke_color=Q_COL, stroke_width=2, fill_color=Q_COL, fill_opacity=float(o))
                        for o in (0.5, 0.8, 0.3, 0.6, 0.9, 0.4)]).arrange(RIGHT, buff=0).next_to(g, DOWN, buff=0.35).align_to(g, LEFT)
        beams = VGroup(*[Line(qcol.get_right() + DOWN * 0.0, g[n * P].get_left(), stroke_width=0) for n in range(N)])
        self.play(FadeOut(nb), Flash(g, color=Q_COL, flash_radius=2.6, line_length=0.3), FadeIn(yrow, shift=UP * 0.3),
                  FadeIn(T("y", 26, Q_COL, bold=True).next_to(yrow, LEFT, buff=0.2)), run_time=1.4)
        self.pause(1.4)
        self.clear_all()


class S07SSDMatrix(Deck):
    def construct(self):
        self.title("Unroll it: attention with a decay mask", MAMBA2)
        a = np.array([0.9, 0.8, 0.5, 0.75])
        L = decay_mask(a)

        # 1. the past, faded step by step
        names = ["Leo", "hid", "the key", "bridge"]
        toks = VGroup(*[chip(n, TOKEN, 22) for n in names]).arrange(RIGHT, buff=0.7).move_to(UP * 1.4)
        weights = [L[3, s] for s in range(4)]
        self.play(LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in toks], lag_ratio=0.15))
        wlabs = VGroup()
        for t, w in zip(toks, weights):
            veil = t.copy().set_fill(BG1, 0.0)
            wl = T(f"{w:.3g}".lstrip("0") if w < 1 else "1", 24, MAMBA2, bold=True).next_to(t, DOWN, buff=0.3)
            wlabs.add(wl)
            self.play(t.animate.set_opacity(0.25 + 0.75 * w), FadeIn(wl, shift=UP * 0.1), run_time=0.55)
        now = T("now", 20, FADE).next_to(toks[3], UP, buff=0.2)
        self.play(FadeIn(now))
        hint = M(r".3=.8\cdot.5\cdot.75", size=30, color=MAMBA2).next_to(wlabs, DOWN, buff=0.35)
        self.play(Write(hint))
        self.pause()

        # 2. all rows at once: the decay matrix L
        self.play(FadeOut(VGroup(toks, wlabs, now, hint)))
        Lh = heat(L, 0.72, pos=MAMBA2, numbers=True, nsize=20).move_to(LEFT * 3.7 + DOWN * 0.1)
        Ltxt = M("L", size=44, color=MAMBA2).next_to(Lh, UP, buff=0.3)
        self.play(FadeIn(Lh, lag_ratio=0.03), FadeIn(Ltxt), run_time=1.3)

        rng = np.random.default_rng(7)
        QK = np.tril(rng.uniform(0.2, 1, (4, 4)))
        QKh = heat(QK, 0.72, pos=ATTN).move_to(RIGHT * 0.3 + DOWN * 0.1)
        QKt = M(r"QK^\top", size=44, color=ATTN).next_to(QKh, UP, buff=0.3)
        circ = M(r"\odot", size=60).move_to((Lh.get_right() + QKh.get_left()) / 2)
        self.play(FadeIn(circ), FadeIn(QKh), FadeIn(QKt))
        res = heat(L * QK, 0.72, pos=V_COL).move_to(RIGHT * 4.9 + DOWN * 0.1)
        eqs = M("=", size=60).move_to((QKh.get_right() + res.get_left()) / 2)
        self.play(FadeIn(eqs), FadeIn(res))
        self.pause()

        # 3. same recipe as softmax attention, different mask
        eq = M(r"Y=\big(L\odot QK^\top\big)V", size=54).to_edge(DOWN, buff=0.7)
        self.play(Write(eq))
        sm = heat(causal_softmax(4, 2), 0.5, pos=ATTN).scale(0.9)
        sm_t = T("softmax attention", 20, ATTN, bold=True)
        ss = VGroup(sm_t, sm).arrange(DOWN, buff=0.15).scale(0.7).move_to(RIGHT * 5.6 + UP * 2.6)
        self.play(FadeIn(ss, shift=LEFT * 0.3))
        self.pause(1.4)
        self.clear_all()


class S08SSDChunks(Deck):
    def construct(self):
        self.title("Fast to train: chunks and low rank", MAMBA2)
        n, cell = 8, 0.4
        cells = VGroup()
        for i in range(n):
            for j in range(i + 1):
                same = (i // 2 == j // 2)
                cells.add(Square(cell, stroke_width=1, stroke_color=BG1, fill_color=MAMBA2 if same else ATTN,
                                 fill_opacity=0.85 if same else 0.35).move_to([j * cell, -i * cell, 0]))
        cells.move_to(LEFT * 4.3 + DOWN * 0.3)
        self.play(FadeIn(cells, lag_ratio=0.01), run_time=1.2)
        dense = pill("inside a chunk: dense matmul", MAMBA2, 18).next_to(cells, UP, buff=0.35).shift(RIGHT * 0.2)
        self.play(FadeIn(dense))
        self.pause(0.4)

        # an off-diagonal block is low rank: C x C = (C x N)(N x C)
        blk = VGroup(*[s for s in cells if abs(s.get_center()[0] - cells[0].get_center()[0] - 0 * cell) < 1e9][:0])
        big = Square(1.2, stroke_color=ATTN, stroke_width=3, fill_color=ATTN, fill_opacity=0.35).move_to(RIGHT * 0.9 + UP * 1.7)
        tall = Rectangle(width=0.35, height=1.2, stroke_color=K_COL, stroke_width=3, fill_color=K_COL, fill_opacity=0.5)
        wide = Rectangle(width=1.2, height=0.35, stroke_color=Q_COL, stroke_width=3, fill_color=Q_COL, fill_opacity=0.5)
        eqs = M("=", size=44)
        fac = VGroup(tall, M(r"\times", size=36), wide).arrange(RIGHT, buff=0.25)
        rowg = VGroup(big.copy(), eqs, fac).arrange(RIGHT, buff=0.4).move_to(RIGHT * 2.8 + UP * 2.0)
        r_lab = T("far-away block: rank ≤ N", 20, ATTN, bold=True).next_to(rowg, DOWN, buff=0.2)
        self.play(FadeIn(rowg, shift=LEFT * 0.3), FadeIn(r_lab))
        self.pause()

        steps = [("1", "inside each chunk", MAMBA2), ("2", "summarise chunk → state", V_COL),
                 ("3", "pass states along", MAMBA), ("4", "add the past back in", Q_COL)]
        rows = VGroup()
        for k, t, c in steps:
            dot = VGroup(Circle(radius=0.22, stroke_width=0, fill_color=c, fill_opacity=1), T(k, 22, BG1, bold=True))
            dot[1].move_to(dot[0])
            rows.add(VGroup(dot, T(t, 22, RESID)).arrange(RIGHT, buff=0.25))
        rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to(RIGHT * 3.2 + DOWN * 0.95)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.3), run_time=1.8)
        cost = M(r"\text{train: } O\!\big(T\,(NP+C(N{+}P))\big)\quad\text{decode: } O(NP)\ \text{per token}", size=34).to_edge(DOWN, buff=0.7)
        self.play(Write(cost))
        self.pause(1.4)
        self.clear_all()
