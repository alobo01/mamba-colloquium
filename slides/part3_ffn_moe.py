"""Part 3 — Workshops: why an FFN, MoE as a hospital, MoE in production, the residual bus (S09–S12)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *

ROOM = [ATTN, Q_COL, MAMBA2, V_COL, MAMBA, FIN, K_COL, MOE]


class S09FFN(Deck):
    def construct(self):
        self.title("Why a feed-forward network?", MOE)
        names = ["Leo", "hid", "the", "key", "under"]
        toks = VGroup(*[chip(n, TOKEN, 24) for n in names]).arrange(RIGHT, buff=0.7).move_to(UP * 1.5)
        self.play(LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in toks], lag_ratio=0.1))

        # attention: tokens talk to each other
        arcs = VGroup()
        for i, j in [(0, 3), (1, 3), (2, 3), (3, 4), (0, 4)]:
            a = ArcBetweenPoints(toks[i].get_top(), toks[j].get_top(), angle=-0.9, stroke_color=ATTN, stroke_width=4)
            arcs.add(a)
        lab_a = pill("attention: tokens talk to each other", ATTN, 22).move_to(DOWN * 0.1 + UP * 0.35)
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), FadeIn(lab_a), run_time=1.6)
        self.pause()

        # FFN: each token thinks alone (expand, squeeze)
        self.play(FadeOut(arcs), FadeOut(lab_a))
        hours = VGroup()
        for t in toks:
            top = t.get_bottom() + DOWN * 0.15
            poly = Polygon(top + LEFT * 0.2, top + RIGHT * 0.2, top + DOWN * 0.7 + RIGHT * 0.5, top + DOWN * 1.4 + RIGHT * 0.2,
                           top + DOWN * 1.4 + LEFT * 0.2, top + DOWN * 0.7 + LEFT * 0.5,
                           stroke_color=MOE, stroke_width=3, fill_color=MOE, fill_opacity=0.18)
            dots = VGroup(*[Dot(radius=0.05, color=MOE).move_to(top + DOWN * 0.7 + RIGHT * dx) for dx in (-0.3, -0.1, 0.1, 0.3)])
            hours.add(VGroup(poly, dots))
        lab_f = pill("FFN: each token thinks on its own", MOE, 22).move_to(DOWN * 2.3)
        self.play(LaggedStart(*[FadeIn(h, shift=DOWN * 0.2) for h in hours], lag_ratio=0.12), FadeIn(lab_f))
        self.play(LaggedStart(*[Flash(h[1][k], color=MOE, flash_radius=0.18, line_length=0.08) for h in hours for k in (1, 2)], lag_ratio=0.03), run_time=1.2)
        self.pause()

        # where the parameters live
        self.play(FadeOut(VGroup(toks, hours, lab_f)))
        total = 10.8
        a_w, f_w = total / 3, total * 2 / 3
        bar_a = Rectangle(width=a_w, height=1.0, stroke_width=0, fill_color=ATTN, fill_opacity=0.9)
        bar_f = Rectangle(width=f_w, height=1.0, stroke_width=0, fill_color=MOE, fill_opacity=0.9)
        bars = VGroup(bar_a, bar_f).arrange(RIGHT, buff=0.04).move_to(UP * 0.2)
        ta = M(r"\text{attention}\ 4d^2", size=34, color=BG1).move_to(bar_a)
        tf = M(r"\text{FFN}\ 8d^2", size=40, color=BG1).move_to(bar_f)
        self.play(GrowFromEdge(bar_a, LEFT), GrowFromEdge(bar_f, LEFT), run_time=1.1)
        self.play(FadeIn(ta), FadeIn(tf))
        br = Brace(bar_f, DOWN, color=MOE)
        two3 = M(r"\approx \tfrac{2}{3}\ \text{of each layer}", size=40, color=MOE).next_to(br, DOWN, buff=0.15)
        self.play(GrowFromCenter(br), Write(two3))
        mem_a = pill("memory of this context", ATTN, 20).move_to(LEFT * 3.6 + DOWN * 2.6)
        mem_f = pill("memory of the training data", MOE, 20).move_to(RIGHT * 3.0 + DOWN * 2.6)
        self.play(FadeIn(mem_a, shift=UP * 0.2), FadeIn(mem_f, shift=UP * 0.2))
        self.pause(1.4)
        self.clear_all()


class S10MoE(Deck):
    def construct(self):
        self.title("MoE: a hospital with specialist rooms", MOE)
        n = 8
        building = card(12.9, 2.9, MOE, 0.05, radius=0.3, sw=2).move_to(UP * 0.95)
        rooms = VGroup(*[card(1.25, 1.7, c, 0.14, radius=0.16, sw=2.5) for c in ROOM]).arrange(RIGHT, buff=0.24).move_to(building)
        for r, c, k in zip(rooms, ROOM, range(n)):
            sym = [Circle(radius=0.2), Triangle().scale(0.22), Square(0.36), Star(5, outer_radius=0.24, inner_radius=0.11),
                   RegularPolygon(6).scale(0.22), Circle(radius=0.2), Triangle().scale(0.22).rotate(PI), Square(0.36).rotate(PI / 4)][k]
            sym.set_stroke(c, 3).set_fill(c, 0.25).move_to(r)
            r.add(sym)
        self.play(FadeIn(building), LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in rooms], lag_ratio=0.08), run_time=1.5)
        for r in rooms:
            r.save_state()

        plus = VGroup(RoundedRectangle(width=0.16, height=0.52, corner_radius=0.04, stroke_width=0, fill_color=MOE, fill_opacity=1),
                      RoundedRectangle(width=0.52, height=0.16, corner_radius=0.04, stroke_width=0, fill_color=MOE, fill_opacity=1))
        nurse = VGroup(Circle(radius=0.5, stroke_color=MOE, stroke_width=4, fill_color=MOE, fill_opacity=0.2),
                       plus).move_to(LEFT * 3.2 + DOWN * 1.35)
        nurse_t = T("router", 22, MOE, bold=True).next_to(nurse, DOWN, buff=0.15)
        self.play(FadeIn(nurse), FadeIn(nurse_t))
        self.pause()

        picks = [([1, 4], "Leo"), ([2, 6], "key"), ([0, 4], "bridge")]
        for idx, (pk, word) in enumerate(picks):
            pt = chip(word, TOKEN, 22).move_to(LEFT * 6.0 + DOWN * 1.35)
            self.play(pt.animate.next_to(nurse, LEFT, buff=0.2), run_time=0.5)
            lines, anims = VGroup(), []
            for j, wgt in zip(pk, (0.6, 0.4)):
                ln = Line(nurse.get_top(), rooms[j].get_bottom(), color=ROOM[j], stroke_width=2 + 6 * wgt)
                lines.add(ln)
                anims.append(rooms[j].animate.set_fill(ROOM[j], 0.55))
            dim = [rooms[j].animate.fade(0.7) for j in range(n) if j not in pk]
            self.play(*[Create(l) for l in lines], *anims, *dim, run_time=0.8)
            if idx == 0:
                eq = M(r"y=\sum_{i\in\text{top-}k} g_i\,E_i(x)", size=36, color=MOE).move_to(RIGHT * 3.6 + DOWN * 1.45)
                self.play(Write(eq))
            self.play(FadeOut(lines), FadeOut(pt), *[Restore(r) for r in rooms], run_time=0.5)
        self.pause()

        b1 = pill("8 rooms built", MOE, 24).move_to(LEFT * 3.2 + DOWN * 3.05)
        b2 = pill("only 2 open per token", MAMBA2, 24, filled=True).move_to(RIGHT * 3.4 + DOWN * 3.05)
        self.play(FadeIn(b1, shift=UP * 0.2), FadeIn(b2, shift=UP * 0.2))
        self.pause(1.4)
        self.clear_all()


class S11MoEProduction(Deck):
    def construct(self):
        self.title("How modern LLMs run MoE", MOE)
        xs = [-4.75, 0, 4.75]
        panels = VGroup(*[card(4.4, 5.3, MOE, 0.06).move_to([x, -0.35, 0]) for x in xs])
        heads = VGroup(*[T(t, 24, MOE, bold=True).move_to([x, 1.85, 0]) for x, t in zip(xs, ["many small + 1 shared", "keep the load even", "experts across GPUs"])])
        self.play(FadeIn(panels, lag_ratio=0.2), FadeIn(heads, lag_ratio=0.2))

        # 1 shared + many small experts
        x = xs[0]
        shared = card(1.2, 1.0, V_COL, 0.5).move_to([x, 0.6, 0])
        shared_t = T("shared", 16, BG1, bold=True).move_to(shared)
        small = VGroup(*[RoundedRectangle(width=0.5, height=0.5, corner_radius=0.08, stroke_color=c, stroke_width=2, fill_color=c, fill_opacity=0.12)
                         for c in (ROOM * 2)]).arrange_in_grid(4, 4, buff=0.12).move_to([x, -1.0, 0])
        self.play(FadeIn(shared), FadeIn(shared_t), FadeIn(small, lag_ratio=0.03))
        hot = [1, 5, 6, 10, 12, 14, 3, 9]
        self.play(LaggedStart(*[small[i].animate.set_fill(ROOM[i % 8], 0.8) for i in hot], lag_ratio=0.08), run_time=1.3)
        ex = T("DeepSeek-V3: 256 + 1, top-8", 17, FADE).move_to([x, -2.55, 0])
        self.play(FadeIn(ex))
        self.pause()

        # 2 load balancing
        x = xs[1]
        skew = np.array([0.95, 0.3, 0.8, 0.15, 0.7, 0.2, 0.55, 0.25])
        even = np.full(8, 0.5)
        bars = VGroup(*[Rectangle(width=0.32, height=2.3 * v, stroke_width=0, fill_color=PROB_COL if v > 0.65 else ROOM[i], fill_opacity=0.9)
                        for i, v in enumerate(skew)]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN).move_to([x, -0.2, 0])
        base = Line(bars.get_corner(DL) + LEFT * 0.1, bars.get_corner(DR) + RIGHT * 0.1, color=FADE, stroke_width=2)
        self.play(FadeIn(base), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08), run_time=1.2)
        tgt = VGroup(*[Rectangle(width=0.32, height=2.3 * v, stroke_width=0, fill_color=ROOM[i], fill_opacity=0.9)
                       for i, v in enumerate(even)]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN).move_to(bars, aligned_edge=DOWN)
        nudge = T("nudge the router's bias", 17, FADE).move_to([x, -2.55, 0])
        self.play(Transform(bars, tgt), FadeIn(nudge), run_time=1.6)
        self.pause()

        # 3 experts across GPUs, all-to-all
        x = xs[2]
        gpus = VGroup(*[card(1.55, 1.05, MOE, 0.1, radius=0.12, sw=2) for _ in range(4)]).arrange_in_grid(2, 2, buff=0.9).move_to([x, 0.0, 0])
        for g, k in zip(gpus, range(4)):
            g.add(VGroup(*[Square(0.3, stroke_color=ROOM[2 * k + j], stroke_width=2, fill_color=ROOM[2 * k + j], fill_opacity=0.5) for j in (0, 1)])
                  .arrange(RIGHT, buff=0.12).move_to(g))
        mesh = VGroup()
        for i in range(4):
            for j in range(i + 1, 4):
                mesh.add(Line(gpus[i].get_center(), gpus[j].get_center(), color=MOE, stroke_width=1.5, stroke_opacity=0.55))
        self.play(FadeIn(gpus, lag_ratio=0.15), Create(mesh))
        dots = VGroup(*[Dot(radius=0.07, color=RESID).move_to(l.get_start()) for l in mesh])
        self.add(dots)
        self.play(*[d.animate.move_to(l.get_end()) for d, l in zip(dots, mesh)], run_time=1.2, rate_func=there_and_back_with_pause)
        a2a = T("tokens travel: all-to-all", 17, FADE).move_to([x, -2.55, 0])
        self.play(FadeIn(a2a))
        self.pause(1.4)
        self.clear_all()


PROB_COL = "#FF6B81"


class S12Bus(Deck):
    def construct(self):
        self.title("The residual stream: one shared bus", RESID, accent=MOE)
        belt_y = -0.5
        belt = Line(LEFT * 6.4 + UP * belt_y, RIGHT * 0.6 + UP * belt_y, color=RESID, stroke_width=8)
        belt_glow = glow(belt, RESID, 3)
        r_lab = T("r", 28, RESID, bold=True).next_to(belt, LEFT, buff=0.1)
        self.play(Create(belt), FadeIn(belt_glow), FadeIn(r_lab))
        specs = [("attention", ATTN, -5.3), ("MoE", MOE, -3.6), ("Mamba-2", MAMBA2, -1.9), ("attention", ATTN, -0.2)]
        stack = VGroup()
        stack_x = 3.0
        base_y = -2.4
        for k, (name, c, x) in enumerate(specs):
            m = card(1.5, 0.85, c, 0.2, radius=0.14).move_to([x, 1.2, 0])
            mt = T(name, 17, c, bold=True).move_to(m)
            rd = arrow([x - 0.35, belt_y + 0.1, 0], [x - 0.35, 0.72, 0], FADE, 3, 0.12)
            wr = arrow([x + 0.35, 0.72, 0], [x + 0.35, belt_y + 0.1, 0], c, 6, 0.16)
            seg = Rectangle(width=1.1, height=0.55, stroke_width=0, fill_color=c, fill_opacity=0.9)
            seg.move_to([stack_x, base_y + 0.275 + 0.58 * k, 0])
            self.play(FadeIn(m), FadeIn(mt), GrowArrow(rd), run_time=0.4)
            self.play(GrowArrow(wr), FadeIn(seg, shift=UP * 0.2), run_time=0.6)
            stack.add(seg)
        s_lab = T("sum of every write", 18, RESID).next_to(stack, UP, buff=0.2)
        self.play(FadeIn(s_lab))
        eq = M(r"r^{(\ell+1)}=r^{(\ell)}+\mathrm{block}_\ell\!\big(\mathrm{norm}\,r^{(\ell)}\big)", size=34).to_edge(DOWN, buff=0.7).shift(LEFT * 3.2)
        self.play(Write(eq))
        self.pause()

        # two axes: depth (residual) vs time (Mamba state)
        pan = card(4.6, 4.4, FADE, 0.05).move_to(RIGHT * 4.6 + UP * 0.1)
        # place panel after clearing the stack to avoid overlap
        self.play(FadeOut(VGroup(stack, s_lab)))
        pts = [[RIGHT * (3.1 + 1.0 * c) + UP * (-1.0 + 1.1 * r) for c in range(4)] for r in range(3)]
        dots = VGroup(*[Dot(p, radius=0.08, color=TOKEN) for row in pts for p in row])
        self.play(FadeIn(pan), FadeIn(dots))
        tarr = VGroup(*[arrow(pts[1][c] + RIGHT * 0.12, pts[1][c + 1] - RIGHT * 0.12, MAMBA2, 5, 0.14) for c in range(3)])
        darr = VGroup(*[arrow(pts[r][c] + UP * 0.12, pts[r + 1][c] - UP * 0.12, RESID, 4, 0.12) for c in (0, 3) for r in range(2)])
        t_lab = T("time: Mamba state", 18, MAMBA2, bold=True).next_to(pan, DOWN, buff=-0.45)
        d_lab = T("depth: residual", 18, RESID, bold=True).next_to(pan.get_top(), DOWN, buff=0.15)
        self.play(LaggedStart(*[GrowArrow(a) for a in tarr], lag_ratio=0.2), FadeIn(t_lab))
        self.play(LaggedStart(*[GrowArrow(a) for a in darr], lag_ratio=0.1), FadeIn(d_lab))
        self.pause(1.4)
        self.clear_all()
