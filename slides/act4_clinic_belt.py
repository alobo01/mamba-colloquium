"""Act IV — The clinic, the conveyor belt, and real Transformers (slides 17–21)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *

ACT = "Act IV · The clinic, the belt, and real Transformers"
EXPERT_GLYPHS = [Circle, Square, Triangle, RegularPolygon, Star, Circle, Square, Triangle]


def glyph(i, color, size=0.28):
    cls = EXPERT_GLYPHS[i % len(EXPERT_GLYPHS)]
    if cls is RegularPolygon:
        g = RegularPolygon(6)
    elif cls is Star:
        g = Star(5, outer_radius=1)
    else:
        g = cls()
    g.set_stroke(color, 2).set_fill(color, 0.25)
    return g.scale_to_fit_height(size)


def neuron_layer(n, color, radius=0.11, buff=0.08):
    return VGroup(*[Circle(radius=radius, stroke_color=color, stroke_width=1.5) for _ in range(n)]).arrange(DOWN, buff=buff)


class S17WhyFFN(Deck):
    def construct(self):
        self.header("17", "Why FFNs, and why are they the expensive part?", ACT, "PROPERTY")
        # left: mixing between positions
        rails = VGroup(*[Line(DOWN * 1.2, UP * 1.2, color=RESID, stroke_width=2) for _ in range(4)]).arrange(RIGHT, buff=0.7).shift(LEFT * 4.2 + DOWN * 0.2)
        mix_lab = T("mixing: between tokens\n(attention / SSM)", 20, ATTN).next_to(rails, UP, buff=0.25)
        self.play(Create(rails), FadeIn(mix_lab))
        arcs = VGroup(*[CurvedArrow(rails[i].get_center(), rails[j].get_center() + UP * 0.3, angle=-PI / 3, color=V_COL, stroke_width=3)
                        for i, j in ((0, 2), (1, 3), (0, 3))])
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.3))
        lin = T("output = weighted sum of values:\nlinear in V", 18, FADE).next_to(rails, DOWN, buff=0.3)
        self.play(FadeIn(lin))
        # right: per-token refinery d -> 4d -> d
        l_in = neuron_layer(4, RESID); l_mid = neuron_layer(12, MOE, radius=0.08, buff=0.05); l_out = neuron_layer(4, RESID)
        net = VGroup(l_in, l_mid, l_out).arrange(RIGHT, buff=1.2).shift(RIGHT * 2.6 + DOWN * 0.2)
        edges = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=0.6, color=FADE) for a in l_in for b in l_mid],
                       *[Line(a.get_center(), b.get_center(), stroke_width=0.6, color=FADE) for a in l_mid for b in l_out])
        ref_lab = T("thinking: each token alone (FFN)", 20, MOE).next_to(net, UP, buff=0.25)
        dims = VGroup(M("d", size=28).next_to(l_in, DOWN), M(r"d_{ff}\approx 4d", size=28).next_to(l_mid, DOWN), M("d", size=28).next_to(l_out, DOWN))
        self.play(FadeIn(edges), FadeIn(net), FadeIn(ref_lab), FadeIn(dims))
        acts = np.array([0.9, 0, 0.1, 0.8, 0, 0, 0.7, 0, 0.05, 0, 0.6, 0])
        self.play(*[n.animate.set_fill(MOE, a) for n, a in zip(l_mid, acts)], run_time=1)
        self.pause()

        # FFN as key-value memory over the training set
        kv = M(r"\mathrm{FFN}(r)=\sum_{i=1}^{d_{ff}}\underbrace{\sigma(\mathbf k_i^\top r)}_{\text{match to key } i}\ \mathbf v_i", size=36).to_edge(DOWN, buff=1.0)
        rhyme = T("attention = key-value memory over the CONTEXT · FFN = key-value memory over the TRAINING DATA (Geva et al. 2021)", 17, MOE).next_to(kv, DOWN, buff=0.2)
        rhyme.scale_to_fit_width(min(rhyme.width, 12.8)).set_x(0)
        self.play(FadeOut(lin), Write(kv))
        self.play(FadeIn(rhyme))
        self.pause()
        self.clear_all(0.5)

        self.header("17", "The bill: FFN ≈ ⅔ of the layer at short context", ACT, "PROPERTY")
        d = 4096
        Ttr = ValueTracker(512)
        ax = Axes(x_range=[0, 3, 1], y_range=[0, 1, 0.25], x_length=6, y_length=3.6, tips=False,
                  axis_config={"stroke_color": FADE}, y_axis_config={"include_numbers": False}).shift(LEFT * 3.4 + DOWN * 0.5)
        names = [("FFN  16d²", MOE), ("attn proj  8d²", ATTN), ("scores+mix  4Td", Q_COL)]

        def vals():
            T_ = Ttr.get_value()
            v = np.array([16 * d * d, 8 * d * d, 4 * T_ * d], dtype=float)
            return v / v.sum()

        def bars():
            v = vals()
            g = VGroup()
            for i, (nm, c) in enumerate(names):
                top = ax.c2p(i + 0.5, v[i])[1]
                base = ax.c2p(i + 0.5, 0)[1]
                r = Rectangle(width=1.2, height=max(0.01, top - base), stroke_width=0, fill_color=c, fill_opacity=0.85)
                r.move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN)
                g.add(VGroup(r, T(f"{100 * v[i]:.0f}%", 18, c).next_to(r, UP, buff=0.08)))
            return g

        b = always_redraw(bars)
        lbls = VGroup(*[T(nm, 16, c).move_to(ax.c2p(i + 0.5, 0) + DOWN * 0.3) for i, (nm, c) in enumerate(names)])
        tl = always_redraw(lambda: T(f"decode FLOPs per token per layer · d = {d}, context T = {int(Ttr.get_value()):,}", 20).next_to(ax, UP, buff=0.3).to_edge(LEFT, buff=0.4))
        self.play(Create(ax), FadeIn(lbls))
        self.add(b, tl)
        params = VGroup(M(r"\text{weights: } W_{Q,K,V,O}=4d^2", size=30, color=ATTN), M(r"\mathrm{FFN}=8d^2\ (2\cdot d\cdot 4d\ \text{or SwiGLU } 3\cdot d\cdot\tfrac83 d)", size=30, color=MOE),
                        M(r"\Rightarrow\ \text{FFN}\approx\tfrac23\ \text{of the layer}", size=34)).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(RIGHT * 3.6 + UP * 0.6)
        params.scale_to_fit_width(6.0).to_edge(RIGHT, buff=0.3)
        self.play(FadeIn(params, lag_ratio=0.3), run_time=1.5)
        self.pause()
        self.play(Ttr.animate.set_value(4 * d), run_time=2.5)
        cross = T("crossover T ≈ 4d ≈ 16k tokens:\nscores catch up with the FFN", 20, Q_COL).next_to(params, DOWN, buff=0.4).to_edge(RIGHT, buff=0.3)
        self.play(FadeIn(cross))
        self.play(Ttr.animate.set_value(64 * 1024), run_time=2.5)
        q = T("Why must every token pay for the whole refinery?", 26, MOE).to_edge(DOWN, buff=0.35)
        self.play(Write(q))
        self.pause(1.5)
        b.clear_updaters(); tl.clear_updaters()
        self.clear_all()


class S18Clinic(Deck):
    def construct(self):
        self.header("18", "Mixture-of-Experts: the specialist clinic", ACT, "SCHEMATIC")
        n = 8
        doors = VGroup()
        for i in range(n):
            ang = PI * (0.95 - 0.9 * i / (n - 1))
            pos = np.array([4.2 * np.cos(ang), 2.6 * np.sin(ang) - 1.2, 0])
            door = RoundedRectangle(width=0.8, height=0.95, corner_radius=0.1, stroke_color=FADE, fill_color="#1B2030", fill_opacity=1).move_to(pos)
            doors.add(VGroup(door, glyph(i, FADE).move_to(door)))
        desk = RoundedRectangle(width=1.8, height=0.6, corner_radius=0.15, stroke_color=MOE, fill_color=MOE, fill_opacity=0.25).shift(DOWN * 1.6)
        desk_lab = T("triage desk = router", 18, MOE).next_to(desk, DOWN, buff=0.1)
        legend = T("patient = token · chart = residual vector · rooms = expert FFNs", 18, FADE).to_edge(DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(d, shift=UP * 0.2) for d in doors], lag_ratio=0.08), FadeIn(desk), FadeIn(desk_lab), FadeIn(legend))
        self.pause()

        def visit(label, color, scores, k=2, keep=False):
            pt = token_square(label, color=color, fs=18).next_to(desk, LEFT, buff=1.6)
            self.play(FadeIn(pt, shift=RIGHT * 0.5))
            self.play(pt.animate.next_to(desk, UP, buff=0.15))
            bars = VGroup()
            for d, s in zip(doors, scores):
                bars.add(Rectangle(width=0.12, height=0.05 + 0.8 * s, stroke_width=0, fill_color=MOE, fill_opacity=0.9).next_to(d, RIGHT, buff=0.05, aligned_edge=DOWN))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05), run_time=0.8)
            top = np.argsort(scores)[::-1][:k]
            paths = VGroup(*[CubicBezier(pt.get_top(), pt.get_top() + UP * 1.2, doors[i].get_bottom() + DOWN * 0.8, doors[i].get_bottom(), color=MOE, stroke_width=4) for i in top])
            self.play(*[doors[i][0].animate.set_stroke(MOE, 4).set_fill(MOE, 0.35) for i in top], *[doors[i][1].animate.set_stroke(MOE) for i in top], Create(paths))
            self.play(*[ShowPassingFlash(p.copy().set_stroke(RESID, 6), time_width=0.4) for p in paths], run_time=0.8)
            g = scores[top] / scores[top].sum()
            merge = T("chart += " + " + ".join(f"{w:.2f}·FFN{i + 1}" for w, i in zip(g, top)), 20, MOE).next_to(desk, LEFT, buff=0.3).shift(UP * 0.2)
            self.play(FadeIn(merge))
            self.pause(0.6)
            out = [pt, bars, paths, merge]
            if not keep:
                self.play(*[FadeOut(m) for m in out], *[doors[i][0].animate.set_stroke(FADE, 2).set_fill("#1B2030", 1) for i in top], *[doors[i][1].animate.set_stroke(FADE) for i in top])
            return out, top

        rng = np.random.default_rng(0)
        s1 = rng.uniform(0.05, 0.4, n); s1[2] = 0.95; s1[5] = 0.7
        visit("bank (money)", "#E86FA6", s1)
        s2 = rng.uniform(0.05, 0.4, n); s2[0] = 0.9; s2[6] = 0.75
        visit("bank (river)", ATTN, s2)
        note = T("routing is per token, per layer — toy weights, experts are not topic specialists", 18, FADE).next_to(desk_lab, DOWN, buff=0.1)
        self.play(FadeIn(note))
        self.pause()
        self.play(FadeOut(note))

        # shared expert (GP) + fine-grained combinatorics
        gp = RoundedRectangle(width=1.0, height=0.95, corner_radius=0.1, stroke_color=RESID, fill_color=RESID, fill_opacity=0.2).move_to(UP * 0.4)
        gp_lab = T("shared expert:\nthe GP everyone sees", 16, RESID).next_to(gp, DOWN, buff=0.08)
        self.play(FadeIn(gp), FadeIn(gp_lab))
        eq = M(r"\mathrm{MoE}(r)=\sum_{e\in\mathrm{TopK}(W_rr)} g_e(r)\,\mathrm{FFN}_e(r)+\sum_{j\in\mathcal S}\mathrm{FFN}^{\rm shared}_j(r)", size=30).next_to(self.mobjects[0], DOWN, buff=0.15).set_x(0)
        self.play(Write(eq))
        self.pause()
        combos = VGroup(M(r"\binom{16}{2}=120", size=34), M(r"\longrightarrow", size=34), M(r"\binom{64}{8}\approx 4.4\times10^{9}", size=34)).arrange(RIGHT).move_to(RIGHT * 0.0 + UP * 0.4)
        fg = T("fine-grained experts: same cost, vastly more referral combinations (DeepSeekMoE)", 18, MOE).next_to(combos, DOWN, buff=0.15)
        self.play(FadeOut(gp), FadeOut(gp_lab), doors.animate.set_opacity(0.15), FadeIn(combos), FadeIn(fg))
        self.pause()
        self.play(FadeOut(combos), FadeOut(fg), FadeOut(legend), doors.animate.set_opacity(1))
        bill = VGroup(T("stored: 8 rooms (memory)", 20, FADE), T("active: 2 rooms per token (FLOPs)", 20, MOE), T("dispatched: tokens moved between GPUs (comms)", 20, Q_COL)).arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=0.3)
        bill.scale_to_fit_width(12.5)
        self.play(LaggedStart(*[FadeIn(b) for b in bill], lag_ratio=0.3))
        self.pause(1.5)
        self.clear_all()


class S19ServingMoE(Deck):
    def construct(self):
        self.header("19", "How modern LLMs actually run MoE", ACT, "PROPERTY")
        n = 6
        rooms = VGroup(*[VGroup(RoundedRectangle(width=1.1, height=0.8, corner_radius=0.1, stroke_color=MOE), glyph(i, MOE)) for i in range(n)]).arrange(RIGHT, buff=0.5).shift(UP * 1.2)
        for r in rooms:
            r[1].move_to(r[0])
        self.play(FadeIn(rooms))
        rng = np.random.default_rng(1)
        dots = VGroup(*[Dot(radius=0.06, color=TOKEN).move_to([rng.uniform(-5, 5), -2.4 + rng.uniform(-0.3, 0.3), 0]) for _ in range(40)])
        self.play(FadeIn(dots, lag_ratio=0.02))
        # collapse: most go to room 3
        targets = [2 if rng.random() < 0.75 else rng.integers(0, n) for _ in dots]
        queue_anims = []
        counts = [0] * n
        for d, t in zip(dots, targets):
            counts[t] += 1
            queue_anims.append(d.animate.move_to(rooms[t].get_bottom() + DOWN * (0.15 + 0.12 * counts[t])))
        collapse = T("no balancing → routing collapse:\none queue, idle rooms", 22, MAMBA).move_to(RIGHT * 3.8 + DOWN * 0.6)
        self.play(LaggedStart(*queue_anims, lag_ratio=0.02), FadeIn(collapse), run_time=2)
        self.pause()
        # capacity + drop through residual
        cap = 6
        seats = DashedLine(rooms[2].get_bottom() + DOWN * (0.15 + 0.12 * cap + 0.06) + LEFT * 0.5, rooms[2].get_bottom() + DOWN * (0.15 + 0.12 * cap + 0.06) + RIGHT * 0.5, color=RESID)
        cap_lab = T("capacity", 16, RESID).next_to(seats, LEFT, buff=0.1)
        over = [d for d, t, k in zip(dots, targets, range(len(dots))) if t == 2][cap:]
        belt = Arrow(DOWN * 3.2 + LEFT * 6, DOWN * 3.2 + RIGHT * 6, color=RESID, stroke_width=3, buff=0)
        drop = T("overflow skips treatment and rides the residual unchanged (Switch)", 18, RESID).next_to(belt, UP, buff=0.05)
        self.play(FadeOut(collapse), Create(seats), FadeIn(cap_lab), Create(belt), FadeIn(drop))
        self.play(*[d.animate.move_to(belt.get_center() + RIGHT * rng.uniform(-5, 5)).set_color(FADE) for d in over], run_time=1.2)
        self.pause()
        # balancing: bias dials (aux-loss-free)
        self.play(FadeOut(dots), FadeOut(seats), FadeOut(cap_lab), FadeOut(belt), FadeOut(drop))
        dials = VGroup()
        for r in rooms:
            c = Circle(radius=0.18, stroke_color=FADE).next_to(r, UP, buff=0.15)
            dials.add(VGroup(c, Line(c.get_center(), c.get_center() + UP * 0.16, color=MOE, stroke_width=3)))
        bal = T("aux-loss-free balancing (DeepSeek-V3): a per-expert bias, used only for selection,\nnudged up for idle rooms and down for busy ones", 18, MOE).shift(DOWN * 0.3)
        self.play(FadeIn(dials), FadeIn(bal))
        self.play(*[Rotate(dl[1], angle=(-0.8 if i == 2 else 0.4), about_point=dl[0].get_center()) for i, dl in enumerate(dials)])
        even = VGroup(*[Dot(radius=0.06, color=TOKEN).move_to(rooms[i % n].get_bottom() + DOWN * (0.15 + 0.12 * (i // n))) for i in range(30)])
        self.play(FadeIn(even, lag_ratio=0.03), run_time=1.2)
        self.pause()
        self.play(FadeOut(VGroup(rooms, dials, bal, even)))

        # expert parallelism: GPUs + ambulances (all-to-all)
        gpus = VGroup(*[RoundedRectangle(width=2.6, height=1.5, corner_radius=0.15, stroke_color=ATTN) for _ in range(4)]).arrange_in_grid(2, 2, buff=(3.0, 0.6)).shift(UP * 0.2)
        glabs = VGroup()
        for gi, g in enumerate(gpus):
            ex = VGroup(*[glyph(2 * gi + j, MOE, 0.32) for j in range(2)]).arrange(RIGHT, buff=0.3).move_to(g)
            glabs.add(VGroup(ex, T(f"GPU {gi}", 16, ATTN).next_to(g, UP, buff=0.05)))
        hub = Dot(radius=0.15, color=Q_COL).move_to(gpus.get_center())
        self.play(FadeIn(gpus), FadeIn(glabs), FadeIn(hub))
        amb = VGroup()
        for gi in range(4):
            for gj in range(4):
                if gi != gj:
                    t = Triangle(color=Q_COL, fill_opacity=1).scale(0.08).move_to(gpus[gi].get_center())
                    amb.add(t)
        moves = []
        k = 0
        for gi in range(4):
            for gj in range(4):
                if gi != gj:
                    moves.append(MoveAlongPath(amb[k], ArcBetweenPoints(gpus[gi].get_center(), gpus[gj].get_center(), angle=PI / 4)))
                    k += 1
        a2a = T("all-to-all dispatch → expert compute → all-to-all combine: communication is a real cost", 18, Q_COL).to_edge(DOWN, buff=0.9)
        self.add(amb)
        self.play(*moves, FadeIn(a2a), run_time=1.6)
        self.play(*[m for m in [MoveAlongPath(amb[i], ArcBetweenPoints(amb[i].get_center(), gpus[[gi for gi in range(4) for gj in range(4) if gi != gj][i]].get_center(), angle=PI / 4)) for i in range(len(amb))]], run_time=1.2)
        mb = T("small-batch decode: many experts' weights read from HBM for few FLOPs → memory-bound", 18, FADE).next_to(a2a, DOWN, buff=0.12)
        self.play(FadeIn(mb))
        self.pause()
        self.play(FadeOut(VGroup(gpus, glabs, hub, amb, a2a, mb)))

        rows = [("2017", "Sparsely-gated MoE (Shazeer)", "noisy top-k, aux loss"),
                ("2020", "GShard", "top-2, expert parallelism"),
                ("2021", "Switch Transformer", "top-1, capacity factor, dropping"),
                ("2023/24", "Mixtral 8×7B", "top-2 of 8 · 47B total / 13B active"),
                ("2024", "DeepSeekMoE", "fine-grained + shared experts"),
                ("2024", "DeepSeek-V3", "256 routed + 1 shared, top-8, bias balancing · 671B / 37B"),
                ("2025", "Qwen3-MoE · Llama 4 · gpt-oss", "many experts, small top-k (see model cards)")]
        tab = VGroup(*[VGroup(T(a, 18, FADE), T(b, 18, MOE), T(c, 18, RESID)) for a, b, c in rows])
        for r in tab:
            r[1].move_to(r[0].get_center() + RIGHT * 1.6, aligned_edge=LEFT)
        for r in tab:
            r[0].set_x(-6.0, LEFT) if hasattr(r[0], "set_x") else None
        tab.arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(UP * 0.0)
        for r in tab:
            r[0].set_x(-5.8)
            r[1].move_to([-4.6, r[0].get_y(), 0], aligned_edge=LEFT)
            r[2].move_to([-0.4, r[0].get_y(), 0], aligned_edge=LEFT)
        verify = T("numbers: verify against model cards before presenting", 16, FADE).to_edge(DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in tab], lag_ratio=0.25), FadeIn(verify), run_time=2.5)
        self.pause(1.5)
        self.clear_all()


class S20ResidualStream(Deck):
    def construct(self):
        self.header("20", "The residual stream: how every layer talks to every other", ACT, "SCHEMATIC")
        L = 6
        x0 = -3.2
        rail = Line(DOWN * 2.6, UP * 2.2, color=RESID, stroke_width=3).set_x(x0)
        layer_cols = [ATTN, MOE, MAMBA2, MOE, ATTN, MOE]
        layer_names = ["attention", "FFN", "Mamba-2", "MoE", "attention", "FFN"]
        blocks = VGroup()
        for l in range(L):
            y = -2.2 + l * 0.78
            b = RoundedRectangle(width=1.6, height=0.5, corner_radius=0.1, stroke_color=layer_cols[l], fill_color=layer_cols[l], fill_opacity=0.15).move_to([x0 - 2.0, y, 0])
            blocks.add(VGroup(b, T(layer_names[l], 16, layer_cols[l]).move_to(b)))
        emb = T("E[τ]", 20, V_COL).next_to(rail, DOWN, buff=0.1)
        self.play(Create(rail), FadeIn(blocks, lag_ratio=0.1), FadeIn(emb))
        # running vector sum drawn to the right of the rail
        origin = np.array([x0 + 1.0, -2.4, 0])
        tip = origin.copy()
        arrows = VGroup()
        a0 = vec_arrow(origin, origin + np.array([0.3, 0.6, 0]), V_COL)
        self.play(GrowArrow(a0))
        tip = a0.get_end()
        rng = np.random.default_rng(2)
        for l in range(L):
            y = -2.2 + l * 0.78
            read = Line(rail.get_center() * 0 + [x0, y, 0], blocks[l][0].get_right(), color=layer_cols[l], stroke_width=2)
            write = Arrow(blocks[l][0].get_right() + UP * 0.12, [x0, y + 0.3, 0], buff=0, color=layer_cols[l], stroke_width=3, max_tip_length_to_length_ratio=0.25)
            d = np.array([rng.uniform(-0.3, 0.5), 0.55, 0])
            arr = vec_arrow(tip, tip + d, layer_cols[l], sw=4)
            self.play(Create(read), run_time=0.25)
            self.play(GrowArrow(write), GrowArrow(arr), run_time=0.4)
            arrows.add(arr)
            tip = tip + d
        total = vec_arrow(origin, tip, RESID, sw=6)
        self.play(GrowArrow(total))
        eq = M(r"r^{(\ell+1)}_t=r^{(\ell)}_t+F_\ell\big(\mathrm{Norm}(r^{(\ell)})\big)_t", r"\qquad r^{(L)}_t=E[\tau_t]+\sum_\ell u^{(\ell)}_t", size=30).to_edge(RIGHT, buff=0.3).shift(UP * 1.8)
        eq.scale_to_fit_width(6.4).move_to([3.5, 1.8, 0])
        self.play(Write(eq))
        bullets = VGroup(T("write = ADD (nothing is overwritten)", 18), T("read = linear projection (W_Q, W_K, B, C, …)", 18),
                         T("a shared channel (Elhage et al. 2021)", 18, FADE)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(eq, DOWN, buff=0.3).align_to(eq, LEFT)
        self.play(FadeIn(bullets, lag_ratio=0.3))
        self.pause()

        # logit lens readout
        lens_ax = VGroup()
        probs = [0.05, 0.08, 0.15, 0.35, 0.6, 0.82]
        lens = VGroup(*[Rectangle(width=0.18, height=0.02 + 1.2 * p, stroke_width=0, fill_color=Q_COL, fill_opacity=0.85) for p in probs]).arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
        lens.next_to(bullets, DOWN, buff=0.4).align_to(bullets, LEFT)
        ll = T("logit lens: P(\"bridge\") read off\nthe belt at each layer", 16, Q_COL).next_to(lens, RIGHT, buff=0.2)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in lens], lag_ratio=0.15), FadeIn(ll))
        self.pause()
        self.clear_all(0.5)

        # superposition toy + two-axes picture
        self.header("20", "More features than dimensions · two axes, two memories", ACT, "SCHEMATIC")
        pent = VGroup(*[vec_arrow(ORIGIN, 1.2 * np.array([np.cos(2 * PI * k / 5 + PI / 2), np.sin(2 * PI * k / 5 + PI / 2), 0]), c) for k, c in enumerate([Q_COL, K_COL, V_COL, MAMBA, MOE])])
        pent.shift(LEFT * 4.5 + DOWN * 0.2)
        pl = T("5 features in 2D: superposition\n(Elhage et al. 2022)", 18, FADE).next_to(pent, DOWN, buff=0.3)
        self.play(LaggedStart(*[GrowArrow(a) for a in pent], lag_ratio=0.15), FadeIn(pl))
        # grid: time horizontal, depth vertical
        nT, nL = 5, 4
        gx0, gy0 = -2.3, -2.0
        nodes = VGroup(*[Dot([gx0 + 1.3 * t, gy0 + 1.15 * l, 0], radius=0.07, color=RESID) for l in range(nL) for t in range(nT)])
        self.play(FadeIn(nodes))
        up = VGroup(*[Arrow([gx0 + 1.3 * t, gy0 + 1.15 * l, 0], [gx0 + 1.3 * t, gy0 + 1.15 * (l + 1), 0], buff=0.1, color=RESID, stroke_width=3) for l in range(nL - 1) for t in range(nT)])
        side = VGroup(*[Arrow([gx0 + 1.3 * t, gy0 + 1.15 * 1, 0], [gx0 + 1.3 * (t + 1), gy0 + 1.15 * 1, 0], buff=0.1, color=MAMBA2, stroke_width=4) for t in range(nT - 1)])
        attn = VGroup(*[CurvedArrow([gx0 + 1.3 * s, gy0 + 1.15 * 2, 0], [gx0 + 1.3 * 4, gy0 + 1.15 * 2 + 0.1, 0], angle=-PI / 4, color=ATTN, stroke_width=3) for s in (0, 2)])
        ax_t = T("time →", 18, FADE).next_to(nodes, DOWN, buff=0.25)
        ax_d = T("depth ↑", 18, FADE).next_to(nodes, LEFT, buff=0.25).rotate(PI / 2)
        self.play(FadeIn(ax_t), FadeIn(ax_d), LaggedStart(*[GrowArrow(a) for a in up], lag_ratio=0.03), run_time=1.2)
        lab_up = T("residual rₜ⁽ˡ⁾: across depth", 17, RESID).next_to(nodes, RIGHT, buff=0.3).shift(UP * 1.2)
        self.play(FadeIn(lab_up))
        self.play(LaggedStart(*[GrowArrow(a) for a in side], lag_ratio=0.2), run_time=1.2)
        lab_side = T("Mamba state Sₜ⁽ˡ⁾:\nacross time, one layer", 17, MAMBA2).next_to(lab_up, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(lab_side))
        self.play(*[Create(a) for a in attn])
        lab_attn = T("attention: reaches sideways\nto earlier positions' K/V", 17, ATTN).next_to(lab_side, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(lab_attn))
        self.pause()
        self.play(FadeOut(VGroup(pent, pl)))
        stack = VGroup(*[RoundedRectangle(width=0.5, height=0.5, corner_radius=0.08, stroke_color=c, fill_color=c, fill_opacity=0.4) for c in
                         [MAMBA2, MOE, MAMBA2, MOE, ATTN, MOE, MAMBA2, MOE]]).arrange(UP, buff=0.06).to_edge(LEFT, buff=0.8).shift(DOWN * 0.3)
        tags = VGroup(*[T(ch, 16, BG).move_to(b) for ch, b in zip("MEMEAEME", stack)])
        hyb = T("hybrid stack\nM = Mamba-2\nA = attention\nE = MoE/FFN\n\nJamba · Nemotron-H\n· Nemotron 3 Nano\n(verify layer order)", 16, FADE).next_to(stack, RIGHT, buff=0.25)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in stack], lag_ratio=0.1), FadeIn(tags), FadeIn(hyb))
        self.pause(1.5)
        self.clear_all()


class S21TransformerToolbox(Deck):
    def construct(self):
        self.header("21", "\"Transformers aren't that naive\": the real toolbox", ACT, "PROPERTY")
        n = 8

        def mini(mask, color=ATTN):
            return heatmap(mask, cell=0.17, pos_color=color)

        causal = np.tril(np.ones((n, n)))
        band = np.array([[1.0 if 0 <= i - j < 3 else 0 for j in range(n)] for i in range(n)])
        sink = band.copy(); sink[:, 0] = np.tril(np.ones((n, n)))[:, 0]
        sparse = band.copy(); sparse[:, 0] = causal[:, 0]; sparse[5:, 2] = 1
        panels = [
            ("MQA", "Shazeer 2019 · PaLM, Falcon", "all heads share 1 K/V", "kv"),
            ("GQA", "Ainslie et al. 2023 · Llama 2-70B, Llama 3", "groups share K/V", "kv2"),
            ("Sliding window", "Longformer 2020 · Mistral 7B", "T² → TW, KV → W", band),
            ("Local : global", "Gemma 2 (1:1), Gemma 3 (5:1)", "most layers windowed", "interleave"),
            ("Attention sinks", "StreamingLLM, Xiao et al. 2023", "stable infinite streams", sink),
            ("MLA", "DeepSeek-V2 2024", "K/V compressed to a latent", "mla"),
            ("FlashAttention", "Dao et al. 2022 / 23 / 24", "IO ↓ · FLOPs still T²", "flash"),
            ("PagedAttention", "Kwon et al. 2023 · vLLM", "KV in pages, less waste", "paged"),
            ("Sparse / NSA", "BigBird 2020 · DeepSeek NSA 2025", "select a few blocks", sparse),
            ("RoPE + YaRN", "Su et al. 2021 · Peng et al. 2023", "rotate Q/K; extend context", "rope"),
        ]
        cells = VGroup()
        for name, who, what, kind in panels:
            if isinstance(kind, np.ndarray):
                icon = mini(kind)
            elif kind == "kv":
                icon = VGroup(VGroup(*[Rectangle(width=0.14, height=0.5, stroke_width=0, fill_color=Q_COL, fill_opacity=0.8) for _ in range(6)]).arrange(RIGHT, buff=0.04),
                              Rectangle(width=0.5, height=0.5, stroke_width=0, fill_color=K_COL, fill_opacity=0.8)).arrange(RIGHT, buff=0.15)
            elif kind == "kv2":
                icon = VGroup(VGroup(*[Rectangle(width=0.14, height=0.5, stroke_width=0, fill_color=Q_COL, fill_opacity=0.8) for _ in range(6)]).arrange(RIGHT, buff=0.04),
                              VGroup(*[Rectangle(width=0.22, height=0.5, stroke_width=0, fill_color=K_COL, fill_opacity=0.8) for _ in range(2)]).arrange(RIGHT, buff=0.05)).arrange(RIGHT, buff=0.15)
            elif kind == "interleave":
                icon = VGroup(mini(band).scale(0.55), mini(causal).scale(0.55), mini(band).scale(0.55)).arrange(UP, buff=0.04)
            elif kind == "mla":
                icon = VGroup(Rectangle(width=0.5, height=1.0, stroke_width=0, fill_color=K_COL, fill_opacity=0.8), Arrow(LEFT * 0.3, RIGHT * 0.3, buff=0, color=FADE),
                              Rectangle(width=0.5, height=0.3, stroke_width=0, fill_color=K_COL, fill_opacity=0.8)).arrange(RIGHT, buff=0.08)
            elif kind == "flash":
                icon = VGroup(mini(causal), Square(0.36, stroke_color=MOE, stroke_width=3))
                icon[1].move_to(icon[0][0]).shift(RIGHT * 0.09 + DOWN * 0.09)
            elif kind == "paged":
                icon = VGroup(*[Rectangle(width=0.3, height=0.22, stroke_color=V_COL, fill_color=V_COL, fill_opacity=0.5 if (i * 7) % 3 else 0.05) for i in range(12)]).arrange_in_grid(3, 4, buff=0.05)
            else:  # rope
                icon = VGroup(Circle(radius=0.45, stroke_color=FADE), vec_arrow(ORIGIN, 0.45 * RIGHT, Q_COL), vec_arrow(ORIGIN, 0.45 * np.array([np.cos(1), np.sin(1), 0]), K_COL))
            icon.scale_to_fit_height(min(1.1, icon.height))
            title = T(name, 20, RESID)
            who_t = T(who, 13, FADE)
            what_t = T(what, 14, MOE)
            card = VGroup(title, icon, what_t, who_t).arrange(DOWN, buff=0.12)
            frame = RoundedRectangle(width=2.45, height=2.55, corner_radius=0.12, stroke_color="#2A3040").move_to(card)
            for m in (who_t, what_t, title):
                if m.width > 2.3:
                    m.scale_to_fit_width(2.3)
            cells.add(VGroup(frame, card))
        cells.arrange_in_grid(2, 5, buff=(0.12, 0.2)).shift(DOWN * 0.35)
        for c in cells:
            self.play(FadeIn(c, shift=UP * 0.2), run_time=0.55)
            if c is cells[6]:
                tile = c[1][1][1]
                self.play(tile.animate.shift(RIGHT * 0.34), run_time=0.3)
                self.play(tile.animate.shift(DOWN * 0.34 + LEFT * 0.34), run_time=0.3)
        self.pause()
        summary = T("GQA, MQA, MLA, Flash keep every card addressable — windows give up distant cards.\nHybrids take another route: replace most walls with notebooks.", 20).to_edge(DOWN, buff=0.15)
        self.play(cells.animate.shift(UP * 0.25), FadeIn(summary))
        self.pause(1.5)
        self.clear_all()
