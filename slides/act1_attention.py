"""Act I — Theo's archive (slides 1–6)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *

FIN = "#E86FA6"  # finance sense of "bank" (not a palette object)
STORY = ["Leo", "green key", "bridge", "Noah", "red key", "cupboard"]


class S01ColdOpen(Deck):
    def construct(self):
        theo = brother("Theo", ATTN, "cards")
        manu = brother("Manu", MAMBA, "notebook")
        maxx = brother("Max", MAMBA2, "ledger")
        cast = VGroup(theo, manu, maxx).arrange(RIGHT, buff=2.4).shift(DOWN * 1.3)
        ribbon = VGroup(*[token_square(w, fs=20) for w in STORY]).arrange(RIGHT, buff=0.2).to_edge(UP, buff=0.9)
        ribbon.shift(RIGHT * 9)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in cast], lag_ratio=0.25))
        self.play(ribbon.animate.shift(LEFT * 9), run_time=2.2, rate_func=smooth)
        self.pause()

        # Theo pins every card; Manu rewrites one page; Max batches into tiles
        wall = VGroup(*[token_card(w[:6], w=0.55, h=0.7, size=11) for w in STORY]).arrange(RIGHT, buff=0.06)
        wall.next_to(theo, UP, buff=0.35)
        page = Rectangle(width=1.6, height=1.0, stroke_color=MAMBA, fill_color=MAMBA, fill_opacity=0.15).next_to(manu, UP, buff=0.35)
        tiles = VGroup(*[Square(0.45, stroke_color=MAMBA2, fill_color=MAMBA2, fill_opacity=0.2) for _ in range(2)]).arrange(RIGHT, buff=0.1)
        tiles.next_to(maxx, UP, buff=0.4)
        self.play(TransformFromCopy(ribbon, wall), FadeIn(page), FadeIn(tiles), run_time=1.5)
        for i in range(3):
            ink = Line(LEFT * 0.6, RIGHT * 0.6, color=MAMBA, stroke_width=3).move_to(page).shift(UP * (0.3 - 0.25 * i))
            self.play(page.animate.set_fill(opacity=0.15 + 0.1 * i), Create(ink), tiles[i % 2].animate.set_fill(opacity=0.6), run_time=0.5)
        self.pause()

        q = token_square("Where is Leo's key?", color=Q_COL, fs=22).next_to(ribbon, DOWN, buff=0.25)
        self.play(FadeIn(q, shift=DOWN * 0.3))
        self.play(Indicate(wall[2], color=Q_COL, scale_factor=1.4),
                  page.animate.set_fill(opacity=0.6), tiles.animate.set_fill(opacity=0.9), run_time=1.4)
        self.pause()
        title = T("Same question. Three ways to remember.", 40).to_edge(DOWN, buff=0.4)
        self.play(Write(title))
        self.pause(1.5)
        self.clear_all()


class S02QKVLibrary(Deck):
    def construct(self):
        self.header("2", "Query, Key, Value as a library", "Act I · Theo's archive")
        tok = token_square("r_t", fs=22).shift(LEFT * 5 + UP * 1.2)
        lenses = VGroup(*[Ellipse(width=0.35, height=1.0, stroke_color=c, fill_color=c, fill_opacity=0.25)
                          for c in (Q_COL, K_COL, V_COL)]).arrange(DOWN, buff=0.25).next_to(tok, RIGHT, buff=0.9)
        lab = VGroup(M("W_Q", color=Q_COL, size=30), M("W_K", color=K_COL, size=30), M("W_V", color=V_COL, size=30))
        for l, e in zip(lab, lenses):
            l.next_to(e, LEFT, buff=0.1).shift(UP * 0.35)
        outs = VGroup(T("q: what I'm looking for", 20, Q_COL), T("k: how I can be found", 20, K_COL),
                      T("v: what I hand over", 20, V_COL))
        for o, e in zip(outs, lenses):
            o.next_to(e, RIGHT, buff=0.4)
        self.play(FadeIn(tok), FadeIn(lenses), FadeIn(lab))
        self.play(LaggedStart(*[TransformFromCopy(tok, o) for o in outs], lag_ratio=0.3), run_time=1.8)
        self.pause()
        self.play(FadeOut(VGroup(tok, lenses, lab, outs)))

        cards = VGroup(*[token_card(w, w=1.5, h=1.5, size=18) for w in STORY]).arrange(RIGHT, buff=0.25).shift(DOWN * 2.1)
        query = token_square("q: Leo's key → where?", color=Q_COL, fs=20).shift(UP * 1.7)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.4) for c in cards], lag_ratio=0.12), FadeIn(query))
        # dot-product beam sweep with graded scores
        scores = np.array([1.6, 0.9, 2.6, 0.2, 0.6, 0.8])
        bars = VGroup()
        for c, s in zip(cards, scores):
            b = Rectangle(width=0.5, height=0.35 * s, stroke_width=0, fill_color=K_COL, fill_opacity=0.9)
            b.next_to(c, UP, buff=0.1)
            bars.add(b)
        beam = Line(query.get_bottom(), cards[0][1].get_top(), color=Q_COL, stroke_width=4)
        self.play(Create(beam))
        for i in range(6):
            self.play(beam.animate.put_start_and_end_on(query.get_bottom(), cards[i][1].get_top()),
                      GrowFromEdge(bars[i], DOWN), run_time=0.45)
        sc = M(r"\text{score}_{s}=q_t^\top k_s", size=34).move_to(LEFT * 3.6 + UP * 0.9)
        self.play(FadeOut(beam), Write(sc))
        self.pause()

        # softmax: exponentiate then normalise into a shared 100% budget
        w = np.exp(scores) / np.exp(scores).sum()
        new_bars = VGroup()
        for c, wi in zip(cards, w):
            nb = Rectangle(width=0.5, height=max(0.03, 2.2 * wi), stroke_width=0, fill_color=K_COL, fill_opacity=0.9)
            nb.next_to(c, UP, buff=0.1)
            new_bars.add(nb)
        pct = VGroup(*[T(f"{100*wi:.0f}%", 16, RESID).next_to(nb, UP, buff=0.05) for nb, wi in zip(new_bars, w)])
        sm = M(r"\alpha_{ts}=\mathrm{softmax}_s\!\left(\tfrac{q_t^\top k_s}{\sqrt{d_k}}\right)", size=34).move_to(sc)
        self.play(Transform(bars, new_bars), FadeIn(pct), TransformMatchingTex(sc, sm), run_time=1.5)
        self.pause()

        # values flow back, scaled by weight, summed tip to tail
        out = RoundedRectangle(width=1.6, height=1.0, corner_radius=0.1, stroke_color=V_COL,
                               fill_color=V_COL, fill_opacity=0.5).next_to(query, RIGHT, buff=1.2)
        out_lab = T("o_t = blend", 18, BG).move_to(out)
        flights = [c[0].copy().set_fill(V_COL, opacity=0.7 * wi + 0.05) for c, wi in zip(cards, w)]
        self.play(LaggedStart(*[f.animate.scale(0.25).move_to(out) for f in flights], lag_ratio=0.1), run_time=1.6)
        self.play(FadeIn(out), FadeIn(out_lab), *[FadeOut(f) for f in flights])
        eq = M(r"o_t=\sum_{s\le t}\alpha_{ts}\,v_s", size=40, color=V_COL).move_to(RIGHT * 3.8 + UP * 0.4)
        self.play(Write(eq))
        self.pause()
        mat = M(r"O=\mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}+\mathcal M_{\rm causal}\Big)V", size=40)
        self.play(FadeOut(VGroup(pct, bars, eq, sm, out, out_lab)), query.animate.shift(UP * 0.2))
        mat.next_to(query, DOWN, buff=0.4)
        self.play(Write(mat))
        self.dont_say("attention finds *the* matching card. It is always a soft weighted average.")
        self.pause(1.5)
        self.clear_all()


class S03PrefillKVCache(Deck):
    def construct(self):
        self.header("3", "Prefill, the KV cache, and O(T) decode", "Act I · Theo's archive", "PROPERTY")
        n = 8
        cell = 0.36

        def tri(n, color=ATTN, op=0.75):
            g = VGroup()
            for i in range(n):
                for j in range(i + 1):
                    g.add(Square(cell, stroke_width=0.8, stroke_color=BG, fill_color=color, fill_opacity=op).move_to([j * cell, -i * cell, 0]))
            return g

        # 1. wasteful: recompute the whole triangle for every new token
        lbl = T("No cache: recompute everything per new token", 24, FADE).shift(UP * 2.1)
        self.play(FadeIn(lbl))
        pile = VGroup()
        grid = tri(n).move_to(LEFT * 2.5 + DOWN * 0.3)
        self.play(FadeIn(grid, lag_ratio=0.02), run_time=1)
        counter = M(r"\text{work for this token}\ \propto T^2", size=34).next_to(grid, RIGHT, buff=1)
        self.play(Write(counter))
        for k in range(3):
            ghost = grid.copy().set_fill(FADE, 0.35).scale(0.35)
            ghost.move_to(RIGHT * (3.4 + 0.4 * k) + DOWN * (1.4 - 0.25 * k))
            self.play(TransformFromCopy(grid, ghost), run_time=0.55)
            pile.add(ghost)
        pile_lab = T("redo pile", 20, FADE).next_to(pile, DOWN, buff=0.1)
        self.play(FadeIn(pile_lab))
        self.pause()

        # 2. past K and V are final under a causal mask
        col = VGroup(*[sq for sq in grid if abs(sq.get_center()[0] - grid[0].get_center()[0]) < 1e-3])
        stamp = T("final: K,V never change", 22, K_COL).next_to(grid, UP, buff=0.15)
        self.play(col.animate.set_fill(K_COL, 0.9), FadeIn(stamp))
        self.pause()

        # 3. KV cache: append one card per token, one query row scans
        self.play(FadeOut(VGroup(pile, pile_lab, counter, lbl, stamp)), grid.animate.set_fill(ATTN, 0.75))
        self.play(grid.animate.scale(0.9).move_to(LEFT * 3.8 + DOWN * 0.3))
        pre_lab = T("PREFILL: whole prompt in parallel\nfills the cache · O(T²) scores", 20, ATTN).next_to(grid, UP, buff=0.25)
        self.play(FadeIn(pre_lab))
        drawers = VGroup(*[token_card("", w=0.38, h=0.55) for _ in range(n)]).arrange(RIGHT, buff=0.05).move_to(RIGHT * 2.6 + UP * 0.2)
        dec_lab = T("DECODE: one new row scans T cached cards", 20, Q_COL).next_to(drawers, UP, buff=0.5)
        self.play(TransformFromCopy(grid, drawers), FadeIn(dec_lab))
        meter_bg = Rectangle(width=4.8, height=0.25, stroke_color=FADE).next_to(drawers, DOWN, buff=1.0).align_to(drawers, LEFT)
        meter = Rectangle(width=4.8 * n / 14, height=0.25, stroke_width=0, fill_color=V_COL, fill_opacity=0.8).align_to(meter_bg, LEFT).move_to(meter_bg, coor_mask=[0, 1, 0])
        mlab = T("KV cache memory  ∝ T", 18, V_COL).next_to(meter_bg, DOWN, buff=0.1)
        self.play(FadeIn(meter_bg), FadeIn(meter), FadeIn(mlab))
        for k in range(4):
            new = token_card("", w=0.38, h=0.55).next_to(drawers, RIGHT, buff=0.05)
            qrow = Rectangle(width=drawers.width + 0.4, height=0.12, stroke_width=0, fill_color=Q_COL, fill_opacity=0.8).next_to(drawers, DOWN, buff=0.12).align_to(drawers, LEFT)
            self.play(FadeIn(new, shift=LEFT * 0.3), run_time=0.35)
            drawers.add(new)
            self.play(GrowFromEdge(qrow, LEFT), meter.animate.stretch_to_fit_width(4.8 * (n + k + 1) / 14).align_to(meter_bg, LEFT), run_time=0.6)
            self.play(FadeOut(qrow), run_time=0.2)
        self.pause()

        eqs = M(r"\underbrace{O(T^2 d)}_{\text{prefill}}\quad \underbrace{O(T d)}_{\text{one decode step}}\quad \underbrace{2\,T\,H_{kv}\,d_h}_{\text{cache / layer}}", size=36).to_edge(DOWN, buff=1.2)
        self.play(Write(eqs))
        ex = T("Llama-2-7B (MHA) ≈ 0.5 MB / token · Llama-3-8B (GQA) ≈ 128 KB / token  (fp16, verify)", 18, FADE).next_to(eqs, DOWN, buff=0.15)
        self.play(FadeIn(ex))
        self.pause()
        self.play(FadeOut(ex))
        self.dont_say("cached decode is O(1). It is O(T) per token and O(T) memory; the cache is not prefill.")
        self.pause(1.5)
        self.clear_all()


class S04MultiHead(Deck):
    def construct(self):
        self.header("4", "Many heads: why separate Q, K and V?", "Act I · Theo's archive", "SCHEMATIC")
        one = token_square("q = k = v ?", color=FADE, fs=24).shift(UP * 1.2)
        why = T("\"what I want\" = \"how I'm found\" = \"what I give\"", 22, FADE).next_to(one, DOWN, buff=0.3)
        self.play(FadeIn(one), FadeIn(why))
        cross = Cross(VGroup(one, why), stroke_color=MAMBA, stroke_width=6)
        self.play(Create(cross))
        slogan = T("QKᵀ = address bus (where to look)     V = data bus (what moves)", 26).next_to(why, DOWN, buff=0.6)
        self.play(Write(slogan))
        self.pause()
        self.play(FadeOut(VGroup(one, why, cross)), slogan.animate.scale(0.8).to_edge(UP, buff=1.2))

        # three heads: three projections -> three patterns over the same sentence
        n = 6
        pats = []
        p1 = np.full((n, n), 0.03); p1[4, 0] = 1; p1[5, 0] = 0.7; p1[2, 1] = .8
        p2 = np.full((n, n), 0.03); p2[2, 1] = 1; p2[5, 4] = 1
        p3 = np.zeros((n, n))
        for i in range(n): p3[i, max(0, i - 1)] = 1
        for p in (p1, p2, p3):
            p[np.triu_indices(n, 1)] = 0
            pats.append(p)
        cols = (ATTN, K_COL, Q_COL)
        names = ("head 1: key → Leo", "head 2: bridge → under", "head 3: previous token")
        maps = VGroup()
        for p, c, nm in zip(pats, cols, names):
            h = heatmap(p, cell=0.32, pos_color=c)
            maps.add(VGroup(h, T(nm, 18, c).next_to(h, DOWN, buff=0.15)))
        maps.arrange(RIGHT, buff=0.9).shift(DOWN * 0.4)
        prisms = VGroup(*[Triangle(stroke_color=c, fill_color=c, fill_opacity=0.3).scale(0.3).next_to(m, UP, buff=0.2) for m, c in zip(maps, cols)])
        self.play(LaggedStart(*[AnimationGroup(FadeIn(pr), FadeIn(m, shift=UP * 0.2)) for pr, m in zip(prisms, maps)], lag_ratio=0.3), run_time=2)
        ill = T("illustrative possibilities, not guaranteed roles", 18, FADE).next_to(maps, DOWN, buff=0.35)
        self.play(FadeIn(ill))
        self.pause()

        concat = VGroup(*[Rectangle(width=1.0, height=0.35, stroke_width=0, fill_color=c, fill_opacity=0.8) for c in cols]).arrange(RIGHT, buff=0)
        concat.next_to(ill, DOWN, buff=0.35).shift(LEFT * 1.5)
        wo = M(r"\xrightarrow{\;W_O\;}\ \text{write to residual}", size=30).next_to(concat, RIGHT, buff=0.3)
        self.play(*[TransformFromCopy(m[0], c) for m, c in zip(maps, concat)], run_time=1.2)
        self.play(Write(wo))
        self.pause()
        eq = M(r"\mathrm{head}_h=\mathrm{softmax}\!\Big(\tfrac{Q_hK_{g(h)}^\top}{\sqrt{d_h}}+\mathcal M\Big)V_{g(h)}", size=34)
        eq.next_to(slogan, DOWN, buff=0.25)
        self.play(FadeOut(VGroup(ill, concat, wo)), maps.animate.shift(DOWN * 0.4), Write(eq))
        teaser = T("Teaser: do all heads need their own K/V?  g(h) → GQA (slide 21)", 20, FADE).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(teaser), prisms[1].animate.set_opacity(0.2), prisms[2].animate.set_opacity(0.2))
        self.pause(1.5)
        self.clear_all()


class S05BankResidual(Deck3D):
    def construct(self):
        title = T("5  \"Bank\": embedding = start, residual stream = journey", 30).to_corner(UL, buff=0.35)
        badge = evidence_badge("SCHEMATIC").to_corner(UR, buff=0.35)
        lanes = VGroup(T("…watched fish near the bank", 22, ATTN), T("…deposited money at the bank", 22, FIN)).arrange(DOWN, aligned_edge=LEFT).next_to(title, DOWN, aligned_edge=LEFT, buff=0.25)
        self.add_fixed_in_frame_mobjects(title, badge, lanes)
        self.play(FadeIn(title), FadeIn(badge), FadeIn(lanes))
        axes = ThreeDAxes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-2, 2.5, 1], x_length=6, y_length=6, z_length=4.5,
                          axis_config={"stroke_color": FADE, "stroke_width": 1.5})
        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES, zoom=0.9)
        self.play(Create(axes), run_time=1)
        start = np.array([0.8, 0.6, 0.6])
        emb = T("E[bank]: identical start in both sentences", 22, V_COL).to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(emb)
        a0 = Arrow3D(ORIGIN, axes.c2p(*start), color=V_COL)
        self.play(Create(a0), FadeIn(emb))
        self.pause()

        deltas = {"river": ([-0.7, 0.6, 0.3], [-0.9, 0.5, 0.25], [-0.6, 0.4, 0.2]),
                  "money": ([0.7, -0.6, 0.3], [0.8, -0.7, 0.25], [0.5, -0.6, 0.2])}
        cols = {"river": ATTN, "money": FIN}
        self.play(FadeOut(emb))
        step = T("each layer ADDS an update: r⁽ˡ⁺¹⁾ = r⁽ˡ⁾ + u⁽ˡ⁾", 24).to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(step)
        self.play(FadeIn(step))
        finals = {}
        for i in range(3):
            anims = []
            for lane in ("river", "money"):
                tip = start + np.sum(deltas[lane][:i], axis=0) if i else start
                new = tip + np.array(deltas[lane][i])
                anims.append(Create(Arrow3D(axes.c2p(*tip), axes.c2p(*new), color=cols[lane], thickness=0.015)))
                finals[lane] = new
            self.play(*anims, run_time=0.9)
        res = [Arrow3D(ORIGIN, axes.c2p(*finals[l]), color=cols[l], thickness=0.03) for l in ("river", "money")]
        tags = [T("river-ish", 26, ATTN), T("finance-ish", 26, FIN)]
        self.play(*[Create(r) for r in res])
        for tg, l in zip(tags, ("river", "money")):
            tg.rotate(PI / 2, RIGHT).move_to(axes.c2p(*(finals[l] * 1.25)))
        self.play(*[FadeIn(t) for t in tags])
        self.pause()
        self.play(FadeOut(step))
        eq = M(r"r_t^{(L)}=E[\tau_t]+\sum_{\ell<L}u_t^{(\ell)}", size=40).to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(Write(eq))
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        note = T("RoPE models inject position inside Q/K, not into the residual", 18, FADE).next_to(eq, UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.pause(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


class S06LongContextTax(Deck):
    def construct(self):
        self.header("6", "The long-context tax", "Act I · Theo's archive", "PROPERTY")
        sizes = [4, 8, 16, 32]
        grid = None
        lab = None
        for n in sizes:
            cell = 4.0 / n
            g = VGroup(*[Square(cell, stroke_width=0.5 if n < 32 else 0.2, stroke_color=BG, fill_color=ATTN, fill_opacity=0.75).move_to([j * cell, -i * cell, 0])
                         for i in range(n) for j in range(i + 1)])
            g.move_to(LEFT * 3.2 + DOWN * 0.4)
            l = M(rf"T={n * 16}\quad \tfrac{{T^2}}{{2}}\approx {(n * 16) ** 2 // 2:,}".replace(",", "{,}"), size=30).next_to(g, UP, buff=0.2)
            if grid is None:
                self.play(FadeIn(g), FadeIn(l))
            else:
                self.play(ReplacementTransform(grid, g), ReplacementTransform(lab, l), run_time=0.8)
            grid, lab = g, l
        self.pause()
        ax = Axes(x_range=[0, 10, 2], y_range=[0, 10, 2], x_length=4.8, y_length=3.2, tips=False,
                  axis_config={"stroke_color": FADE, "include_numbers": False}).shift(RIGHT * 3.2 + DOWN * 0.3)
        quad = ax.plot(lambda x: x ** 2 / 10, x_range=[0, 10], color=ATTN)
        lin = ax.plot(lambda x: x * 0.6, x_range=[0, 10], color=V_COL)
        xl = T("context T", 18, FADE).next_to(ax, DOWN, buff=0.1)
        ql = T("prefill compute ∝ T²", 18, ATTN).next_to(ax.c2p(9, 8.1), LEFT, buff=0.2)
        ll = T("KV memory ∝ T", 18, V_COL).next_to(ax.c2p(10, 6), DOWN, buff=0.2).shift(LEFT * 0.6)
        self.play(Create(ax), FadeIn(xl))
        self.play(Create(quad), Create(lin), FadeIn(ql), FadeIn(ll), run_time=1.5)
        asym = T("asymptotic illustration, not a benchmark", 16, FADE).next_to(ax, UP, buff=0.1)
        self.play(FadeIn(asym))
        self.pause()
        pipe = VGroup(Rectangle(width=6.5, height=0.5, stroke_color=V_COL, fill_color=V_COL, fill_opacity=0.25), T("decode: HBM → chip must stream the whole KV wall for every token", 18, V_COL))
        pipe[1].next_to(pipe[0], UP, buff=0.08)
        pipe.to_edge(DOWN, buff=0.9)
        self.play(FadeIn(pipe))
        self.play(pipe[0].animate.stretch_to_fit_height(0.18), run_time=1.2)
        theo = brother("Theo", ATTN, "cards").scale(0.6).to_edge(DOWN, buff=0.15).to_edge(LEFT, buff=0.4)
        bubble = T("What if my notebook never grew?", 22, MAMBA).next_to(theo, RIGHT, buff=0.3)
        self.play(FadeIn(theo), Write(bubble))
        self.pause(1.5)
        self.clear_all()
