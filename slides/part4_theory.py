"""Part 4 — Real Transformers, the trade-off, universality, the derivation, interpretability (S13–S16)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *
from math_contracts.test_ssd_equivalence import ssd_recurrent, ssd_matrix

PROB = "#FF6B81"


# ----------------------------------------------------------- toolbox icons
def _mask(pattern, color, n=6, cell=0.17):
    g = VGroup()
    for i in range(n):
        for j in range(i + 1):
            on = pattern(i, j)
            g.add(Square(cell, stroke_width=0.6, stroke_color=BG1, fill_color=color if on else SOFT, fill_opacity=0.95 if on else 0.6)
                  .move_to([j * cell, -i * cell, 0]))
    return g.move_to(ORIGIN)


def _qkv(nq, nkv, color):
    qs = VGroup(*[Dot(radius=0.065, color=Q_COL) for _ in range(nq)]).arrange(RIGHT, buff=0.2)
    ks = VGroup(*[Dot(radius=0.075, color=color) for _ in range(nkv)]).arrange(RIGHT, buff=0.2 * nq / nkv + 0.0)
    ks.move_to(qs).shift(DOWN * 0.8)
    lines = VGroup(*[Line(q.get_center(), ks[min(nkv - 1, i * nkv // nq)].get_center(), color=color, stroke_width=1.8, stroke_opacity=0.8)
                     for i, q in enumerate(qs)])
    return VGroup(lines, qs, ks)


def icon(kind, color):
    if kind == "mqa":
        return _qkv(6, 1, color)
    if kind == "gqa":
        return _qkv(6, 2, color)
    if kind == "swa":
        return _mask(lambda i, j: i - j <= 1, color)
    if kind == "localglobal":
        return _mask(lambda i, j: i - j <= 1 or j % 3 == 0, color)
    if kind == "sink":
        return _mask(lambda i, j: j == 0 or i - j <= 1, color)
    if kind == "sparse":
        return _mask(lambda i, j: (i * 3 + j * 5) % 4 == 0 or i == j, color)
    if kind == "mla":
        wide = Rectangle(width=1.0, height=0.28, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.35)
        lat = Rectangle(width=0.35, height=0.28, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.9)
        wide2 = wide.copy()
        g = VGroup(wide, lat, wide2).arrange(DOWN, buff=0.22)
        arrs = VGroup(*[Arrow(a.get_bottom(), b.get_top(), buff=0.03, stroke_width=2, tip_length=0.08, color=FADE) for a, b in [(wide, lat), (lat, wide2)]])
        return VGroup(g, arrs)
    if kind == "flash":
        grid = VGroup(*[Square(0.26, stroke_color=color, stroke_width=1.5, fill_color=color, fill_opacity=0.12) for _ in range(9)]).arrange_in_grid(3, 3, buff=0.04)
        grid[4].set_fill(color, 0.9)
        return grid
    if kind == "paged":
        cols = [color, V_COL, MAMBA2, K_COL]
        return VGroup(*[Rectangle(width=0.3, height=0.5, stroke_color=c, stroke_width=2, fill_color=c, fill_opacity=0.5) for c in cols]).arrange(RIGHT, buff=0.14).rotate(0)
    if kind == "rope":
        c = Circle(radius=0.45, stroke_color=color, stroke_width=3)
        a = arrow(ORIGIN, UP * 0.4 + RIGHT * 0.2, color, 4, 0.12)
        b = arrow(ORIGIN, RIGHT * 0.42 + UP * 0.02, Q_COL, 3, 0.1).rotate(-0.5, about_point=ORIGIN)
        return VGroup(c, a, b)
    return VGroup()


class S13Toolbox(Deck):
    def construct(self):
        self.title("Real Transformers are cleverer than that", ATTN)
        items = [
            ("MQA", "Shazeer 2019", "PaLM", "mqa", MAMBA2),
            ("GQA", "Ainslie et al. 2023", "Llama 2 70B, Llama 3", "gqa", MAMBA2),
            ("Sliding window", "Longformer 2020", "Mistral 7B", "swa", V_COL),
            ("Local + global", "Google 2024", "Gemma 2 / 3", "localglobal", V_COL),
            ("Attention sinks", "Xiao et al. 2023", "StreamingLLM", "sink", V_COL),
            ("MLA", "DeepSeek 2024", "DeepSeek-V2 / V3", "mla", MAMBA2),
            ("FlashAttention", "Dao et al. 2022", "nearly every LLM", "flash", Q_COL),
            ("PagedAttention", "vLLM 2023", "LLM serving", "paged", Q_COL),
            ("Sparse attention", "BigBird 2020 · NSA 2025", "DeepSeek", "sparse", K_COL),
            ("RoPE + YaRN", "Su 2021 · Peng 2023", "Llama, Qwen", "rope", FIN),
        ]
        tiles = VGroup()
        for name, who, model, ic, c in items:
            fr = card(2.55, 2.95, c, 0.09, radius=0.18)
            icn = icon(ic, c).scale(1.1).move_to(fr.get_top() + DOWN * 0.95)
            nm = T(name, 20, c, bold=True).move_to(fr.get_center() + DOWN * 0.45)
            w = T(who, 14, TOKEN).move_to(fr.get_center() + DOWN * 0.9)
            m = T(model, 14, FADE).move_to(fr.get_center() + DOWN * 1.2)
            for t in (nm, w, m):
                if t.width > 2.3:
                    t.scale_to_fit_width(2.3)
            tiles.add(VGroup(fr, icn, nm, w, m))
        tiles.arrange_in_grid(2, 5, buff=0.17).move_to(DOWN * 0.45)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.3, scale=0.94) for t in tiles], lag_ratio=0.12), run_time=3.2)
        self.pause(1.6)
        self.clear_all()


class S14Frontier(Deck):
    def construct(self):
        self.title("The trade-off, and what 'universal' means", MAMBA2)
        ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=7.2, y_length=4.2, tips=False,
                  axis_config={"color": FADE, "include_ticks": False, "stroke_width": 2}).move_to(LEFT * 1.6 + DOWN * 0.35)
        xl = T("memory kept per layer", 20, FADE).next_to(ax, DOWN, buff=0.15)
        yl = T("recall of exact details", 20, FADE).rotate(PI / 2).next_to(ax, LEFT, buff=0.15)
        front = ax.plot(lambda x: 9.4 * (1 - np.exp(-0.38 * x)) / (1 - np.exp(-3.8)), x_range=[0.3, 9.8], color=FADE, stroke_width=3)
        front = DashedVMobject(front, num_dashes=40)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(front))
        pts = [("Mamba", 1.3, 1.6, MAMBA, "r"), ("Mamba-2", 3.1, 3.6, MAMBA2, "r"), ("Transformer", 9.2, 9.1, ATTN, "l")]
        objs = {}
        for name, x, y, c, side in pts:
            d = Dot(ax.c2p(x, y), radius=0.15, color=c)
            lab = T(name, 22, c, bold=True).next_to(d, RIGHT if side == "r" else LEFT, buff=0.2)
            objs[name] = VGroup(glow(Circle(radius=0.15).move_to(d), c, 3, 2, 3), d, lab)
            self.play(FadeIn(objs[name], scale=0.7), run_time=0.7)
        hyb = Dot(ax.c2p(4.7, 7.7), radius=0.17, color=V_COL)
        hlab = VGroup(T("hybrid", 22, V_COL, bold=True), T("Mamba-2 + a few attention layers", 15, FADE)).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        hlab.next_to(hyb, RIGHT, buff=0.25)
        harr = Arrow(ax.c2p(3.1, 3.6), ax.c2p(4.6, 7.4), buff=0.22, color=V_COL, stroke_width=4)
        self.play(GrowArrow(harr), FadeIn(hyb, scale=0.5), FadeIn(hlab))
        # MoE: capacity without compute - its own thermometer
        th = RoundedRectangle(width=0.4, height=3.6, corner_radius=0.2, stroke_color=MOE, stroke_width=2.5).move_to(RIGHT * 5.4 + DOWN * 0.3)
        lvl = RoundedRectangle(width=0.32, height=0.5, corner_radius=0.16, stroke_width=0, fill_color=MOE, fill_opacity=0.9).move_to(th.get_bottom() + UP * 0.28)
        t_lab = T("MoE: more capacity\nsame compute", 20, MOE, bold=True).next_to(th, UP, buff=0.3)
        self.play(FadeIn(th), FadeIn(lvl), FadeIn(t_lab))
        self.play(lvl.animate.stretch_to_fit_height(3.2).move_to(th.get_bottom() + UP * 1.75), run_time=1.5)
        self.pause()

        # universality, in two senses
        self.play(FadeOut(VGroup(ax, xl, yl, front, hyb, hlab, harr, th, lvl, t_lab, *objs.values())))
        cl = card(6.3, 4.6, ATTN, 0.07).move_to(LEFT * 3.7 + DOWN * 0.3)
        cr = card(6.3, 4.6, MAMBA2, 0.07).move_to(RIGHT * 3.7 + DOWN * 0.3)
        tl = T("Can it compute anything?", 24, ATTN, bold=True).move_to(cl.get_top() + DOWN * 0.45)
        tr = T("Do they learn the same circuits?", 24, MAMBA2, bold=True).move_to(cr.get_top() + DOWN * 0.45)
        self.play(FadeIn(cl), FadeIn(cr), FadeIn(tl), FadeIn(tr))
        # left: staircase approximations converge on a curve
        f = lambda x: 0.9 * np.sin(1.6 * x) + 0.4 * np.sin(3.1 * x + 1)
        xs0 = LEFT * 5.9
        def approx(n):
            xs = np.linspace(0, 4.6, n)
            pts_ = [xs0 + RIGHT * x + UP * (f(x) * 0.7 - 0.55) for x in xs]
            return VMobject(stroke_color=ATTN, stroke_width=5).set_points_as_corners(pts_)
        target = VMobject(stroke_color=TOKEN, stroke_width=2).set_points_as_corners(
            [xs0 + RIGHT * x + UP * (f(x) * 0.7 - 0.55) for x in np.linspace(0, 4.6, 200)])
        a = approx(5)
        self.play(Create(DashedVMobject(target, num_dashes=60)), Create(a))
        for n in (9, 18, 60):
            self.play(Transform(a, approx(n)), run_time=0.7)
        cap_l = T("Yun et al. 2020", 18, FADE).move_to(cl.get_bottom() + UP * 0.35)
        self.play(FadeIn(cap_l))
        # right: two constellations align
        base = np.array([[-1.5, 0.9], [-0.6, 1.4], [0.3, 0.6], [1.2, 1.2], [0.9, -0.5], [-0.4, -0.7], [-1.3, -0.4]])
        cx = cr.get_center() + DOWN * 0.3
        A = VGroup(*[Dot(cx + np.array([x, y, 0]) * 0.9, radius=0.1, color=ATTN) for x, y in base])
        ang = 1.0
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        B = VGroup(*[Dot(cx + np.append(R @ np.array([x, y]), 0) * 0.9, radius=0.1, color=MAMBA2) for x, y in base])
        self.play(FadeIn(A), FadeIn(B))
        self.play(Rotate(B, -ang, about_point=cx), run_time=1.6)
        cap_r = T("shared features in Transformers and Mamba", 17, FADE).move_to(cr.get_bottom() + UP * 0.35)
        self.play(FadeIn(cap_r), Flash(cx, color=MAMBA2, flash_radius=1.4))
        self.pause(1.4)
        self.clear_all()


class S15Derivation(Deck):
    def construct(self):
        self.title("Deriving Mamba-2's attention form", MAMBA2)
        lh = T("from the notebook", 26, MAMBA2, bold=True).move_to(LEFT * 3.6 + UP * 2.25)
        rh = T("from attention", 26, ATTN, bold=True).move_to(RIGHT * 3.6 + UP * 2.25)
        self.play(FadeIn(lh), FadeIn(rh))

        def eq(tex, color, x, y, w=6.4, size=40):
            m = M(tex, size=size, color=color)
            if m.width > w:
                m.scale_to_fit_width(w)
            return m.move_to([x, y, 0])

        Y = [1.35, 0.1, -1.15]
        L = [eq(r"S_t=a_t\,S_{t-1}+k_t\,v_t^\top", MAMBA2, -3.6, Y[0]),
             eq(r"S_t=\sum_{s\le t}a_{s+1}\cdots a_t\;k_s v_s^\top", MAMBA2, -3.6, Y[1]),
             eq(r"y_t=\sum_{s\le t}\underbrace{a_{s+1}\cdots a_t}_{L_{ts}}\,(q_t^\top k_s)\,v_s", MAMBA2, -3.6, Y[2])]
        R = [eq(r"O=\mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt d}\Big)V", ATTN, 3.6, Y[0]),
             eq(r"O=\big(QK^\top\odot\mathrm{causal}\big)V", ATTN, 3.6, Y[1]),
             eq(r"O=\big(QK^\top\odot L\big)V", ATTN, 3.6, Y[2])]
        body = VGroup(lh, rh)
        for k in range(3):
            body.add(L[k], R[k])
            self.play(FadeIn(L[k], shift=DOWN * 0.2), FadeIn(R[k], shift=DOWN * 0.2), run_time=0.9)
            self.wait(0.5)
        self.pause()

        meet = M(r"Y=\big(L\odot QK^\top\big)V", size=56, color=RESID)
        box = SurroundingRectangle(meet, color=MAMBA2, buff=0.25, corner_radius=0.2, stroke_width=4)
        grp = VGroup(box, meet).move_to(DOWN * 2.75)
        al = Arrow(L[2].get_bottom() + DOWN * 0.25, box.get_left(), buff=0.1, color=MAMBA2, stroke_width=5, path_arc=0.6)
        ar = Arrow(R[2].get_bottom() + DOWN * 0.25, box.get_right(), buff=0.1, color=ATTN, stroke_width=5, path_arc=-0.6)
        halo = glow(box, MAMBA2, 3)
        self.play(Create(al), Create(ar), run_time=0.9)
        self.play(FadeIn(grp, scale=0.9), FadeIn(halo))
        self.pause()

        # numeric check with the tested reference code
        rng = np.random.default_rng(11)
        a = np.array([0.9, 0.8, 0.5, 0.75]); K = rng.normal(size=(4, 2)); Q = rng.normal(size=(4, 2)); V = rng.normal(size=(4, 1))
        y1 = ssd_recurrent(a, K, Q, V)[:, 0]; y2 = ssd_matrix(a, K, Q, V)[:, 0]
        fmt = lambda y: "   ".join(f"{v:+.2f}" for v in y)
        self.play(FadeOut(body), FadeOut(al), FadeOut(ar), FadeOut(halo), grp.animate.move_to(UP * 1.2), run_time=1.0)
        c1 = VGroup(T("recurrence", 24, MAMBA2, bold=True), T(fmt(y1), 28, TOKEN)).arrange(DOWN, buff=0.2).move_to(LEFT * 3.7 + DOWN * 1.1)
        c2 = VGroup(T("matrix form", 24, ATTN, bold=True), T(fmt(y2), 28, TOKEN)).arrange(DOWN, buff=0.2).move_to(RIGHT * 3.7 + DOWN * 1.1)
        eqs = M("=", size=70, color=MAMBA2).move_to(DOWN * 1.3)
        self.play(FadeIn(c1, shift=UP * 0.2), FadeIn(c2, shift=UP * 0.2), FadeIn(eqs, scale=1.5))
        self.pause(1.5)
        self.clear_all()


class S16Interp(Deck):
    def construct(self):
        self.title("Looking inside: how we test it", RESID, accent=K_COL)
        specs = [("logit lens", "read the stream early", ATTN), ("patching", "swap a part, see what breaks", MOE),
                 ("induction heads", "copy what came next", Q_COL), ("sparse features", "split into clean parts", K_COL),
                 ("Mamba's hidden attention", "the state as an attention map", MAMBA2)]
        tiles = VGroup()
        for name, tag, c in specs:
            fr = card(2.5, 3.5, c, 0.09, radius=0.18)
            nm = T(name, 19, c, bold=True).move_to(fr.get_center() + DOWN * 0.85)
            tg = T(tag, 14, FADE).move_to(fr.get_center() + DOWN * 1.3)
            for t in (nm, tg):
                if t.width > 2.3:
                    t.scale_to_fit_width(2.3)
            tiles.add(VGroup(fr, nm, tg))
        tiles.arrange(RIGHT, buff=0.2).move_to(UP * 0.0 + DOWN * 0.2)
        centre = lambda k: tiles[k][0].get_center() + UP * 0.85

        # 1 logit lens: bars sharpen over layers
        bars = lambda hs: VGroup(*[Rectangle(width=0.22, height=h, stroke_width=0, fill_color=ATTN, fill_opacity=0.9) for h in hs]).arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
        lens = bars([0.5, 0.45, 0.4, 0.5, 0.45]).move_to(centre(0))
        # 2 patching: swap block between two rows
        r1 = VGroup(*[Square(0.3, stroke_color=MOE, stroke_width=2, fill_color=MOE, fill_opacity=0.5) for _ in range(3)]).arrange(RIGHT, buff=0.06)
        r2 = VGroup(*[Square(0.3, stroke_color=PROB, stroke_width=2, fill_color=PROB, fill_opacity=0.5) for _ in range(3)]).arrange(RIGHT, buff=0.06)
        patch = VGroup(r1, r2).arrange(DOWN, buff=0.35).move_to(centre(1))
        # 3 induction: A B ... A -> B
        seq = VGroup(*[chip(t, TOKEN, 16) for t in ("A", "B", "…", "A")]).arrange(RIGHT, buff=0.08).scale(0.8).move_to(centre(2))
        # 4 SAE: dense -> sparse
        dense = vec(5, K_COL, 0.2, vals=[0.8, 0.6, 0.9, 0.7, 0.5]).move_to(centre(3) + LEFT * 0.55)
        sparse = VGroup(*[Square(0.2, stroke_color=K_COL, stroke_width=1.2, fill_color=K_COL, fill_opacity=0.9 if i in (1, 5) else 0.05) for i in range(7)]).arrange(DOWN, buff=0.02).move_to(centre(3) + RIGHT * 0.55)
        sae = VGroup(dense, Arrow(LEFT * 0.2, RIGHT * 0.2, buff=0, stroke_width=3, color=FADE).move_to(centre(3)), sparse)
        # 5 hidden attention: lower-triangular heat
        ha = heat(np.tril(np.random.default_rng(5).uniform(0.1, 1, (5, 5))), 0.26, pos=MAMBA2).move_to(centre(4))

        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.3) for t in tiles], lag_ratio=0.12), run_time=1.6)
        lens.scale(1.4); patch.scale(1.3); ha.scale(1.3)
        self.play(FadeIn(lens), FadeIn(patch), FadeIn(seq), FadeIn(sae), FadeIn(ha))
        ind_arc = ArcBetweenPoints(seq[3].get_top(), seq[1].get_top(), angle=-1.0, stroke_color=Q_COL, stroke_width=4)
        self.play(Transform(lens, bars([0.2, 0.25, 1.2, 0.3, 0.2]).move_to(centre(0)).align_to(lens, DOWN)),
                  r1[1].animate.set_fill(PROB, 0.9), r2[1].animate.set_fill(MOE, 0.9),
                  Create(ind_arc),
                  sparse[1].animate.scale(1.25), sparse[5].animate.scale(1.25), run_time=1.8)
        self.pause(1.2)

        # closing: three paths merge into one residual vector
        self.play(FadeOut(tiles), FadeOut(VGroup(lens, patch, seq, sae, ha, ind_arc)))
        start = [LEFT * 5.5 + UP * 1.4, LEFT * 5.5, LEFT * 5.5 + DOWN * 1.4]
        end = RIGHT * 1.6
        paths = VGroup(*[CubicBezier(s + RIGHT * 0.2, s + RIGHT * 3, end + LEFT * 2.2, end, stroke_color=c, stroke_width=7)
                         for s, c in zip(start, (ATTN, MAMBA, MAMBA2))])
        names = VGroup(*[T(n, 24, c, bold=True).next_to(s, UP, buff=0.12).shift(RIGHT * 0.5) for n, c, s in zip(("Theo", "Manu", "Max"), (ATTN, MAMBA, MAMBA2), start)])
        res = arrow(end, end + RIGHT * 3.6, RESID, 12, 0.35)
        self.play(FadeIn(names), LaggedStart(*[Create(p) for p in paths], lag_ratio=0.2), run_time=1.8)
        self.play(GrowArrow(res), FadeIn(glow(Line(end, end + RIGHT * 3.4), RESID, 3)))
        fin = T("Different memories.  Often similar computations.", 38, RESID, bold=True).to_edge(DOWN, buff=0.9)
        self.play(Write(fin), run_time=1.6)
        self.pause(2.0)
        self.clear_all()
