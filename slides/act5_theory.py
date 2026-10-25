"""Act V — Theory, the tunnel, and mechanisms (slides 22–26)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *
from math_contracts.test_ssd_equivalence import ssd_recurrent, ssd_matrix, causal_linear_attention

ACT = "Act V · Theory, the tunnel, and mechanisms"


class S22Frontier(Deck):
    def construct(self):
        self.header("22", "The efficiency–expressivity frontier", ACT, "SCHEMATIC")
        ax = Axes(x_range=[0, 10, 10], y_range=[0, 10, 10], x_length=6.2, y_length=4.6, tips=True,
                  axis_config={"stroke_color": FADE, "include_ticks": False}).shift(LEFT * 3.4 + DOWN * 0.4)
        xl = T("memory kept as context grows: constant → linear", 16, FADE).next_to(ax, DOWN, buff=0.15)
        yl = T("direct addressability of past tokens", 16, FADE).rotate(PI / 2).next_to(ax, LEFT, buff=0.15)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        pts = [("full attention", 9.3, 9.5, ATTN), ("GQA / MLA", 7.4, 7.9, ATTN), ("sliding window", 4.5, 5.0, ATTN),
               ("Mamba-1 / Mamba-2", 1.2, 2.4, MAMBA2), ("hybrids", 3.0, 7.6, MOE)]
        dots = VGroup()
        for name, x, y, c in pts:
            d = Dot(ax.c2p(x, y), radius=0.12, color=c)
            l = T(name, 18, c).next_to(d, RIGHT if x < 7 else (LEFT if y > 9 else DOWN), buff=0.12)
            dots.add(VGroup(d, l))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.3), run_time=2)
        moe = Arrow(ax.c2p(5, 1.5), ax.c2p(7.8, 4.2), color=MOE, buff=0, stroke_width=5)
        moel = T("MoE: an orthogonal axis\n(stored capacity vs active compute)", 16, MOE).next_to(moe.get_start(), DOWN, buff=0.1)
        self.play(GrowArrow(moe), FadeIn(moel))
        self.pause()
        cards = [("recall needs state", "Zoology / MQAR, Based (Arora et al. 2024)"),
                 ("copying needs a growing state", "Repeat After Me (Jelassi et al. 2024)"),
                 ("state tracking is hard for both (TC⁰)", "Illusion of State (Merrill et al. 2024)"),
                 ("negative eigenvalues → parity", "Grazzi et al. 2025"),
                 ("hybrids win empirically", "Waleffe et al. 2024 (8B Mamba-2-hybrid)"),
                 ("hybrid separations, task-specific", "Cooper et al. 2026 (verify)")]
        ev = VGroup()
        for claim, src in cards:
            ev.add(VGroup(T(claim, 18, RESID), T(src, 14, FADE)).arrange(DOWN, aligned_edge=LEFT, buff=0.05))
        ev.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(RIGHT, buff=0.4).shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(e, shift=LEFT * 0.3) for e in ev], lag_ratio=0.35), run_time=3)
        self.pause()
        self.clear_all(0.5)

        self.header("22", "Sequence-mixer cost in T (everything else fixed)", ACT, "PROPERTY")
        rows = [("mixer", "prefill", "decode / token", "memory", "addressability"),
                ("dense softmax attention", "T²", "T", "T", "explicit"),
                ("GQA / MQA / MLA", "T²", "T", "T (smaller)", "explicit"),
                ("sliding window W", "TW", "W", "W", "inside window"),
                ("Mamba-1 / Mamba-2", "T", "1", "1", "implicit (state)"),
                ("hybrid", "mix", "attn T · SSM 1", "KV for attn only", "mixed")]
        cols_x = [-5.2, -1.4, 0.7, 2.9, 5.0]
        tab = VGroup()
        for i, r in enumerate(rows):
            line = VGroup()
            for j, cell in enumerate(r):
                c = RESID if i else FADE
                if i and j == 0:
                    c = MAMBA2 if "Mamba" in cell else (MOE if cell == "hybrid" else ATTN)
                t = T(cell, 20 if i else 18, c)
                t.move_to([cols_x[j], 1.6 - 0.65 * i, 0], aligned_edge=LEFT if j == 0 else ORIGIN)
                if j == 0:
                    t.set_x(cols_x[0] - 1.4, LEFT)
                line.add(t)
            tab.add(line)
        rule = Line(LEFT * 6.6, RIGHT * 6.6, color=FADE).move_to([0, 1.6 - 0.33, 0])
        self.play(FadeIn(tab[0]), Create(rule))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in tab[1:]], lag_ratio=0.3), run_time=2)
        foot = T("\"1\" = constant in sequence length, not free · don't say \"hybrids dominate\": results are task-specific", 17, FADE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(foot))
        self.pause(1.5)
        self.clear_all()


class S23Universality(Deck):
    def construct(self):
        self.header("23", "Towards universality: two meanings of one word", ACT, "SCHEMATIC")
        div = DashedLine(UP * 2.6, DOWN * 3.3, color=FADE)
        self.play(Create(div))
        # (A) expressive universality: a blanket covering a function landscape
        la = T("A · expressive universality", 24, ATTN).move_to(LEFT * 3.5 + UP * 2.2)
        ax = Axes(x_range=[0, 6, 1], y_range=[-1.5, 1.5, 1], x_length=5, y_length=2.4, tips=False, axis_config={"stroke_color": FADE, "include_ticks": False}).move_to(LEFT * 3.5 + UP * 0.4)
        f = ax.plot(lambda x: np.sin(1.7 * x) * 0.9 + 0.3 * np.cos(4.1 * x), color=RESID)
        approx = ax.plot(lambda x: np.sin(1.7 * x) * 0.9 + 0.3 * np.cos(4.1 * x) + 0.08 * np.sin(13 * x), color=ATTN, stroke_width=6, stroke_opacity=0.6)
        self.play(FadeIn(la), Create(ax), Create(f))
        self.play(Create(approx), run_time=1.5)
        a_txt = VGroup(T("Transformers approximate continuous seq2seq maps\non compact domains (Yun et al. 2020)", 16),
                       T("RNNs (Siegelmann & Sontag 1995) and Transformers\n(Pérez et al. 2021) are Turing-complete — idealised precision", 16),
                       T("not covered: fixed size, finite precision, learnability", 16, FADE)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(ax, DOWN, buff=0.3)
        a_txt.scale_to_fit_width(min(a_txt.width, 6.0)).set_x(-3.55)
        self.play(FadeIn(a_txt, lag_ratio=0.3), run_time=1.5)
        self.pause()

        # (B) mechanistic universality: constellations align under a rotation
        lb = T("B · mechanistic universality", 24, MAMBA2).move_to(RIGHT * 3.5 + UP * 2.2)
        self.play(FadeIn(lb))
        rng = np.random.default_rng(7)
        base = rng.normal(size=(7, 2)) * 0.8
        center = RIGHT * 3.5 + UP * 0.4
        cols = [ATTN, MAMBA, MAMBA2]
        clouds = VGroup()
        rots = [0.0, 1.1, -0.9]
        for c, ang in zip(cols, rots):
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            pts = base @ R.T + rng.normal(scale=0.08, size=base.shape)
            g = VGroup(*[Dot([p[0], p[1], 0], radius=0.07, color=c) for p in pts])
            g.add(*[Line(g[i].get_center(), g[i + 1].get_center(), color=c, stroke_width=1.5, stroke_opacity=0.6) for i in range(6)])
            clouds.add(g.move_to(center))
        self.play(*[FadeIn(cl) for cl in clouds])
        self.play(Rotate(clouds[1], -rots[1], about_point=center), Rotate(clouds[2], -rots[2], about_point=center), run_time=2)
        b_txt = VGroup(T("do different nets learn the same features & circuits?", 16),
                       T("Olah et al. 2020 · Huh et al. 2024 (Platonic Representation)", 15, FADE),
                       T("Wang et al. ICLR 2025: similar SAE features and induction\ncircuits in Transformers and Mamba — Mamba's off-by-one motif", 16, MAMBA2),
                       T("feature similarity ≠ identical mechanism", 16, MAMBA)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(ax, DOWN, buff=0.3)
        b_txt.scale_to_fit_width(min(b_txt.width, 6.0)).set_x(3.55)
        self.play(FadeIn(b_txt, lag_ratio=0.3), run_time=1.5)
        self.pause()
        q = T("Same computational variable, different memories, same causal consequences?", 24, MOE).to_edge(DOWN, buff=0.15)
        self.play(FadeOut(a_txt[2]), FadeOut(b_txt[1]), Write(q))
        self.pause(1.5)
        self.clear_all()


class S24Tunnel(Deck):
    def construct(self):
        self.header("24", "The tunnel: the attention form of Mamba-2, from both sides", ACT, "IDENTITY")
        mountain = Polygon([-7, -0.9, 0], [-3.5, 1.9, 0], [-1.2, 1.1, 0], [0.6, 2.4, 0], [3.4, 1.0, 0], [7, -0.9, 0],
                           stroke_color=FADE, fill_color="#1A1F2B", fill_opacity=1)
        self.play(FadeIn(mountain))
        y_t = -0.2
        left_bore = Rectangle(width=0.01, height=0.35, stroke_width=0, fill_color=MAMBA2, fill_opacity=0.9).move_to([-6.6, y_t, 0], aligned_edge=LEFT)
        right_bore = Rectangle(width=0.01, height=0.35, stroke_width=0, fill_color=ATTN, fill_opacity=0.9).move_to([6.6, y_t, 0], aligned_edge=RIGHT)
        ll = T("Max digs from the recurrence (Act III)", 18, MAMBA2).move_to([-3.8, y_t - 0.55, 0])
        rl = T("Theo digs from attention", 18, ATTN).move_to([3.8, y_t - 0.55, 0])
        self.add(left_bore, right_bore)
        self.play(FadeIn(ll), FadeIn(rl))
        self.play(left_bore.animate.stretch_to_fit_width(5.6, about_edge=LEFT), run_time=1.5)
        left_eq = M(r"S_t=a_tS_{t-1}+k_tv_t^\top\ \Rightarrow\ y_t=\sum_s L_{ts}(q_t^\top k_s)v_s", size=28, color=MAMBA2).move_to([-3.6, 0.7, 0])
        self.play(Write(left_eq))
        self.pause()

        panel_y = -2.2
        steps = [
            (r"o_t=\frac{\sum_{s\le t}\exp(q_t^\top k_s)\,v_s}{\sum_{s\le t}\exp(q_t^\top k_s)}", "1 · start: causal softmax attention"),
            (r"\exp(q^\top k)\to\phi(q)^\top\phi(k),\ \phi=\mathrm{id},\ \text{drop the normaliser}\ \Rightarrow\ O=\mathrm{tril}(QK^\top)V", "2 · kernel trick (Katharopoulos et al. 2020)"),
            (r"o_t^\top=\sum_{s\le t}q_t^\top k_s\,v_s^\top=q_t^\top\underbrace{\Big(\sum_{s\le t}k_sv_s^\top\Big)}_{S_t\in\mathbb R^{N\times P}}", "3 · associativity: move the brackets — the T×T matrix is never needed"),
            (r"S_t=S_{t-1}+k_tv_t^\top", "4 · linear attention is an RNN with a matrix state (but nothing fades)"),
            (r"S_t=a_tS_{t-1}+k_tv_t^\top\ \Longleftrightarrow\ Y=(L\odot QK^\top)V", "5 · add forgetting: the mask of ones becomes the survival mask L"),
        ]
        cur_eq = cur_lab = None
        widths = [1.5, 3.0, 4.5, 6.0, 7.6]  # right bore meets the left one at x = -1.0
        for (tex, lab), w in zip(steps, widths):
            e = M(tex, size=32).move_to([0, panel_y, 0])
            if e.width > 12.5:
                e.scale_to_fit_width(12.5)
            l = T(lab, 18, ATTN).next_to(e, UP, buff=0.2)
            if cur_eq is None:
                self.play(Write(e), FadeIn(l))
            else:
                self.play(FadeOut(cur_eq, shift=UP * 0.2), FadeOut(cur_lab), FadeIn(e, shift=UP * 0.2), FadeIn(l), run_time=0.9)
            self.play(right_bore.animate.stretch_to_fit_width(w, about_edge=RIGHT), run_time=0.8)
            cur_eq, cur_lab = e, l
            self.pause(0.6)
        flash = Flash(np.array([-1.0, y_t, 0]), color=MOE, line_length=0.5, num_lines=14, flash_radius=0.5)
        self.play(flash)
        boxed = M(r"Y=(L\odot QK^\top)\,V", size=46, color=MOE).move_to([-0.6, 1.0, 0])
        box = SurroundingRectangle(boxed, color=MOE, buff=0.15)
        self.play(ReplacementTransform(left_eq, boxed), Create(box))
        met = T("the bores meet: same identity from both sides  ·  test_recurrent_equals_matrix", 18, MOE).next_to(box, UP, buff=0.15)
        self.play(FadeIn(met))
        self.pause()

        # live numbers from the tested reference implementations
        self.play(FadeOut(VGroup(cur_eq, cur_lab)))
        rng = np.random.default_rng(24)
        Tn, Nn, Pn = 6, 2, 3
        a = rng.uniform(0.5, 0.95, Tn); K = rng.normal(size=(Tn, Nn)); Q = rng.normal(size=(Tn, Nn)); V = rng.normal(size=(Tn, Pn))
        yr = ssd_recurrent(a, K, Q, V)[-1]; ym = ssd_matrix(a, K, Q, V)[-1]
        fmt = lambda v: "[" + ", ".join(f"{x:+.5f}" for x in v) + "]"
        nums = VGroup(T("toy T=6, N=2, P=3 · last output", 16, FADE), T("route 1 (recurrence)   " + fmt(yr), 20, MAMBA2),
                      T("route 2 (matrix)          " + fmt(ym), 20, ATTN), T(f"max |difference| = {np.abs(yr - ym).max():.1e}", 16, FADE)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([0, panel_y, 0])
        self.play(FadeIn(nums, lag_ratio=0.3))
        self.pause()
        self.clear_all(0.5)

        self.header("24", "The family portrait: same template, different masks and writes", ACT, "PROPERTY")
        fam = [("Linear attention", "Katharopoulos 2020", "S = S + k vᵀ", "L = causal ones", RESID),
               ("RetNet", "Sun 2023", "S = γ S + k vᵀ", "L = γ^(t−s), fixed", RESID),
               ("Mamba-2 / SSD", "Dao & Gu 2024", "S = aₜ S + k vᵀ", "L 1-semiseparable, input-dependent", MAMBA2),
               ("GLA", "Yang 2024", "S = diag(αₜ) S + k vᵀ", "per-key-dimension decay", RESID),
               ("Mamba-1", "Gu & Dao 2023", "diagonal Āₜ per channel", "tensor-valued mask", MAMBA),
               ("DeltaNet", "Schlag 2021 · Yang 2024", "S = (I − βkkᵀ) S + β k vᵀ", "overwrite instead of add", RESID),
               ("Gated DeltaNet", "Yang 2025", "decay + delta rule", "recent hybrids (verify)", RESID),
               ("Softmax attention", "Vaswani 2017", "state = the whole KV cache", "exp = infinite-dim features: no fold", ATTN)]
        xs = [-6.6, -3.9, -1.5, 2.4]
        rows = VGroup()
        for i, (m, who, upd, mask, c) in enumerate(fam):
            y = 2.0 - 0.6 * i
            r = VGroup(T(m, 20, c).move_to([xs[0], y, 0], aligned_edge=LEFT), T(who, 15, FADE).move_to([xs[1], y, 0], aligned_edge=LEFT),
                       T(upd, 18).move_to([xs[2], y, 0], aligned_edge=LEFT), T(mask, 16, FADE).move_to([xs[3], y, 0], aligned_edge=LEFT))
            rows.add(r)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in rows], lag_ratio=0.25), run_time=3)
        dont = T("Mamba-2 equals a masked, unnormalised, linear attention-shaped operator — never a softmax, never the cards", 18, MAMBA).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(dont))
        self.pause(1.5)
        self.clear_all()


class S25Interpretability(Deck):
    def construct(self):
        self.header("25", "Interpretability designs: from pictures to interventions", ACT, "SCHEMATIC")
        tiles = []
        # 1 logit lens
        bars = VGroup(*[Rectangle(width=0.18, height=0.05 + 0.9 * p, stroke_width=0, fill_color=Q_COL, fill_opacity=0.85) for p in (0.05, 0.1, 0.2, 0.45, 0.7, 0.85)]).arrange(RIGHT, buff=0.06, aligned_edge=DOWN)
        tiles.append(("logit / tuned lens", "nostalgebraist 2020 · Belrose 2023", bars))
        # 2 induction
        seq = VGroup(*[token_square(s, size=0.4, fs=14) for s in ("A", "B", "…", "A", "?")]).arrange(RIGHT, buff=0.08)
        arr = CurvedArrow(seq[1].get_top(), seq[4].get_top(), angle=-PI / 2.5, color=ATTN, stroke_width=3)
        tiles.append(("induction heads", "Elhage 2021 · Olsson 2022", VGroup(seq, arr)))
        # 3 implicit attention of Mamba-2
        rng = np.random.default_rng(9)
        from math_contracts.test_ssd_equivalence import decay_mask
        Mi = decay_mask(rng.uniform(0.6, 0.95, 8)) * (rng.normal(size=(8, 2)) @ rng.normal(size=(8, 2)).T)
        tiles.append(("implicit attention of SSMs", "Ali, Zimerman & Wolf 2024", heatmap(Mi, cell=0.13)))
        # 4 SAE features
        feats = VGroup(*[Rectangle(width=0.12, height=0.05 + (0.9 if i in (3, 11) else 0.05 * rng.random()), stroke_width=0, fill_color=MOE, fill_opacity=0.9) for i in range(16)]).arrange(RIGHT, buff=0.03, aligned_edge=DOWN)
        tiles.append(("SAEs · transcoders · graphs", "Bricken 2023 · Templeton 2024 · Ameisen 2025", feats))
        # 5 DAS
        das = M(r"r\leftarrow r_{\rm base}+UU^\top(r_{\rm src}-r_{\rm base})", size=24)
        tiles.append(("DAS subspace swap", "Geiger et al. 2024", das))
        # 6 causal tracing / patching
        two = VGroup(Line(LEFT, RIGHT, color=MAMBA2), Line(LEFT, RIGHT, color=MAMBA).shift(DOWN * 0.45))
        cable = CubicBezier(two[1].get_center() + LEFT * 0.2, two[1].get_center() + UP * 0.3, two[0].get_center() + DOWN * 0.3, two[0].get_center() + RIGHT * 0.2, color=MOE)
        tiles.append(("activation / path patching", "Meng 2022 · Wang 2022 · Sharma 2024 (Mamba)", VGroup(two, cable)))
        grid = VGroup()
        for name, who, icon in tiles:
            icon = icon.copy()
            if icon.height > 1.0:
                icon.scale_to_fit_height(1.0)
            if icon.width > 3.2:
                icon.scale_to_fit_width(3.2)
            card = VGroup(T(name, 20), icon, T(who, 13, FADE)).arrange(DOWN, buff=0.15)
            fr = RoundedRectangle(width=3.9, height=2.3, corner_radius=0.12, stroke_color="#2A3040").move_to(card)
            if card[2].width > 3.7:
                card[2].scale_to_fit_width(3.7)
            grid.add(VGroup(fr, card))
        grid.arrange_in_grid(2, 3, buff=(0.25, 0.25)).shift(DOWN * 0.4)
        for g in grid:
            self.play(FadeIn(g, shift=UP * 0.2), run_time=0.6)
        self.pause()
        self.clear_all(0.5)

        # the hybrid switchboard
        self.header("25", "The hybrid experiment: where did the answer come from?", ACT, "SCHEMATIC")
        clean = VGroup(*[token_square(w, fs=16) for w in ("Leo", "bridge", "…", "Where is Leo's key?")]).arrange(RIGHT, buff=0.3).shift(UP * 1.6 + LEFT * 1.0)
        corrupt = VGroup(*[token_square(w, color=MAMBA, fs=16) for w in ("Leo", "cupboard", "…", "Where is Leo's key?")]).arrange(RIGHT, buff=0.3).shift(DOWN * 0.1 + LEFT * 1.0)
        cl = T("clean", 18, MAMBA2).next_to(clean, LEFT, buff=0.3); col = T("corrupt", 18, MAMBA).next_to(corrupt, LEFT, buff=0.3)
        self.play(FadeIn(clean), FadeIn(corrupt), FadeIn(cl), FadeIn(col))
        meter_c = RIGHT * 5.2 + UP * 0.6
        arc = Arc(radius=1.1, start_angle=PI, angle=-PI, color=FADE, stroke_width=5).move_arc_center_to(meter_c)
        val = ValueTracker(-0.8)
        needle = always_redraw(lambda: Line(meter_c, meter_c + 1.0 * np.array([np.cos(PI / 2 - val.get_value() * PI / 2), np.sin(PI / 2 - val.get_value() * PI / 2), 0]), color=MOE, stroke_width=6))
        ml = T("answer margin\nlogit(bridge) − logit(cupboard)", 16, MOE).next_to(arc, DOWN, buff=0.15)
        hyp = T("HYPOTHESIS TO TEST · mock values", 16, FADE).next_to(ml, DOWN, buff=0.1)
        self.play(Create(arc), FadeIn(ml), FadeIn(hyp))
        self.add(needle)
        patches = [("recurrent state at the query", 0.2, "state already holds the binding?"),
                   ("query residual", 0.5, "answer assembled at the query?"),
                   ("correct source K/V", 0.85, "attention reads the source directly?"),
                   ("wrong source K/V (control)", -0.7, "specificity check"),
                   ("knock out source → query edge", -0.5, "is that path necessary?")]
        info = None
        for name, v, meaning in patches:
            cab = CubicBezier(corrupt[1].get_bottom() if "source" in name else corrupt[3].get_bottom(),
                              corrupt[3].get_bottom() + DOWN * 0.8, clean[3].get_top() + UP * 0.8, clean[3].get_top() if "source" not in name else clean[1].get_top(), color=MOE, stroke_width=3)
            t = VGroup(T("patch: " + name, 20, MOE), T("→ " + meaning, 18)).arrange(DOWN, aligned_edge=LEFT, buff=0.08).to_edge(DOWN, buff=1.3).set_x(-1.5)
            self.play(Create(cab), FadeIn(t) if info is None else ReplacementTransform(info, t), val.animate.set_value(v), run_time=1.0)
            self.wait(0.6)
            self.play(FadeOut(cab), run_time=0.3)
            info = t
        tree = T("behaviour solved? → variable decodable? → intervention flips the answer? → via which path? → replicates?", 17, RESID).to_edge(DOWN, buff=0.55)
        tri = T("decodable ≠ used · probe accuracy, interchange accuracy and answer margin are three different numbers", 16, FADE).next_to(tree, DOWN, buff=0.1)
        self.play(FadeOut(info), FadeIn(tree), FadeIn(tri))
        self.pause(1.5)
        needle.clear_updaters()
        self.clear_all()


class S26Closing(Deck):
    def construct(self):
        theo = brother("Theo", ATTN, "cards"); manu = brother("Manu", MAMBA, "notebook"); maxx = brother("Max", MAMBA2, "ledger")
        cast = VGroup(theo, manu, maxx).arrange(RIGHT, buff=2.6).shift(DOWN * 1.6)
        recap = VGroup(*[T(s, 18, c) for s, c in (("card wall", ATTN), ("notebook", MAMBA), ("matrix identity", MAMBA2), ("clinic", MOE), ("belt", RESID), ("tunnel", MOE))]).arrange(RIGHT, buff=0.5).to_edge(UP, buff=0.5)
        dots = VGroup(*[T("→", 18, FADE).move_to((recap[i].get_right() + recap[i + 1].get_left()) / 2) for i in range(len(recap) - 1)])
        self.play(LaggedStart(*[FadeIn(r) for r in recap], lag_ratio=0.2), FadeIn(dots))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in cast], lag_ratio=0.25))
        target = UP * 0.9
        belt = Line(DOWN * 0.4, UP * 2.0, color=RESID, stroke_width=3)
        arrows = VGroup(*[Arrow(b[0].get_top(), target, buff=0.2, color=c, stroke_width=4) for b, c in zip(cast, (ATTN, MAMBA, MAMBA2))])
        clinic = Arrow(RIGHT * 6 + UP * 0.2, target, buff=0.2, color=MOE, stroke_width=4)
        cl = T("clinic", 18, MOE).next_to(clinic.get_start(), DOWN, buff=0.1)
        self.play(Create(belt))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2), GrowArrow(clinic), FadeIn(cl))
        res = vec_arrow(DOWN * 0.4, UP * 1.6 + RIGHT * 0.4, RESID, sw=7)
        self.play(GrowArrow(res))
        self.pause()
        final = T("Different memories. Sometimes similar computations. Causality decides.", 32)
        final.scale_to_fit_width(12.6).to_edge(DOWN, buff=0.4)
        self.play(Write(final), run_time=2)
        self.pause(2.5)
        self.clear_all(1.0)
