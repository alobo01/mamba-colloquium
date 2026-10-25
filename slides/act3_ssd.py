"""Act III — Max's mathematics: SSD in detail (slides 12–16).

Every number shown on screen comes from math_contracts/test_ssd_equivalence.py.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *
from math_contracts.test_ssd_equivalence import decay_mask, ssd_recurrent, ssd_chunked, ssd_matrix

ACT = "Act III · Max's mathematics"


def sheet(N, P, cell=0.55, data=None, color=MAMBA2):
    data = np.zeros((N, P)) if data is None else data
    vmax = max(1e-9, np.abs(data).max())
    g = VGroup()
    for n in range(N):
        for p in range(P):
            v = data[n, p] / vmax
            g.add(Square(cell, stroke_color=color, stroke_width=1.5, fill_color=color if v >= 0 else MAMBA,
                         fill_opacity=0.08 + 0.8 * abs(v)).move_to([p * cell, -n * cell, 0]))
    return g.move_to(ORIGIN)


def column(values, color, cell=0.55):
    g = VGroup(*[Square(cell, stroke_color=color, stroke_width=1.5, fill_color=color, fill_opacity=0.15 + 0.7 * abs(v)) for v in values])
    return g.arrange(DOWN, buff=0)


def row(values, color, cell=0.55):
    g = VGroup(*[Square(cell, stroke_color=color, stroke_width=1.5, fill_color=color, fill_opacity=0.15 + 0.7 * abs(v)) for v in values])
    return g.arrange(RIGHT, buff=0)


class S12StateSheet(Deck):
    def construct(self):
        self.header("12", "The state is an N×P sheet: fade, write, read", ACT, "IDENTITY")
        N, P = 4, 6
        rng = np.random.default_rng(1)
        S = rng.uniform(-1, 1, (N, P))
        g = sheet(N, P, data=S).shift(LEFT * 1.8 + UP * 0.2)
        bN = Brace(g, LEFT, color=FADE); bP = Brace(g, UP, color=FADE)
        lN = M("N", size=32, color=FADE).next_to(bN, LEFT); lP = M("P", size=32, color=FADE).next_to(bP, UP)
        self.play(FadeIn(g), GrowFromCenter(bN), GrowFromCenter(bP), FadeIn(lN), FadeIn(lP))
        eq = M(r"S_t", "=", r"a_t", r"S_{t-1}", "+", r"b_t", r"x_t^\top", r",\qquad", r"y_t", "=", r"S_t^\top", r"c_t", size=40).to_edge(DOWN, buff=1.0)
        eq[2].set_color(FADE); eq[5].set_color(K_COL); eq[6].set_color(V_COL); eq[11].set_color(Q_COL)
        self.play(Write(eq))
        self.pause()

        # 1. fade by scalar a_t
        a = 0.6
        S = a * S
        step = T("1 · fade: every cell × aₜ\n(one scalar per head)", 22, FADE).next_to(g, RIGHT, buff=0.8)
        self.play(FadeIn(step), Indicate(eq[2:4], color=FADE))
        self.play(*[sq.animate.set_fill(opacity=sq.get_fill_opacity() * a) for sq in g], run_time=1.0)
        self.pause()

        # 2. rank-1 write b_t x_t^T
        b = np.array([0.9, 0.2, -0.6, 0.4]); x = np.array([0.8, -0.3, 0.5, 1.0, -0.7, 0.2])
        bc = column(b, K_COL).next_to(g, LEFT, buff=1.0)
        xr = row(x, V_COL).next_to(g, UP, buff=0.9)
        self.play(FadeOut(step), FadeOut(VGroup(bN, lN, bP, lP)))
        step = T("2 · write: outer product\nbₜxₜᵀ (rank 1)", 22, K_COL).next_to(g, RIGHT, buff=0.8)
        self.play(FadeIn(bc, shift=RIGHT * 0.3), FadeIn(xr, shift=DOWN * 0.3), FadeIn(step), Indicate(eq[5:7]))
        S = S + np.outer(b, x)
        new = sheet(N, P, data=S).move_to(g)
        self.play(LaggedStart(*[Transform(g[i], new[i]) for i in range(N * P)], lag_ratio=0.03), run_time=1.6)
        self.pause()

        # 3. read y_t = S^T c_t
        c = np.array([0.3, 1.0, -0.4, 0.6])
        cc = column(c, Q_COL).next_to(g, LEFT, buff=1.0)
        y = S.T @ c
        yr = row(y / np.abs(y).max(), RESID).next_to(g, DOWN, buff=0.5)
        ylab = M("y_t", size=32).next_to(yr, LEFT, buff=0.2)
        self.play(FadeOut(step))
        step = T("3 · read: yₜ = Sₜᵀcₜ", 22, Q_COL).next_to(g, RIGHT, buff=0.8)
        self.play(ReplacementTransform(bc, cc), FadeOut(xr), FadeIn(step), Indicate(eq[10:12]))
        sweep = SurroundingRectangle(g[0:P], color=Q_COL, buff=0.02)
        self.play(Create(sweep))
        for n in range(1, N):
            self.play(sweep.animate.move_to(VGroup(*g[n * P:(n + 1) * P])), run_time=0.3)
        self.play(FadeOut(sweep), FadeIn(yr, shift=DOWN * 0.2), FadeIn(ylab))
        self.pause()
        rename = M(r"b_t\to k_t\ (\text{write key})\qquad c_t\to q_t\ (\text{read query})\qquad x_t\to v_t\ (\text{value})", size=32)
        rename.to_edge(DOWN, buff=0.35)
        self.play(eq.animate.shift(UP * 0.35), FadeIn(rename))
        fw = T("a fast-weight memory:\nfade, write with a key,\nread with a query", 20, MAMBA2).next_to(step, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(FadeIn(fw))
        self.pause(1.5)
        self.clear_all()


class S13Unroll(Deck):
    def construct(self):
        self.header("13", "Unroll the notebook: a weighted sum over the past", ACT, "IDENTITY")
        lines = [
            M(r"S_1", "=", r"k_1v_1^\top", size=40),
            M(r"S_2", "=", r"a_2", r"k_1v_1^\top", "+", r"k_2v_2^\top", size=40),
            M(r"S_3", "=", r"a_3a_2", r"k_1v_1^\top", "+", r"a_3", r"k_2v_2^\top", "+", r"k_3v_3^\top", size=40),
            M(r"S_t", "=", r"\sum_{s=1}^{t}", r"\Big(\prod_{u=s+1}^{t}a_u\Big)", r"k_s v_s^\top", size=40),
        ]
        for l in lines:
            l.shift(UP * 1.9)
        cur = lines[0]
        self.play(Write(cur))
        for nxt in lines[1:]:
            self.play(TransformMatchingTex(cur, nxt), run_time=1.2)
            cur = nxt
            self.wait(0.4)
        self.pause()

        # ghost layers with survival veils, scrubbed by a tracker
        a = np.array([0.0, 0.85, 0.7, 0.9, 0.6, 0.8])
        t_tr = ValueTracker(1)
        ghosts = VGroup()
        for s in range(6):
            gsh = sheet(3, 4, cell=0.32, data=np.random.default_rng(s).uniform(-1, 1, (3, 4)))
            gsh.shift(RIGHT * (s * 0.45) + DOWN * (s * 0.12))
            ghosts.add(gsh)
        ghosts.move_to(LEFT * 3.0 + DOWN * 1.0)

        def veil_op(s):
            t = int(round(t_tr.get_value()))
            if s >= t:
                return 0.0
            return float(np.prod(a[s + 1:t])) if s + 1 < t else 1.0

        for s, gsh in enumerate(ghosts):
            gsh.add_updater(lambda m, s=s: m.set_opacity(veil_op(s)))
        self.add(ghosts)
        tl = always_redraw(lambda: T(f"t = {int(round(t_tr.get_value()))}: veil = survival ∏aᵤ", 20, FADE).next_to(ghosts, DOWN, buff=0.3))
        self.add(tl)
        self.play(t_tr.animate.set_value(6), run_time=3, rate_func=linear)
        self.pause()

        boxed = M(r"y_t=\sum_{s=1}^{t}", r"\underbrace{\Big[\prod_{u=s+1}^{t}a_u\Big]}_{\text{survival}}", r"\underbrace{(q_t^\top k_s)}_{\text{match}}", r"\underbrace{v_s}_{\text{content}}", size=38)
        boxed[1].set_color(FADE); boxed[2].set_color(Q_COL); boxed[3].set_color(V_COL)
        boxed.move_to(RIGHT * 2.6 + DOWN * 1.0)
        box = SurroundingRectangle(boxed, color=MAMBA2, buff=0.15)
        self.play(Write(boxed), Create(box))
        self.pause()
        attn = M(r"\text{Theo: } o_t=\sum_{s\le t}\alpha_{ts}\,v_s", size=34, color=ATTN).next_to(box, UP, buff=0.3)
        diffs = T("no softmax normalisation · extra decay · signed weights", 18, MAMBA).next_to(box, DOWN, buff=0.2)
        self.play(FadeIn(attn, shift=LEFT * 0.3))
        self.play(FadeIn(diffs))
        self.dont_say("Mamba-2 stores past tokens and looks them up. It stores only Sₜ; this is algebra.")
        self.pause(1.5)
        for gsh in ghosts:
            gsh.clear_updaters()
        tl.clear_updaters()
        self.clear_all()


class S14MatrixForm(Deck):
    def construct(self):
        self.header("14", "The matrix form: two heatmaps that rhyme", ACT, "IDENTITY")
        a = np.array([0.9, 0.8, 0.5, 0.75])  # a_1 is unused when S0 = 0
        L = decay_mask(a)
        strip = VGroup(*[M(f"a_{i+1}={a[i]:g}", size=32, color=FADE if i == 0 else RESID) for i in range(4)]).arrange(RIGHT, buff=0.6).shift(UP * 2.2)
        mat = DecimalMatrix(np.zeros((4, 4)), element_to_mobject_config={"num_decimal_places": 3, "font_size": 32}, h_buff=1.15)
        mat.shift(LEFT * 3.4 + DOWN * 0.3)
        Llab = M("L=", size=40).next_to(mat, LEFT)
        ents = mat.get_entries()
        for e in ents:
            e.set_opacity(0)
        self.play(FadeIn(strip), FadeIn(mat.get_brackets()), FadeIn(Llab))
        rule = M(r"L_{ts}=\prod_{u=s+1}^{t}a_u\ (s\le t),\qquad 0\ (s>t)", size=34).shift(RIGHT * 3.4 + UP * 0.9)
        self.play(Write(rule))
        for t in range(4):
            anims = []
            for s in range(4):
                e = ents[4 * t + s]
                e.set_value(L[t, s])
                e.set_color(MAMBA2 if s <= t else FADE)
                anims.append(e.animate.set_opacity(1))
            self.play(LaggedStart(*anims, lag_ratio=0.15), *[Indicate(strip[u], color=MOE) for u in range(1, t + 1)], run_time=0.9)
        hl = [SurroundingRectangle(ents[13], color=MOE), SurroundingRectangle(ents[12], color=MOE)]
        chk = VGroup(M(r"L_{4,2}=a_3a_4=.5\times.75=.375", size=30), M(r"L_{4,1}=a_2a_3a_4=.8\times.5\times.75=.3", size=30)).arrange(DOWN, aligned_edge=LEFT).next_to(rule, DOWN, buff=0.5)
        self.play(Create(hl[0]), Write(chk[0]))
        self.play(Create(hl[1]), Write(chk[1]))
        tested = T("checked by test_on_screen_4x4_mask", 16, FADE).next_to(chk, DOWN, buff=0.2)
        self.play(FadeIn(tested))
        self.pause()

        self.play(FadeOut(VGroup(strip, mat, Llab, rule, chk, tested, *hl)))
        rng = np.random.default_rng(4)
        n = 12
        A = causal_softmax(n, seed=2)
        Q = rng.normal(size=(n, 2)); K = rng.normal(size=(n, 2))
        Ms = decay_mask(rng.uniform(0.6, 0.95, n)) * (Q @ K.T)
        h1 = heatmap(A, cell=0.3, pos_color=ATTN).shift(LEFT * 3.2 + DOWN * 0.2)
        h2 = heatmap(Ms, cell=0.3, pos_color=MAMBA2, neg_color=MAMBA).shift(RIGHT * 3.2 + DOWN * 0.2)
        e1 = M(r"\mathrm{softmax}(QK^\top+\mathcal M)\,V", size=34, color=ATTN).next_to(h1, UP, buff=0.3)
        e2 = M(r"Y=(L\odot QK^\top)\,V", size=40, color=MAMBA2).next_to(h2, UP, buff=0.3)
        self.play(FadeIn(h1, lag_ratio=0.01), FadeIn(h2, lag_ratio=0.01), Write(e1), Write(e2), run_time=1.6)
        ptr1 = SurroundingRectangle(VGroup(*h1[7 * n:8 * n]), color=RESID, buff=0.01)
        ptr2 = SurroundingRectangle(VGroup(*h2[7 * n:8 * n]), color=RESID, buff=0.01)
        self.play(Create(ptr1), Create(ptr2))
        self.play(ptr1.animate.move_to(VGroup(*h1[10 * n:11 * n])), ptr2.animate.move_to(VGroup(*h2[10 * n:11 * n])))
        self.pause()
        l1 = T("SSD ≠ softmax attention (no normalisation, signed)", 22, MAMBA).to_edge(DOWN, buff=0.9)
        l2 = T("same algebra ≠ same memory: Theo stores T cards, Max one N×P sheet", 22, MAMBA).next_to(l1, DOWN, buff=0.15)
        self.play(FadeIn(l1))
        self.play(FadeIn(l2))
        self.pause(1.5)
        self.clear_all()


class S15Semiseparable(Deck):
    def construct(self):
        self.header("15", "Why a T×T matrix can cost O(T): semiseparable blocks", ACT, "IDENTITY")
        n, C, N = 16, 4, 2
        rng = np.random.default_rng(5)
        a = rng.uniform(0.7, 0.97, n)
        Q = rng.normal(size=(n, N)); K = rng.normal(size=(n, N))
        Mx = decay_mask(a) * (Q @ K.T)
        cell = 0.26
        h = heatmap(Mx, cell=cell).shift(LEFT * 3.0 + DOWN * 0.3)
        self.play(FadeIn(h, lag_ratio=0.005), run_time=1.2)
        cuts = VGroup()
        for k in range(1, n // C):
            x = h.get_left()[0] + k * C * cell
            y = h.get_top()[1] - k * C * cell
            cuts.add(DashedLine([x, h.get_top()[1], 0], [x, h.get_bottom()[1], 0], color=MOE),
                     DashedLine([h.get_left()[0], y, 0], [h.get_right()[0], y, 0], color=MOE))
        self.play(LaggedStart(*[Create(c) for c in cuts], lag_ratio=0.15))
        lab = T(f"chunks of C = {C}", 20, MOE).next_to(h, DOWN, buff=0.2)
        self.play(FadeIn(lab))
        self.pause()

        I, J = 3, 1
        blk = VGroup(*[h[i * n + j] for i in range(I * C, (I + 1) * C) for j in range(J * C, (J + 1) * C)])
        frame = SurroundingRectangle(blk, color=RESID, buff=0.02)
        self.play(Create(frame))
        big = blk.copy()
        self.play(big.animate.scale(2.2).move_to(RIGHT * 1.6 + UP * 0.1))
        eq = M(r"L_{ts}=\underbrace{\prod_{u=b_I}^{t}a_u}_{u_t}\ \underbrace{\prod_{u=e_J+1}^{b_I-1}a_u}_{\gamma_{IJ}}\ \underbrace{\prod_{u=s+1}^{e_J}a_u}_{w_s}", size=32)
        eq.to_edge(RIGHT, buff=0.4).shift(DOWN * 1.6)
        self.play(Write(eq))
        self.pause()
        tall = VGroup(*[Rectangle(width=0.3, height=0.3, stroke_color=Q_COL, fill_color=Q_COL, fill_opacity=0.5) for _ in range(C * N)]).arrange_in_grid(C, N, buff=0)
        wide = VGroup(*[Rectangle(width=0.3, height=0.3, stroke_color=K_COL, fill_color=K_COL, fill_opacity=0.5) for _ in range(C * N)]).arrange_in_grid(N, C, buff=0)
        bead = Dot(radius=0.13, color=FADE)
        fold = VGroup(tall, bead, wide).arrange(RIGHT, buff=0.3).move_to(big)
        self.play(Transform(big, fold), run_time=1.5)
        tags = VGroup(T("C×N", 18, Q_COL).next_to(tall, DOWN, buff=0.1), M(r"\gamma_{IJ}", size=28).next_to(bead, UP, buff=0.1),
                      T("N×C", 18, K_COL).next_to(wide, DOWN, buff=0.1))
        self.play(FadeIn(tags))
        rank = M(r"M_{IJ}=\gamma_{IJ}\,\mathrm{diag}(u_I)\,Q_IK_J^\top\,\mathrm{diag}(w_J)\ \Rightarrow\ \mathrm{rank}\le N", size=32).next_to(fold, UP, buff=0.4).to_edge(RIGHT, buff=0.4)
        self.play(Write(rank))
        self.pause()
        name = T("L is 1-semiseparable · M = L⊙QKᵀ is N-semiseparable", 22, MAMBA2).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(name))
        self.pause(1.5)
        self.clear_all()


class S16FourSteps(Deck):
    def construct(self):
        self.header("16", "The four SSD steps and the complexity bill", ACT, "IDENTITY")
        nC = 4
        boxes = VGroup(*[RoundedRectangle(width=2.3, height=1.2, corner_radius=0.1, stroke_color=MAMBA2) for _ in range(nC)]).arrange(RIGHT, buff=0.55).shift(UP * 1.3)
        toks = VGroup(*[VGroup(*[Dot(radius=0.05, color=TOKEN) for _ in range(4)]).arrange(RIGHT, buff=0.35).move_to(b).shift(DOWN * 0.35) for b in boxes])
        self.play(FadeIn(boxes), FadeIn(toks))
        steps = [
            ("1 · intra-chunk: (L⊙QKᵀ)V locally, all chunks at once (matmuls)", "#7BE08A"),
            ("2 · chunk states: each chunk summarises itself into an N×P sheet", K_COL),
            ("3 · state passing: a short scan over T/C chunks, not T tokens", RESID),
            ("4 · correction: read the incoming state with each query", Q_COL),
        ]
        cap = None
        # 1
        c = T(steps[0][0], 22, steps[0][1]).shift(DOWN * 0.3)
        self.play(FadeIn(c), *[b.animate.set_fill(steps[0][1], 0.35) for b in boxes])
        cap = c
        self.pause(0.5)
        # 2
        sheets = VGroup(*[Square(0.4, stroke_color=K_COL, fill_color=K_COL, fill_opacity=0.6).next_to(b, UR, buff=-0.2) for b in boxes])
        c = T(steps[1][0], 22, steps[1][1]).move_to(cap)
        self.play(ReplacementTransform(cap, c), LaggedStart(*[GrowFromCenter(s) for s in sheets], lag_ratio=0.1))
        cap = c
        self.pause(0.5)
        # 3 sequential relay
        c = T(steps[2][0], 22, steps[2][1]).move_to(cap)
        self.play(ReplacementTransform(cap, c))
        cap = c
        relays = []
        for i in range(nC - 1):
            arr = CurvedArrow(sheets[i].get_top(), boxes[i + 1].get_corner(UL) + RIGHT * 0.3, angle=-PI / 3, color=RESID)
            self.play(Create(arr), run_time=0.45)
            relays.append(arr)
        self.pause(0.5)
        # 4
        c = T(steps[3][0], 22, steps[3][1]).move_to(cap)
        self.play(ReplacementTransform(cap, c), *[Indicate(b, color=Q_COL) for b in boxes[1:]])
        cap = c
        self.pause()

        # complexity bill
        tbl = VGroup(
            M(r"\text{train / prefill (chunked):}\ O\big(T\,[NP+C(N+P)]\big)", size=32, color=MAMBA2),
            M(r"\text{decode per token: } O(NP)\qquad \text{state: } O(NP)", size=32, color=MAMBA2),
            M(r"\text{vs attention: } O(T^2d)\ \text{prefill},\ O(Td)\ \text{decode},\ O(T)\ \text{KV}", size=32, color=ATTN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(cap, DOWN, buff=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in tbl], lag_ratio=0.4), run_time=2)
        self.pause()

        # both runtimes, same output
        rng = np.random.default_rng(11)
        Tn, Nn, Pn = 8, 2, 3
        a = rng.uniform(0.5, 0.95, Tn); K = rng.normal(size=(Tn, Nn)); Q = rng.normal(size=(Tn, Nn)); V = rng.normal(size=(Tn, Pn))
        y_rec = ssd_recurrent(a, K, Q, V)[-1]
        y_chk = ssd_chunked(a, K, Q, V, C=4)[-1]
        fmt = lambda v: "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"
        out = VGroup(T("toy T=8: last output y₈", 18, FADE),
                     T("recurrent  " + fmt(y_rec), 20, MAMBA),
                     T("chunked    " + fmt(y_chk), 20, MAMBA2)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        out.to_edge(DOWN, buff=0.3)
        self.play(FadeOut(tbl[2]), FadeIn(out))
        self.pause()
        warn = T("SSD reorganises work into matmuls; it does not remove it", 18, MAMBA).next_to(out, UP, buff=0.15)
        self.play(FadeIn(warn))
        self.pause(1.5)
        self.clear_all()
