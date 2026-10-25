"""Act II — The notebook brothers: RNN → S4 → Mamba-1 → Mamba-2 (slides 7–11)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *

RUNGS = ["sequential training", "content-blind dynamics", "scan can't use tensor cores", "fixed-size state: addressability?"]


def ladder(solved=0, lit=None):
    """Persistent 'problem ladder' on the left edge; rungs < solved are struck through."""
    lit = solved if lit is None else lit
    rows = VGroup()
    for i, r in enumerate(RUNGS):
        col = FADE if i < solved else (MAMBA if i <= lit else "#2A3040")
        t = T(r, 16, col)
        row = VGroup(Dot(radius=0.06, color=col), t).arrange(RIGHT, buff=0.15)
        if i < solved:
            row.add(Line(t.get_left(), t.get_right(), color=FADE, stroke_width=2))
        rows.add(row)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    head = T("problem ladder", 16, FADE)
    return VGroup(head, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(DL, buff=0.35)


class S07RNN(Deck):
    def construct(self):
        self.header("7", "Recurrence came first: the RNN and its problems", "Act II · The notebook brothers", "PROPERTY")
        line = NumberLine(x_range=[0, 4, 1], length=11, include_ticks=False, color=FADE).shift(UP * 1.2)
        events = [("1990", "Elman RNN"), ("1997", "LSTM"), ("2014", "GRU · seq2seq"), ("2014", "attention, to help RNNs\n(Bahdanau)"), ("2017", "Transformer:\nno recurrence")]
        dots = VGroup()
        for i, (yr, name) in enumerate(events):
            d = Dot(line.n2p(i), color=MAMBA if i < 4 else ATTN)
            lab = T(name, 18, d.get_color()).next_to(d, UP, buff=0.2)
            year = T(yr, 16, FADE).next_to(d, DOWN, buff=0.15)
            dots.add(VGroup(d, lab, year))
        self.play(Create(line))
        self.play(LaggedStart(*[FadeIn(d, shift=UP * 0.2) for d in dots], lag_ratio=0.3), run_time=2)
        back = T("…and now we're bringing recurrence back", 22, MAMBA).next_to(line, DOWN, buff=1.0)
        self.play(Write(back))
        self.pause()
        self.play(FadeOut(VGroup(line, dots, back)))

        # dominoes: h_t waits for h_{t-1}
        self.lad = ladder(0, 0)
        dominos = VGroup(*[Rectangle(width=0.25, height=1.1, stroke_color=MAMBA, fill_color=MAMBA, fill_opacity=0.4) for _ in range(9)]).arrange(RIGHT, buff=0.45)
        dominos.shift(UP * 0.9 + RIGHT * 1.2)
        labels = VGroup(*[M(f"h_{i+1}", size=26, color=MAMBA).next_to(d, DOWN, buff=0.15) for i, d in enumerate(dominos)])
        self.play(FadeIn(dominos), FadeIn(labels), FadeIn(self.lad))
        p1 = T("Problem 1: h₅ must wait for h₄ — no parallelism over time", 22).next_to(dominos, UP, buff=0.7)
        self.play(FadeIn(p1))
        self.play(Succession(*[Rotate(d, -PI / 3, about_point=d.get_corner(DR)) for d in dominos], run_time=2.5))
        self.pause()

        # gradient pulse multiplying through Jacobians
        eq = M(r"h_t=\tanh(Wh_{t-1}+Ux_t+b)", r"\qquad \frac{\partial h_T}{\partial h_s}=\prod_{t=s+1}^{T}\mathrm{diag}(1-h_t^2)\,W", size=34).shift(DOWN * 1.2 + RIGHT * 1.2)
        self.play(Write(eq))
        p2 = T("Problem 2: the backward signal shrinks or explodes", 22).next_to(eq, DOWN, buff=0.4)
        self.play(FadeIn(p2))
        for factor, col in ((0.6, FADE), (1.5, RESID)):
            pulse = Dot(dominos[-1].get_top() + UP * 0.3, radius=0.12, color=col)
            self.add(pulse)
            for d in reversed(dominos[:-1]):
                r = pulse.radius * factor if hasattr(pulse, "radius") else 0.12
                new_w = np.clip(pulse.width * factor, 0.02, 0.6)
                self.play(pulse.animate.move_to(d.get_top() + UP * 0.3).set(width=new_w), run_time=0.18)
            self.play(FadeOut(pulse), run_time=0.2)
        gates = T("LSTM / GRU gates relieve this partly", 18, FADE).next_to(p2, DOWN, buff=0.15)
        self.play(FadeIn(gates))
        root = SurroundingRectangle(eq[0], color=MAMBA)
        why = T("both problems: a nonlinearity inside the recurrence", 20, MAMBA).next_to(root, UP, buff=0.1)
        self.play(Create(root), FadeIn(why))
        self.pause(1.5)
        self.clear_all()


class S08LinearSSM(Deck):
    def construct(self):
        self.header("8", "Drop the nonlinearity: linear state spaces (S4)", "Act II · The notebook brothers", "PROPERTY")
        self.lad = ladder(0, 1)
        self.add(self.lad)
        eq = M(r"s_t=\bar A s_{t-1}+\bar B x_t,\quad y_t=C s_t\ \ \Rightarrow\ \ y=\bar K * x,\quad \bar K_j=C\bar A^{\,j}\bar B", size=36).shift(UP * 2.0)
        self.play(Write(eq))
        ax = Axes(x_range=[0, 16, 1], y_range=[0, 2.2, 1], x_length=10, y_length=2.6, tips=False,
                  axis_config={"stroke_color": FADE}).shift(DOWN * 0.2 + RIGHT * 1.0)
        self.play(Create(ax))
        words = ["Leo", "um", "green", "um", "key"]
        pos = [1, 3, 5, 7, 9]
        kernel = lambda j: 1.0 * 0.72 ** j
        ripples = VGroup()
        total = np.zeros(17)
        for w, p in zip(words, pos):
            tok = T(w, 18, MAMBA if w != "um" else FADE).next_to(ax.c2p(p, 0), DOWN, buff=0.2)
            bars = VGroup()
            for j in range(0, 16 - p):
                bars.add(Rectangle(width=0.22, height=max(0.01, 1.1 * kernel(j)), stroke_width=0, fill_color=MAMBA, fill_opacity=0.35)
                         .move_to(ax.c2p(p + j, 0), aligned_edge=DOWN))
                total[p + j] += kernel(j)
            self.play(FadeIn(tok, shift=DOWN * 0.2), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05), run_time=0.8)
            ripples.add(VGroup(tok, bars))
        out = ax.plot_line_graph(list(range(17)), list(np.clip(total * 0.7, 0, 2.1)), line_color=RESID, add_vertex_dots=False)
        self.play(Create(out))
        par = T("Linear ⇒ a convolution ⇒ parallel training (FFT, O(T log T))", 22, MAMBA2).next_to(ax, DOWN, buff=0.55)
        self.play(FadeIn(par))
        self.pause()
        same = T("…but \"um\" and \"Leo\" make the same ripple: content-blind", 22, MAMBA).next_to(par, DOWN, buff=0.2)
        self.play(Indicate(ripples[0][1], color=MAMBA), Indicate(ripples[1][1], color=MAMBA), FadeIn(same))
        self.play(Transform(self.lad, ladder(1, 1)))
        self.pause(1.5)
        self.clear_all()


class S09MambaSelection(Deck):
    def construct(self):
        self.header("9", "Mamba-1: selection", "Act II · The notebook brothers", "SCHEMATIC")
        self.lad = ladder(1, 1)
        self.add(self.lad)
        book = Rectangle(width=3.2, height=2.2, stroke_color=MAMBA, fill_color=MAMBA, fill_opacity=0.08).shift(RIGHT * 2.5 + UP * 0.6)
        btitle = T("Manu's notebook (fixed size)", 18, MAMBA).next_to(book, UP, buff=0.1)
        dial_c = LEFT * 1.5 + UP * 0.6
        arc = Arc(radius=0.9, start_angle=PI, angle=-PI, color=FADE, stroke_width=4).move_arc_center_to(dial_c)
        delta = ValueTracker(0.5)
        needle = always_redraw(lambda: Line(dial_c, dial_c + 0.85 * np.array([np.cos(PI * (1 - delta.get_value())), np.sin(PI * (1 - delta.get_value())), 0]), color=MOE, stroke_width=5))
        dlab = M(r"\Delta_t", size=36, color=MOE).next_to(arc, DOWN, buff=0.15)
        hint = T("small Δ: keep the past   ·   large Δ: reset & write", 16, FADE).next_to(dlab, DOWN, buff=0.1)
        self.play(FadeIn(book), FadeIn(btitle), Create(arc), FadeIn(dlab), FadeIn(hint))
        self.add(needle)
        stream = [("Leo", 0.9), ("um", 0.05), ("green key", 0.8), ("uh", 0.08), ("bridge", 0.85)]
        ink = VGroup()
        for i, (w, d) in enumerate(stream):
            tok = token_square(w, color=MAMBA if d > 0.5 else FADE, fs=18).move_to(LEFT * 1.5 + UP * 2.6)
            self.play(FadeIn(tok, shift=LEFT * 0.3), delta.animate.set_value(d), run_time=0.5)
            decay = np.exp(-2.0 * d)
            line = Line(LEFT * 1.3, RIGHT * 1.3, color=MAMBA, stroke_width=1 + 7 * d).move_to(book).shift(UP * (0.8 - 0.38 * i))
            anims = [m.animate.set_stroke(opacity=m.get_stroke_opacity() * decay) for m in ink]
            self.play(*anims, Create(line), FadeOut(tok, shift=RIGHT * 2), run_time=0.6)
            ink.add(line)
        self.pause()
        eq = M(r"s_t=\bar A_t s_{t-1}+\bar B_t x_t,\quad y_t=C_t^\top s_t,\quad \bar A_t=e^{\Delta_t A},\ \bar B_t\approx\Delta_t B_t", size=34).shift(DOWN * 1.3 + RIGHT * 1.0)
        self.play(Write(eq))
        sub = T("Δ, B, C depend on the current token  →  no fixed kernel, no convolution", 20, MAMBA).next_to(eq, DOWN, buff=0.2)
        self.play(FadeIn(sub))
        self.pause()

        # parallel scan tree
        self.play(FadeOut(VGroup(book, btitle, arc, dlab, hint, ink, sub)), FadeOut(needle), eq.animate.to_edge(UP, buff=1.2))
        leaves = VGroup(*[Dot(radius=0.09, color=MAMBA) for _ in range(8)]).arrange(RIGHT, buff=0.75).shift(DOWN * 1.4 + RIGHT * 1.0)
        tree = VGroup(leaves)
        prev = leaves
        lines = VGroup()
        for lvl in range(3):
            nxt = VGroup(*[Dot(radius=0.09, color=MAMBA2).move_to((prev[2 * i].get_center() + prev[2 * i + 1].get_center()) / 2 + UP * 0.8) for i in range(len(prev) // 2)])
            for i, d in enumerate(nxt):
                lines.add(Line(prev[2 * i].get_center(), d.get_center(), color=FADE), Line(prev[2 * i + 1].get_center(), d.get_center(), color=FADE))
            tree.add(nxt)
            prev = nxt
        op = M(r"(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\ a_2b_1+b_2)\quad\text{associative}\ \Rightarrow\ \text{depth}\ \log T", size=32).next_to(leaves, DOWN, buff=0.4)
        self.play(FadeIn(leaves))
        for lvl in range(1, 4):
            seg = [l for l in lines][sum(2 * (8 >> k) for k in range(1, lvl)):sum(2 * (8 >> k) for k in range(1, lvl + 1))]
            self.play(*[Create(l) for l in seg], FadeIn(tree[lvl]), run_time=0.6)
        self.play(Write(op))
        hw = T("hardware-aware: scan in fast SRAM, write only y to HBM, recompute in backward", 18, FADE).next_to(op, DOWN, buff=0.15)
        self.play(FadeIn(hw))
        self.play(Transform(self.lad, ladder(2, 2)))
        self.pause(1.5)
        self.clear_all()


class S10MambaProblems(Deck):
    def construct(self):
        self.header("10", "Mamba-1's problems", "Act II · The notebook brothers", "PROPERTY")
        self.lad = ladder(2, 2)
        self.add(self.lad)
        ax = Axes(x_range=[0, 2, 1], y_range=[0, 330, 100], x_length=3.2, y_length=3.6, tips=False,
                  axis_config={"stroke_color": FADE}, y_axis_config={"include_numbers": True, "font_size": 18}).shift(LEFT * 3.8 + UP * 0.2)
        b1 = Rectangle(width=0.9, height=ax.c2p(0, 312)[1] - ax.c2p(0, 0)[1], stroke_width=0, fill_color=MAMBA2, fill_opacity=0.85).move_to(ax.c2p(0.5, 0), aligned_edge=DOWN)
        b2 = Rectangle(width=0.9, height=ax.c2p(0, 19.5)[1] - ax.c2p(0, 0)[1], stroke_width=0, fill_color=MAMBA, fill_opacity=0.85).move_to(ax.c2p(1.5, 0), aligned_edge=DOWN)
        l1 = T("BF16 matmul\n(tensor cores)\n≈312", 16, MAMBA2).next_to(b1, UP, buff=0.1)
        l2 = T("non-matmul FP32\n≈19.5", 16, MAMBA).next_to(b2, UP, buff=0.1)
        unit = T("A100 TFLOP/s (datasheet)", 16, FADE).next_to(ax, DOWN, buff=0.2)
        self.play(Create(ax), GrowFromEdge(b1, DOWN), GrowFromEdge(b2, DOWN), FadeIn(l1), FadeIn(l2), FadeIn(unit))
        x16 = T("≈16×", 40, MOE).next_to(ax, RIGHT, buff=0.4)
        self.play(Write(x16))
        txt = T("The selective scan is not a matmul:\nit runs on the slow side.\nSo the state stays small (N = 16).", 22).next_to(x16, RIGHT, buff=0.4)
        self.play(FadeIn(txt))
        self.pause()
        self.play(FadeOut(VGroup(ax, b1, b2, l1, l2, unit, x16, txt)))

        # addressability: many bindings into one page vs a growing wall
        wall_t = T("Theo: wall grows", 20, ATTN).shift(LEFT * 3.2 + UP * 1.9)
        book_t = T("Manu: one page", 20, MAMBA).shift(RIGHT * 3.0 + UP * 1.9)
        page = Rectangle(width=3.2, height=2.2, stroke_color=MAMBA).shift(RIGHT * 3.0 + UP * 0.4)
        self.play(FadeIn(wall_t), FadeIn(book_t), Create(page))
        rng = np.random.default_rng(3)
        wall = counter = None
        n_strokes = 0
        for n in (4, 8, 16, 32):
            g = VGroup(*[Rectangle(width=0.16, height=0.3, stroke_width=1, stroke_color=K_COL, fill_color=V_COL, fill_opacity=0.4)
                         for _ in range(n)]).arrange_in_grid(cols=8, buff=0.06).move_to(LEFT * 3.2 + UP * 0.4)
            new_strokes = [Line(page.get_left() + RIGHT * 0.2 + UP * rng.uniform(-0.9, 0.9), page.get_right() + LEFT * 0.2 + UP * rng.uniform(-0.9, 0.9),
                                color=MAMBA, stroke_width=3, stroke_opacity=0.35) for _ in range(n - n_strokes)]
            n_strokes = n
            c = T(f"{n} bindings", 22).move_to(LEFT * 3.2 + DOWN * 1.2)
            if wall is None:
                self.play(FadeIn(g), FadeIn(c), *[Create(s) for s in new_strokes], run_time=0.8)
            else:
                self.play(ReplacementTransform(wall, g), ReplacementTransform(counter, c), *[Create(s) for s in new_strokes], run_time=0.8)
            wall, counter = g, c
        q = T("capacity / interference?", 26, MAMBA).next_to(page, DOWN, buff=0.3)
        self.play(Write(q))
        self.play(Transform(self.lad, ladder(2, 3)))
        self.play(FadeOut(self.lad))
        self.dont_say("a fixed state must forget. Mamba can do associative recall; capacity is the question.")
        self.pause(1.5)
        self.clear_all()


class S11Mamba2Knob(Deck):
    def construct(self):
        self.header("11", "Mamba-2: one knob per head", "Act II · The notebook brothers", "PROPERTY")
        self.lad = ladder(2, 3)
        self.add(self.lad)
        n = 5
        dials = VGroup()
        for i in range(n):
            c = Circle(radius=0.28, stroke_color=MAMBA)
            ang = PI * (0.2 + 0.15 * i)
            nd = Line(c.get_center(), c.get_center() + 0.25 * np.array([np.cos(ang), np.sin(ang), 0]), color=MOE, stroke_width=4)
            dials.add(VGroup(c, nd).move_to([i * 0.62, -i * 0.55, 0]))
        dials.move_to(LEFT * 1.6 + DOWN * 0.1)
        br = VGroup(M(r"\Big[", size=150).next_to(dials, LEFT, buff=0.1), M(r"\Big]", size=150).next_to(dials, RIGHT, buff=0.1))
        lab1 = M(r"A_t^{\text{Mamba-1}}=\mathrm{diag}(a_{t,1},\dots,a_{t,N})", size=34, color=MAMBA).next_to(dials, UP, buff=0.45)
        self.play(FadeIn(dials), FadeIn(br), Write(lab1))
        self.pause()
        one = VGroup(Circle(radius=0.45, stroke_color=MAMBA2), Line(ORIGIN, 0.4 * RIGHT, color=MOE, stroke_width=5).rotate(PI / 3, about_point=ORIGIN))
        one.move_to(LEFT * 2.6 + DOWN * 0.1)
        eye = M(r"\times\, I_N", size=48, color=MAMBA2).next_to(one, RIGHT, buff=0.3)
        lab2 = M(r"A_t^{\text{Mamba-2}}=a_t\,I_N,\quad a_t=e^{\Delta_t A},\ A<0", size=34, color=MAMBA2).move_to(lab1)
        self.play(ReplacementTransform(dials, one), FadeOut(br), FadeIn(eye), TransformMatchingTex(lab1, lab2), run_time=1.6)
        cap = T("every coordinate of this head's notebook fades at the same rate", 20, FADE).next_to(eye, DOWN, buff=0.6).shift(LEFT * 0.4)
        self.play(FadeIn(cap))
        self.pause()

        tiles = VGroup(*[Square(0.42, stroke_color=MAMBA2, fill_color=MAMBA2, fill_opacity=0.35) for _ in range(12)]).arrange_in_grid(3, 4, buff=0.08).shift(RIGHT * 3.6 + UP * 0.4)
        press = VGroup(Rectangle(width=2.4, height=0.4, fill_color=FADE, fill_opacity=0.8, stroke_width=0).next_to(tiles, UP, buff=0.5),
                       T("tensor-core press", 16, RESID))
        press[1].next_to(press[0], UP, buff=0.08)
        self.play(FadeIn(tiles, lag_ratio=0.05), FadeIn(press))
        self.play(press[0].animate.next_to(tiles, UP, buff=0.02), rate_func=there_and_back, run_time=0.8)
        self.play(press[0].animate.next_to(tiles, UP, buff=0.02).set_fill(MAMBA2), rate_func=there_and_back, run_time=0.8)
        gains = VGroup(
            T("✓ recurrence becomes chunked matmuls (Act III)", 18, MAMBA2),
            T("✓ state N: 16 → 64–256", 18, MAMBA2),
            T("✓ x, B, C, Δ from one parallel in-projection; B, C shared across heads", 18, MAMBA2),
            T("SSD core 2–8× faster than Mamba-1 scan (Dao & Gu 2024, their setup)", 16, FADE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(cap, DOWN, buff=0.3)
        gains.align_to(np.array([-2.7, 0, 0]), LEFT)
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.2) for g in gains], lag_ratio=0.4), run_time=2)
        self.play(Transform(self.lad, ladder(3, 3)))
        self.pause()
        warn = T("At fixed N this is a restriction, not more expressivity", 20, MAMBA).to_edge(DOWN, buff=0.25).shift(RIGHT * 1.6)
        self.play(FadeOut(gains[3]), FadeIn(warn))
        self.pause(1.5)
        self.clear_all()
