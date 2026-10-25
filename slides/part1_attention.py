"""Part 1 — Theo's archive: intro, Q/K/V + heads, KV cache, the residual stream (S01–S04)."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visual.common import *

STORY = ["Leo", "hid", "the green key", "under the bridge"]


class S01Intro(Deck):
    def construct(self):
        ttl = T("Three brothers.  Three memories.", 56, RESID, bold=True)
        self.play(Write(ttl), run_time=1.4)
        self.wait(0.6)
        self.play(ttl.animate.scale(0.55).to_corner(UL, buff=0.5))

        theo = brother("Theo", ATTN, "cards").move_to(LEFT * 4.6 + DOWN * 1.5)
        manu = brother("Manu", MAMBA, "notebook").move_to(DOWN * 1.5)
        maxx = brother("Max", MAMBA2, "tiles").move_to(RIGHT * 4.6 + DOWN * 1.5)
        cast = VGroup(theo, manu, maxx)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.4) for b in cast], lag_ratio=0.25), run_time=1.4)

        ribbon = VGroup(*[chip(w, TOKEN, 24) for w in STORY]).arrange(RIGHT, buff=0.25).to_edge(UP, buff=1.2)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.3) for c in ribbon], lag_ratio=0.2), run_time=1.4)
        self.pause()

        # Theo pins every token; Manu keeps one page; Max works in tiles
        wall = VGroup(*[token_card(w.split()[-1][:6], w=0.95, h=1.1, size=15) for w in STORY]).arrange(RIGHT, buff=0.08)
        wall.next_to(theo, UP, buff=0.3).align_to(theo, LEFT).shift(LEFT * 0.2)
        page = card(2.2, 1.2, MAMBA, 0.18).next_to(manu, UP, buff=0.3)
        lines = VGroup(*[Line(LEFT * 0.8, RIGHT * 0.8, color=MAMBA, stroke_width=3).move_to(page).shift(UP * (0.36 - 0.24 * i)) for i in range(4)])
        tiles = VGroup(*[Square(0.5, stroke_color=MAMBA2, stroke_width=2, fill_color=MAMBA2, fill_opacity=0.2) for _ in range(6)])
        tiles.arrange_in_grid(2, 3, buff=0.1).next_to(maxx, UP, buff=0.4)
        self.play(*[TransformFromCopy(ribbon[i], wall[i]) for i in range(4)], FadeIn(page), FadeIn(tiles), run_time=1.4)
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.25),
                  LaggedStart(*[t.animate.set_fill(opacity=0.7) for t in tiles], lag_ratio=0.15), run_time=1.6)
        self.pause()

        q = chip("Where is Leo's key?", Q_COL, 26).move_to(UP * 1.7)
        self.play(FadeIn(q, scale=0.8))
        halo = glow(wall[3][0], Q_COL)
        self.play(FadeIn(halo), Indicate(wall[3], color=Q_COL, scale_factor=1.25),
                  page.animate.set_fill(MAMBA, 0.4), tiles.animate.set_fill(opacity=0.9), run_time=1.4)
        self.pause(1.4)
        self.clear_all()


class S02QKV(Deck):
    def construct(self):
        self.title("Attention in one picture", ATTN)
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
        self.play(FadeIn(tok), FadeIn(lens), Create(wires))
        self.play(LaggedStart(*[FadeIn(o, shift=RIGHT * 0.3) for o in outs], lag_ratio=0.3), run_time=1.5)
        self.pause()
        self.play(FadeOut(VGroup(tok, lens, wires, outs)))

        # the library
        words = ["Leo", "hid", "the key", "under", "the bridge", "..."]
        cards = VGroup(*[token_card(w, w=1.7, h=1.5, size=18) for w in words]).arrange(RIGHT, buff=0.25).shift(DOWN * 1.6)
        query = chip("query  q", Q_COL, 26).move_to(UP * 1.5 + LEFT * 4)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.1), FadeIn(query))
        scores = np.array([1.5, 0.5, 2.6, 0.2, 2.0, 0.3])
        bars = VGroup(*[Rectangle(width=0.55, height=0.3 * s, stroke_width=0, fill_color=K_COL, fill_opacity=0.9)
                        .next_to(c, UP, buff=0.12) for c, s in zip(cards, scores)])
        beam = Line(query.get_bottom(), cards[0].get_top(), color=Q_COL, stroke_width=4)
        self.add(beam)
        for i in range(len(cards)):
            self.play(beam.animate.put_start_and_end_on(query.get_bottom(), cards[i].get_top()),
                      GrowFromEdge(bars[i], DOWN), run_time=0.35)
        self.play(FadeOut(beam))
        sc = M(r"q\cdot k_s", size=36, color=K_COL).next_to(query, RIGHT, buff=0.8)
        self.play(Write(sc))
        self.pause()

        w = np.exp(scores) / np.exp(scores).sum()
        nb = VGroup(*[Rectangle(width=0.55, height=max(0.04, 2.4 * wi), stroke_width=0, fill_color=K_COL, fill_opacity=0.9)
                      .next_to(c, UP, buff=0.12) for c, wi in zip(cards, w)])
        pct = VGroup(*[T(f"{100 * wi:.0f}%", 16, RESID, bold=True).next_to(b, UP, buff=0.05) for b, wi in zip(nb, w)])
        sm = M(r"\mathrm{softmax}", size=36, color=ATTN).move_to(sc)
        self.play(Transform(bars, nb), FadeIn(pct), Transform(sc, sm), run_time=1.3)
        self.pause()

        out = card(2.0, 1.0, V_COL, 0.4).move_to(RIGHT * 4 + UP * 1.5)
        out_t = T("output", 22, V_COL, bold=True).move_to(out)
        fl = [c[0].copy().set_fill(V_COL, opacity=0.3 + 0.6 * wi).set_stroke(V_COL, opacity=0.6) for c, wi in zip(cards, w)]
        self.play(LaggedStart(*[f.animate.scale(0.3).move_to(out) for f in fl], lag_ratio=0.08), run_time=1.5)
        self.play(FadeIn(out), FadeIn(out_t), *[FadeOut(f) for f in fl])
        self.pause()

        # many heads
        self.play(FadeOut(VGroup(cards, bars, pct, sc, query, out, out_t)))
        heads = VGroup()
        pats = [
            ("head 1", "who did what", np.array([[1, 0, 0, 0], [.8, .2, 0, 0], [.1, .8, .1, 0], [0, .1, .2, .7]])),
            ("head 2", "nearby words", np.array([[1, 0, 0, 0], [.1, .9, 0, 0], [0, .1, .9, 0], [0, 0, .1, .9]])),
            ("head 3", "far back", np.array([[1, 0, 0, 0], [1, 0, 0, 0], [.8, .1, .1, 0], [.9, 0, .05, .05]])),
        ]
        cols = (ATTN, MAMBA2, MOE)
        for (n, d, m), c in zip(pats, cols):
            h = heat(m, 0.42, pos=c)
            fr = card(2.4, 2.9, c, 0.1)
            h.move_to(fr).shift(UP * 0.15)
            lab = T(n, 22, c, bold=True).move_to(fr.get_top() + DOWN * 0.3)
            sub = T(d, 16, FADE).move_to(fr.get_bottom() + UP * 0.3)
            heads.add(VGroup(fr, h, lab, sub))
        heads.arrange(RIGHT, buff=0.5).shift(UP * 0.1)
        self.play(LaggedStart(*[FadeIn(h, shift=UP * 0.3) for h in heads], lag_ratio=0.25), run_time=1.6)
        merge = card(3.4, 0.7, RESID, 0.1).shift(DOWN * 2.6)
        merge_t = M(r"\mathrm{concat}\ \to\ W_O", size=30).move_to(merge)
        self.play(LaggedStart(*[h.animate.scale(0.98) for h in heads], lag_ratio=0.1),
                  Create(merge), Write(merge_t),
                  *[Create(Line(h.get_bottom(), merge.get_top(), color=h[0].get_color(), stroke_width=3)) for h in heads])
        eq = M(r"O=\mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d}}\Big)V", size=40).to_edge(UP, buff=0.55).to_edge(RIGHT, buff=0.7)
        self.play(Write(eq))
        self.pause(1.4)
        self.clear_all()


class S03KVCache(Deck):
    def construct(self):
        self.title("Prefill  →  KV cache  →  decode", ATTN)
        n, cell = 8, 0.34

        def tri(color):
            g = VGroup()
            for i in range(n):
                for j in range(i + 1):
                    g.add(Square(cell, stroke_width=1, stroke_color=BG1, fill_color=color, fill_opacity=0.8).move_to([j * cell, -i * cell, 0]))
            return g

        # prefill: the whole prompt at once
        pre = tri(ATTN).move_to(LEFT * 4.2 + DOWN * 0.3)
        pre_t = T("prefill", 30, ATTN, bold=True).next_to(pre, UP, buff=0.35)
        cost1 = M(r"T^2", size=44, color=ATTN).next_to(pre, DOWN, buff=0.4)
        pre.set_fill(opacity=0.1)
        self.play(FadeIn(pre), FadeIn(pre_t))
        self.play(LaggedStart(*[sq.animate.set_fill(opacity=0.85) for sq in pre], lag_ratio=0.012), run_time=1.6)
        self.play(Write(cost1))
        self.pause()

        # K and V are final -> put them in the cache
        cache = VGroup(*[token_card("", w=0.36, h=0.7) for _ in range(n)]).arrange(RIGHT, buff=0.05).move_to(RIGHT * 3.0 + UP * 0.9, aligned_edge=LEFT).align_to(RIGHT * 1.0, LEFT)
        cache_t = T("KV cache", 30, V_COL, bold=True).move_to(RIGHT * 3.3 + UP * 2.0)
        self.play(TransformFromCopy(pre, cache), FadeIn(cache_t), run_time=1.3)

        # decode: one new query scans the cache
        meter_bg = RoundedRectangle(width=6, height=0.3, corner_radius=0.15, stroke_color=FADE, stroke_width=2).move_to(RIGHT * 3.3 + DOWN * 1.9)
        meter = RoundedRectangle(width=6 * n / 16, height=0.3, corner_radius=0.15, stroke_width=0, fill_color=V_COL, fill_opacity=0.85)
        meter.align_to(meter_bg, LEFT).move_to(meter_bg, coor_mask=[0, 1, 0])
        mt = T("memory", 20, V_COL).next_to(meter_bg, DOWN, buff=0.12)
        self.play(FadeIn(meter_bg), FadeIn(meter), FadeIn(mt))
        for k in range(3):
            q = Rectangle(width=cache.width + 0.5, height=0.14, stroke_width=0, fill_color=Q_COL, fill_opacity=0.9)
            q.next_to(cache, DOWN, buff=0.15).align_to(cache, LEFT)
            newc = token_card("", w=0.36, h=0.7).next_to(cache, RIGHT, buff=0.05)
            self.play(FadeIn(newc, shift=LEFT * 0.3), GrowFromEdge(q, LEFT), run_time=0.55)
            cache.add(newc)
            self.play(FadeOut(q), meter.animate.stretch_to_fit_width(6 * (n + k + 1) / 16).align_to(meter_bg, LEFT), run_time=0.3)
        dec_t = T("decode", 30, Q_COL, bold=True).move_to(RIGHT * 3.3 + DOWN * 0.45)
        cost2 = M(r"T\ \text{per new token}", size=38, color=Q_COL).next_to(dec_t, DOWN, buff=0.25)
        self.play(FadeIn(dec_t), Write(cost2))
        self.pause()

        # without a cache: redo the whole thing every token
        redo = pill("no cache  →  redo  T²  every token", MAMBA, 22).to_edge(DOWN, buff=0.7).shift(LEFT * 3.4)
        okay = pill("with cache  →  T  per token", MAMBA2, 22).to_edge(DOWN, buff=0.7).shift(RIGHT * 3.4)
        self.play(FadeIn(redo, shift=UP * 0.2))
        self.play(FadeIn(okay, shift=UP * 0.2))
        self.pause(1.4)
        self.clear_all()


class S04Bank(Deck):
    def construct(self):
        self.title("Tokens evolve in the residual stream", RESID, accent=ATTN)

        def lane(x, sentence, hi, accent, axis_icon):
            plane = card(6.3, 4.0, accent, 0.07).move_to([x, -0.35, 0])
            parts = sentence.split(hi)
            s = VGroup(T(parts[0], 22, TOKEN), T(hi, 26, accent, bold=True), T(parts[1], 22, TOKEN)).arrange(RIGHT, buff=0.14)
            s.move_to([x, 1.35, 0])
            return plane, s

        lx, rx = -3.55, 3.55
        pl, sl = lane(lx, "She sat on the river bank", "bank", RIVER, None)
        pr, sr = lane(rx, "He paid cash at the bank", "bank", FIN, None)
        self.play(FadeIn(pl), FadeIn(pr), FadeIn(sl), FadeIn(sr))

        # the same starting embedding
        o_l, o_r = pl.get_center() + DOWN * 0.45 + LEFT * 2.2, pr.get_center() + DOWN * 0.45 + LEFT * 2.2
        e0 = [arrow(o, o + RIGHT * 0.8 + UP * 0.08, TOKEN, 6) for o in (o_l, o_r)]
        e_labs = VGroup(*[T("starts as the same embedding", 16, TOKEN).move_to(p.get_bottom() + UP * 0.3, aligned_edge=DOWN) for p in (pl, pr)])
        for lb, p in zip(e_labs, (pl, pr)):
            lb.align_to(p, LEFT).shift(RIGHT * 0.3)
        self.play(*[GrowArrow(a) for a in e0], FadeIn(e_labs))
        self.pause()

        # each layer writes a delta onto the residual (tip to tail)
        steps_l = [(UP * 0.5 + RIGHT * 0.4, ATTN), (UP * 0.15 + RIGHT * 0.55, MOE), (UP * 0.55 + RIGHT * 0.35, ATTN), (UP * 0.2 + RIGHT * 0.5, MOE)]
        steps_r = [(DOWN * 0.3 + RIGHT * 0.5, ATTN), (DOWN * 0.05 + RIGHT * 0.6, MOE), (DOWN * 0.35 + RIGHT * 0.45, ATTN), (DOWN * 0.1 + RIGHT * 0.5, MOE)]
        tip_l, tip_r = e0[0].get_end(), e0[1].get_end()
        for k, ((dl, cl), (dr, cr)) in enumerate(zip(steps_l, steps_r)):
            al, ar = arrow(tip_l, tip_l + dl * 0.95, cl, 5), arrow(tip_r, tip_r + dr * 0.85, cr, 5)
            self.play(GrowArrow(al), GrowArrow(ar), run_time=0.6)
            tip_l, tip_r = al.get_end(), ar.get_end()
        # the final residual vectors
        fl, fr = arrow(o_l, tip_l, RESID, 7), arrow(o_r, tip_r, RESID, 7)
        self.play(Create(fl), Create(fr), FadeIn(glow(Dot(tip_l), RIVER, 3)), FadeIn(glow(Dot(tip_r), FIN, 3)))
        lab_l = T("river", 26, RIVER, bold=True).next_to(tip_l, RIGHT, buff=0.3)
        lab_r = T("money", 26, FIN, bold=True).next_to(tip_r, RIGHT, buff=0.3)
        self.play(FadeIn(lab_l), FadeIn(lab_r))
        eq = M(r"r^{(\ell+1)}=r^{(\ell)}+\mathrm{block}_\ell\!\big(\mathrm{norm}\,r^{(\ell)}\big)", size=34).to_edge(DOWN, buff=0.45)
        self.play(Write(eq))
        self.pause(1.6)
        self.clear_all()
