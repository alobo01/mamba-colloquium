# The Three Brothers
## A Manim visual script for Transformers, Mamba, Mamba‑2 and Mixture‑of‑Experts

**What this is:** a slide‑by‑slide visual script for a talk built in Manim. It is written in the spirit of 3Blue1Brown (a geometric intuition first, then the equation that intuition forces on you), but the characters, palette and art are original.
**Audience:** ML researchers who know the Transformer. No prior knowledge of state‑space models is assumed.
**Length:** about 45 min in full. The ★ slides make up a 25‑min core cut.
**Format:** a [manim‑slides](https://github.com/jeertmans/manim-slides) deck, so you step through it live with the clicker. Every slide is also a normal Manim scene that can be exported as video.
**Companion files:**
- `math_contracts/test_ssd_equivalence.py` checks every identity that appears on screen numerically.
- `reference/original_plan/` holds the original research‑audited storyboard. Its §4 is the full equation contract and its §9 is the source audit.
- `agent_skill/SKILL.md` is the brief for a coding agent.

> **One‑sentence thesis.** *Theo keeps every card, Manu and Max keep one notebook each, and all three hand their work to the same clinic of specialists through one shared conveyor belt.* In architecture terms: attention keeps an addressable history, Mamba compresses history into a state that evolves, Mamba‑2 computes that same state with matrix multiplications, MoE chooses which per‑token computation runs, and the residual stream is how all of them communicate.

---

## Table of contents

- [0. How to build and present this](#0-how-to-build-and-present-this)
- [1. The cast and the visual language](#1-the-cast-and-the-visual-language)
- [Act I — Theo's archive (attention)](#act-i--theos-archive-attention) · slides 1–6
- [Act II — The notebook brothers (RNN → Mamba → Mamba‑2)](#act-ii--the-notebook-brothers-rnn--mamba--mamba2) · slides 7–11
- [Act III — Max's mathematics (SSD in detail)](#act-iii--maxs-mathematics-ssd-in-detail) · slides 12–16
- [Act IV — The clinic, the conveyor belt, and real Transformers](#act-iv--the-clinic-the-conveyor-belt-and-real-transformers) · slides 17–21
- [Act V — Theory, the tunnel, and mechanisms](#act-v--theory-the-tunnel-and-mechanisms) · slides 22–26
- [Appendix A — Reusable mobject library](#appendix-a--reusable-mobject-library)
- [Appendix B — File map and render commands](#appendix-b--file-map-and-render-commands)
- [Appendix C — Math contract additions](#appendix-c--math-contract-additions)
- [Appendix D — References](#appendix-d--references)
- [Appendix E — QA checklist](#appendix-e--qa-checklist)

---

## 0. How to build and present this

### 0.1 Tooling and the Manim agent skill

| Need | Use |
|---|---|
| Rendering engine | **Manim Community Edition** (≥0.19). Requires FFmpeg, plus LaTeX for `MathTex`. Docs: <https://docs.manim.community/en/stable/> |
| Live stepping | **manim‑slides**. Subclass `Slide` / `ThreeDSlide`, call `self.next_slide()` at every click, and `self.next_slide(loop=True)` for idle loops. Run `manim-slides render file.py Class`, then `manim-slides present Class` (or `manim-slides convert` to HTML/PPTX) |
| **Agent skill for coding agents** | [`edoardob90/manim-skills`](https://github.com/edoardob90/manim-skills), a Claude Code skill for ManimCE 0.20.1. Install with `npx skills add edoardob90/manim-skills`, or copy `manim/` into `~/.claude/skills/`. The agent should read `manim/SKILL.md` first, then `animations.md` (updaters, `ValueTracker`), `mobjects.md`, `plotting.md` (axes and curves), `rendering.md` and `gotchas.md` (LaTeX, `.animate` pitfalls), and use `examples/` as a pattern library. Check any API that depends on the version against the current docs |
| Project‑specific contract | `agent_skill/SKILL.md` in this repo: palette, guardrails and test gates |
| Optional narration | `manim-voiceover`, only if a recorded talk is wanted. For a live colloquium, speak over the slides |

Every slide below uses the same template so an agent can turn it into code mechanically:

> **Signature frame**: the one still image the audience should take home.
> **Beats**: numbered steps. Each `⏸` is a `self.next_slide()` (one click).
> **Math**: what appears and at which beat.
> **Manim notes**: mobjects, animations, layout.
> **⚠ Don't say**: the misconception this slide must not create.

### 0.2 Palette (one colour per mathematical object, everywhere)

```python
# visual/palette.py
BG        = "#11141C"  # midnight background
RESID     = "#F2F2F2"  # residual stream / shared workspace
TOKEN     = "#D8D5CD"  # neutral tokens
Q_COL     = "#3AD5D0"  # queries  (what I'm looking for)
K_COL     = "#B78CFF"  # keys     (what I can be found by)
V_COL     = "#FFD166"  # values   (what I hand over)
ATTN      = "#4F8EF7"  # Theo / Transformer / index cards
MAMBA     = "#F28C6B"  # Manu / Mamba-1 / notebook
MAMBA2    = "#69D6B4"  # Max / Mamba-2 / block ledger
MOE       = "#F2C14E"  # clinic / router / experts
FADE      = "#747D8C"  # decay, suppression, ablation
```

**Axis grammar (never break it):** the **horizontal axis is token position (time)** and the **vertical axis is network depth (layers)**. A state that moves sideways is memory across time. A vector that moves upward is the residual stream across layers.

### 0.3 Evidence badges (small pill in the top‑right corner)

| Badge | Meaning |
|---|---|
| `IDENTITY` (white) | An exact mathematical equality, checked in `math_contracts/` |
| `PROPERTY` (blue) | A property of the architecture, or a cited published result |
| `SCHEMATIC` (dotted outline) | A cartoon or toy numbers, **not** a measurement |
| `MEASURED` (solid gold) | Allowed only if real experiment logs are imported, with their source |

### 0.4 Running order and time budget

| Act | Slides | Minutes | Question it answers |
|---|---|---:|---|
| I Theo's archive | 1–6 | 10 | How does attention look things up, cache, and contextualise? |
| II Notebook brothers | 7–11 | 8 | Why recurrence, what did Mamba fix, what did Mamba‑2 fix? |
| III Max's mathematics | 12–16 | 9 | What exactly does Mamba‑2 compute, and why is it fast? |
| IV Clinic, belt, real Transformers | 17–21 | 10 | Why FFNs and MoE, how do layers communicate, and what do real Transformers do? |
| V Theory, tunnel, mechanisms | 22–26 | 8 | Trade‑offs, universality, the attention form of Mamba‑2, interpretability |

**★ Core 25‑min cut:** 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 20, 22, 24, 26.

---

## 1. The cast and the visual language

### 1.1 The brothers analogy

| Character | Architecture | Prop | What he is good at | What it costs him |
|---|---|---|---|---|
| **Theo** (blue sleeves) | Transformer attention | A **wall of index cards**, one card per token per layer. Each card has a purple *title* (key) and a yellow *contents page* (value) | Can go back and **address any earlier card** using the current question | The wall **grows with the story**, and every new question has to scan it |
| **Manu** (coral notebook) | Mamba‑1 (selective SSM) | A **fixed‑size notebook** and a pen whose pressure (Δ) depends on what he is reading. He decides what to write and what to let fade | **Constant memory**. Reads a story of any length at constant cost per word | Can't go back and re‑read old cards; many facts share one page. His bookkeeping can't use the fast "stamping press" (tensor cores) |
| **Max** (mint ledger) | Mamba‑2 (SSD) | The same kind of notebook, but each head has **one fade knob** instead of N | Same constant memory, but his bookkeeping becomes **big matrix multiplications**. He can afford a much larger notebook (N = 64–256 vs 16) | His dynamics are less flexible per head (scalar vs diagonal) |
| **The Clinic** (gold) | Mixture‑of‑Experts FFN | A triage desk (router) and many specialist rooms (experts) | Huge stored capacity, with only a few rooms open per patient | Every room takes up building space (memory), and patients have to be shuttled between buildings (all‑to‑all) |
| **The Conveyor Belt** (white) | Residual stream | A vertical belt under each token that carries a vector upward | Every worker reads from it and **adds** to it. It is the only way they talk | Bandwidth is limited: d dimensions shared by everything |

> The **Clinic is not a fourth brother**. It is a different axis: any brother's layer can be followed by a dense FFN or by an MoE. The brothers are fictional characters, and the talk must never rank them as "smartest".

### 1.2 Two running examples (introduce once, reuse for the whole talk)

1. **Representation:** "The brothers watched fish near the **bank**." / "The brothers deposited money at the **bank**." This example is about *what a vector means*.
2. **Memory:** "Leo left the **green key** under the **bridge**. Noah put the **red key** in the **cupboard**. … Where is Leo's key?" This example is about *how information crosses positions*.

### 1.3 Recurring animation idioms

| Operation | Idiom | Manim |
|---|---|---|
| Token arrives | A card slides in from the right onto the timeline | `LaggedStart(FadeIn(..., shift=LEFT))` |
| Query reads keys | A cyan beam sweeps across the purple titles; bars grow to match dot products | `Line`, `ValueTracker`, `always_redraw` |
| Values flow back | Yellow pages fly to the query and **add up** according to their weights | `Transform` + `.animate.scale(w)` |
| Residual update | A coloured arrow is attached tip to tail; the resultant arrow replaces the old one | `Arrow`, `ReplacementTransform` |
| State decay | The notebook's ink lowers its opacity by a factor of $a_t$ | `ValueTracker` → `set_opacity` |
| Rank‑1 write | Two thin vectors sweep out a coloured rectangle (outer product) | `Rectangle` grid, `LaggedStart` |
| Algebra morph | Terms keep their screen position while the identity rearranges | `TransformMatchingTex` |
| Routing | A token splits into k gold paths and the weighted returns merge | `Arrow`, `AnimationGroup` |
| Counterfactual patch | Clean and corrupt runs side by side; one wire swaps between them | `ReplacementTransform`, `DashedLine` |

**Legibility rules:** at most one primary equation on screen; ≤25 words of on‑screen text per frame; the central 75 % of the frame is reserved for the construction; every symbol's dimension is available as a `Brace` label.

---

## Act I — Theo's archive (attention)

### Slide 1 ★ — Cold open: "Do they remember the same way?" (1:00)

**Signature frame:** three brothers side by side. Theo stands in front of a growing wall of cards, Manu holds a notebook, Max holds a ledger of square tiles. Each has the same question card above him: *"Where is Leo's key?"*

**Beats**
1. A ribbon of story cards scrolls in from the right: *Leo · green key · bridge · Noah · red key · cupboard · …* ⏸
2. Theo pins **each** card to his wall. Manu **rewrites** one page with each card (old ink fades). Max does the same but collects cards into **square batches** first. ⏸
3. The question card drops in. Theo's wall lights up at the *bridge* card. Manu's notebook glows as a whole. Max's ledger glows tile by tile. ⏸
4. Freeze. Title: **"Same question. Three ways to remember."**

**Manim notes:** three `Brother` mobjects (Appendix A) arranged with `VGroup(...).arrange(RIGHT, buff=1.2)`. The story ribbon is a `VGroup` of `TokenCard`s moved by `.animate.shift(LEFT)`. Keep the outcome of the three methods **neutral**: don't show who answers correctly.
**Transition:** the story cards shrink into token squares on a horizontal timeline, and Theo's wall moves to centre stage.

---

### Slide 2 ★ — Query, Key, Value as a library (2:30)

**Signature frame:** a row of index cards. Each has a **purple title strip** (key) and a **yellow body** (value). A **cyan slip** (query) floats above. Bars of different heights rise over the cards, and the yellow bodies blend into a single yellow output card.

**Beats**
1. **Every token makes three things from its current vector $r_t$.** A token square passes through three coloured lenses ($W_Q, W_K, W_V$) and comes out as a cyan slip, a purple title and a yellow page. Caption: *"what I'm looking for / how I can be found / what I'll hand over."* ⏸
2. **Matching is graded, not exact.** The cyan query $q_t$ for *"Leo's key?"* sweeps across the purple titles. Above each card a bar rises to height $q_t^\top k_s$. Show it as a geometric dot product: a small inset with two arrows and the projection shadow of one onto the other. ⏸
3. **Softmax turns scores into a budget.** The bars are exponentiated, then **squeezed so they sum to 1** (they all shrink proportionally into a shared "100 %" bracket). Future cards are greyed out and locked: this is the causal mask. ⏸
4. **Values flow.** Each yellow page flies to the output, scaled by its weight, and the pages stack **tip to tail as vectors**. The output is a *blend*, dominated by the *bridge* card. ⏸
5. Equation build‑up (see Math). The three colours of the equation match the three lenses.

**Math** (built term by term with `TransformMatchingTex` at beat 5):

$$
q_t = W_Q r_t,\qquad k_s = W_K r_s,\qquad v_s = W_V r_s
$$

$$
\alpha_{ts} = \frac{\exp\!\big(q_t^\top k_s/\sqrt{d_k}\big)}{\sum_{j\le t}\exp\!\big(q_t^\top k_j/\sqrt{d_k}\big)},\qquad
o_t = \sum_{s\le t}\alpha_{ts}\,v_s
$$

Then the matrix form for the whole sequence, shown as a small thumbnail heatmap that will return in slide 14:

$$
O = \operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}} + \mathcal{M}_{\text{causal}}\Big)V
$$

**Manim notes:** `IndexCardWall(n=6)`. Bars are `always_redraw(lambda: Rectangle(height=score_tracker.get_value()*...))`. The softmax "squeeze" works by animating a `ValueTracker` from raw `exp` heights to normalised heights. Colour the equation by tokens: `MathTex(r"q_t", "=", "W_Q", "r_t", ...)` then `.set_color_by_tex("q", Q_COL)`.
**⚠ Don't say:** "attention finds *the* matching card". It is always a soft weighted average, and the drawer metaphor is only a first approximation.

---

### Slide 3 ★ — Prefill, the KV cache, and O(T) decode (2:30)

> *Wording note: what is sometimes called "profiling" the prompt is **prefill**. This slide uses that term.*

**Signature frame:** split screen.
- **Left (PREFILL):** a large lower‑triangular $T\times T$ grid that fills in all at once, in big parallel tiles.
- **Right (DECODE):** a single new cyan row that sweeps across $T$ cached purple/yellow columns, while a memory meter at the bottom grows by one card per token.

**Beats**
1. **The wasteful way.** Generate token $T{+}1$ *without* a cache: the whole triangle is recomputed (every K and V for every past token). Generate $T{+}2$: recomputed again. A grey **"redo pile"** of discarded triangles builds up beside the grid. Counter: *work to generate this token ∝ $T^2$*. ⏸
2. **Observation.** Past keys and values **never change**: a causal mask means token 3 cannot see token 7, so its K and V are final once computed. Highlight one past column and stamp it "final". ⏸
3. **The KV cache.** Store every past $(k_s, v_s)$ card in a drawer (per layer, per KV head). Now a new token only computes **its own** $q, k, v$, appends its card, and **one query row** scans the cache. The redo pile vanishes in a puff of grey. ⏸
4. **Prefill vs decode.** Prefill processes the whole prompt **in parallel** (the big triangle, compute‑bound, $O(T^2)$ scores) *and writes the cache*. Decode generates one token at a time (one row, $O(T)$ scores, **memory‑bandwidth‑bound**: the whole cache must be streamed from HBM for every token). ⏸
5. **The memory meter.** A bar under the drawers grows *linearly* with each token. Overlay a worked example (see Math). ⏸

**Math** (one equation per beat, bottom third):

$$
\underbrace{O(T^2 d)}_{\text{prefill scores, per layer}}\qquad
\underbrace{O(T d)}_{\text{one decode step at context }T}\qquad
\underbrace{2\,T\,H_{kv}\,d_h}_{\text{cached numbers per layer}}
$$

Worked meter example (computed from the public configs; verify before the talk):
- Llama‑2‑7B (MHA, 32 layers × 32 KV heads × 128) ≈ **0.5 MB per token** in fp16, so ≈ 2 GB at 4 k tokens.
- Llama‑3‑8B (GQA, 32 × 8 × 128) ≈ **128 KB per token**, so ≈ 16 GB at 128 k tokens.

Total attention work to generate $M$ tokens after a $T$‑token prompt is $O(MT + M^2)$, not $O(M)$.

**Manim notes:** two `Heatmap` grids. The left one uses `LaggedStart(*[FadeIn(tile) for tile in triangle], lag_ratio=0.01)`. The right one is one `Rectangle` row moving with `.animate.shift(RIGHT)` while a `Brace` labelled $T$ grows. The memory meter is a `Rectangle` stretched with `stretch_to_fit_width` via a `ValueTracker`. Badge: `PROPERTY`.
**⚠ Don't say:** "with a KV cache, decoding is O(1)". It is **O(T) per token in compute and O(T) in memory**. The cache removes *recomputation*, not the scan. And the KV cache is **not** prefill: prefill is the phase that *fills* it.

---

### Slide 4 ★ — Many heads: why Q, K, V are worth separating (2:00)

**Signature frame:** one token square shining through **three differently tinted prisms**. Each prism produces its own (q, k, v) triple and its own attention pattern over the same sentence. The three yellow outputs are concatenated and pass through a final mixing lens $W_O$.

**Beats**
1. **Thought experiment: one vector for everything.** Suppose a token uses the *same* vector as its query, key and value. Then a token's "what I want" equals its "how I'm found" equals "what I give". Animate the consequence: every token's dot product with itself is large, so attention collapses onto self and similar tokens, and **"Leo" can't ask for the *location* while advertising its *identity***. Cross out the single vector. ⏸
2. **Separation of roles.** Split it into three: **Q/K decide *where* to look (routing)**, and **V decides *what* is moved (content)**. A slogan appears: *"QK is the address bus, V is the data bus."* Show $QK^\top$ as a heatmap (routing) and $V$ as a column of payloads (content). Changing V leaves the heatmap unchanged, while changing Q/K moves it. ⏸
3. **Several heads = several questions at once.** Three prisms produce three heatmaps on the same sentence:
   - head 1 links *key → Leo*;
   - head 2 links *bridge → under*;
   - head 3 looks at the previous token.

   Label them **"illustrative possibilities"**. ⏸
4. **Concatenate and mix.** The three yellow outputs slot into one long vector, and $W_O$ projects it back to $d$ and **writes it into the residual stream** (preview of slide 20). ⏸
5. **Teaser for slide 21:** dim the K/V lenses of heads 2 and 3 and let them *share* head 1's K/V. "Do all heads really need their own keys?" (GQA/MQA.)

**Math**

$$
\text{head}_h = \operatorname{softmax}\!\Big(\tfrac{Q_hK_{g(h)}^\top}{\sqrt{d_h}}+\mathcal M\Big)V_{g(h)},\qquad
\operatorname{MHA}(R) = \operatorname{concat}_h(\text{head}_h)\,W_O
$$

Here $g(h)=h$ for MHA. The group map $g$ is introduced now so that GQA in slide 21 is a one‑symbol change.

**Manim notes:** three `IndexCardWall`s stacked with slight perspective offset (a `ThreeDSlide` with `phi=70°` works). Heatmaps are `Heatmap(n=8, data=TOY_PATTERNS[h])`. Watermark: `SCHEMATIC` on the head patterns.
**⚠ Don't say:** "head 7 is the coreference head". Heads are not guaranteed to map one‑to‑one onto semantic roles.

---

### Slide 5 ★ — "Bank": embeddings are a starting point, the residual stream is a journey (2:30)

**Signature frame:** 3D axes. Two arrows **start from exactly the same point**: the embedding $E[\text{bank}]$. Coloured layer updates are attached tip to tail. One path curls toward a *river* direction and the other toward a *finance* direction, while the camera slowly orbits.

**Beats**
1. **Lookup.** Two sentence lanes (top: *fish near the bank*, bottom: *money at the bank*). The token "bank" in both lanes is looked up in the same embedding table, a giant matrix $E$. The same row lights up, and two **identical** yellow arrows appear. Caption: *"Same token, same starting vector."* ⏸
2. **The conveyor belt.** Under each token a vertical white rail rises (depth goes up). Layer 1 (attention) adds a small blue arrow: in the top lane it is pulled by *fish*, in the bottom lane by *money*. The arrows are **added**, not replaced. ⏸
3. **Divergence.** Layers 2…L each add an arrow (blue for attention, gold for FFN). The two resultant vectors visibly separate. A `TracedPath` leaves a trail for each tip. ⏸
4. **The 3D reveal.** The camera orbits. Faint labelled directions appear: "river‑ish" and "finance‑ish". The final vectors are near different directions. Watermark: `SCHEMATIC — illustrative projection`. ⏸
5. **Callback to Q/K/V.** Zoom in on layer 7 of the bottom lane: the three lenses of slide 2 are fed **this contextualised vector**, not the raw embedding. "Attention at layer 7 asks questions with a vector that already knows it's a money bank."

**Math**

$$
r_t^{(0)} = E[\tau_t],\qquad r_t^{(\ell+1)} = r_t^{(\ell)} + u_t^{(\ell)},\qquad
r_t^{(L)} = E[\tau_t] + \sum_{\ell<L} u_t^{(\ell)}
$$

The last form is the key intuition: **the final representation is the embedding plus a sum of everything every layer wrote.**

**Manim notes:** `ThreeDSlide`, `ThreeDAxes`, `Arrow3D` for each update. Use `self.begin_ambient_camera_rotation(rate=0.08)` and **`self.next_slide(loop=True)`** so the orbit loops while you talk. Pin the two sentences with `add_fixed_in_frame_mobjects` so the viewer never loses which sense is which.
**⚠ Don't say:** "a position vector is added to the embedding". In RoPE models (Llama, Mistral, Qwen, …) position enters **inside the Q/K rotation**, not the residual. Show position as a side label only. Also, disambiguation doesn't have to happen at the "bank" position alone, and the "directions" are cartoons.

---

### Slide 6 ★ — The long‑context tax (1:30)

**Signature frame:** the attention triangle grows from 32×32 to 512×512 to a horizon that runs off the screen. Theo is dwarfed by his card wall and asks: *"What if my notebook never grew?"*

**Beats**
1. Zoom out continuously: the triangle's tile count is displayed as $T^2/2$. ⏸
2. Two asymptotic curves on mini‑axes: *prefill compute ∝ T²* and *KV memory ∝ T*. Badge: `PROPERTY — asymptotic illustration, not benchmark`. ⏸
3. **Bandwidth.** During decode, a "HBM → chip" pipe has to carry the **entire** wall for every new token. The pipe narrows as the wall grows (decode is memory‑bound). ⏸
4. Theo's thought bubble shows a notebook of fixed size. **Match‑cut** to Manu holding exactly that notebook.

**Manim notes:** `MovingCameraScene` behaviour (`ThreeDSlide` not needed). Animate `self.camera.frame.animate.scale(4)`. Use `Axes.plot` for both curves.
**⚠ Don't say:** "FlashAttention solves this". It reduces memory traffic, not the $T^2$ arithmetic. That comes back in slide 21.

---

## Act II — The notebook brothers (RNN → Mamba → Mamba‑2)

> **Narrative spine of the act:** *each fix creates the next problem.* RNN (sequential, gradients) → linear SSM / S4 (parallel but blind to content) → Mamba‑1 (selective, but the scan can't use tensor cores and the state stays small) → Mamba‑2 (scalar decay per head, so the work becomes matmuls).
> Put a persistent **"problem ladder"** on the left edge: each rung lights up as the slides go by.

### Slide 7 ★ — Recurrence came first: the RNN and its two problems (2:00)

**Signature frame:** a chain of dominoes labelled $h_1 \to h_2 \to \dots \to h_T$. Underneath, a faint backward pulse (the gradient) either **fades to nothing** or **flares to white** as it travels left.

**Beats**
1. **A short history card:** Elman RNN (1990), LSTM (Hochreiter & Schmidhuber 1997), GRU (Cho et al. 2014), seq2seq (Sutskever et al. 2014). Then the twist: **attention was invented to help RNNs** (Bahdanau et al. 2014), and the Transformer (2017) then removed the recurrence entirely. *"We're about to bring it back."* ⏸
2. **The streaming ideal.** Manu reads one card at a time and updates a fixed‑size vector $h_t$. Memory is constant, and the cost per token is constant. This is exactly Theo's wish. ⏸
3. **Problem 1: training is sequential.** Dominoes: $h_5$ can't be computed until $h_4$ falls. A GPU drawn as 1000 idle workers waits on one domino line. ⏸
4. **Problem 2: gradients multiply.** The backward pulse passes through each Jacobian. If each one shrinks it ($\lVert J\rVert<1$), the pulse vanishes; if each one grows it, the pulse explodes. LSTM and GRU gates are drawn as valves that partly relieve this. ⏸
5. **Why both problems come from the same place.** Highlight the $\tanh(W h_{t-1} + \dots)$. The **nonlinearity inside the recurrence** is what prevents both unrolling into a closed form and parallelising.

**Math**

$$
h_t = \tanh(W h_{t-1} + U x_t + b),\qquad
\frac{\partial h_T}{\partial h_s} = \prod_{t=s+1}^{T}\operatorname{diag}\!\big(1-h_t^2\big)\,W
$$

**Manim notes:** dominoes are `Rectangle`s rotated with `Rotate(..., about_point=bottom_corner)` in a `Succession`. The gradient pulse is a `Dot` with `set_opacity` driven by $\prod\lVert J\rVert$ via a `ValueTracker`.
**⚠ Don't say:** "RNNs can't process long sequences". They have been used on long sequences for decades. Vanishing/exploding gradients are a *risk* of optimisation, not an impossibility theorem.

---

### Slide 8 — Drop the nonlinearity: linear state spaces and S4 (1:30)

**Signature frame:** a **ripple tank**. Every arriving token drops a pebble, and the *same‑shaped* ripple (a fixed kernel $\bar K$) spreads to the right. The output is the sum of the ripples. A token labelled *"um"* and a token labelled *"Leo"* make **identical** ripples.

**Beats**
1. Remove the $\tanh$: $s_t = \bar A s_{t-1} + \bar B x_t$, $y_t = C s_t$. This is a **linear time‑invariant (LTI)** state space, the S4 family (Gu, Goel & Ré, ICLR 2022; HiPPO 2020). ⏸
2. **Unroll it:** because the dynamics are linear, the output is a **convolution** with kernel $\bar K = (C\bar B, C\bar A\bar B, C\bar A^2\bar B, \dots)$. Animate the kernel sliding over the sequence. Training is **parallel** (FFT, $O(T\log T)$), and inference is still a recurrence. Problem 1 is solved. ⏸
3. **Problem 3 appears.** The ripple shape doesn't depend on the input. "um" and "Leo" are written with the same strength and fade at the same rate. The model **can't choose** what to remember. The ladder rung reads "content‑blind".

**Math**

$$
y = \bar K * x,\qquad \bar K_j = C\bar A^{\,j}\bar B
$$

**Manim notes:** ripples are `Circle`s expanding with an `UpdateFromAlphaFunc`. The kernel is a small bar chart (`BarChart`) sliding with `.animate.shift`.

---

### Slide 9 ★ — Mamba‑1: selection (2:00)

**Signature frame:** Manu's notebook with a **pen‑pressure dial (Δ)** and an **eraser**. As tokens pass, the dial jumps: *"Leo"* → heavy ink and slow fade, *"um"* → almost no ink, *"ignore that"* → strong erase.

**Beats**
1. **Make the dynamics depend on the input.** $\Delta_t, B_t, C_t$ are now *functions of the current token*. Three dials on Manu's desk spin differently for each card. ⏸
2. **What Δ does (discretisation, zero‑order hold).** $\bar A_t = \exp(\Delta_t A)$ with $A<0$:
   - large $\Delta_t$ means "reset and focus on this token" (old ink fades, new ink is heavy);
   - small $\Delta_t$ means "ignore this token, keep the past".

   Animate $\exp(\Delta A)$ on a small plot as the dial turns. ⏸
3. **The catch: no more convolution.** The kernel now changes with every token, so the ripple trick of slide 8 is gone. Is training sequential again? ⏸
4. **The hardware‑aware parallel scan.** Each step is an affine map $s\mapsto \bar A_t s + \bar B_t x_t$, and **composing affine maps is associative**. Animate a **binary tree** that combines pairs in parallel: $(a_2,b_2)\circ(a_1,b_1) = (a_2a_1,\; a_2b_1+b_2)$, with depth $\log T$. Then show the systems trick: load the parameters into fast SRAM, discretise and scan there, write only $y$ back to slow HBM, and recompute in the backward pass. ⏸
5. Manu reads an endless ribbon at constant cost per card. **Decode is O(1) in T.** The ladder rung "content‑blind" is struck through.

**Math**

$$
s_t = \bar A_t\, s_{t-1} + \bar B_t\, x_t,\quad y_t = C_t^\top s_t,\qquad
\bar A_t = \exp(\Delta_t A),\ \ \bar B_t \approx \Delta_t B_t,\qquad A = \operatorname{diag}(a_1,\dots,a_N)
$$

with an $N$‑dimensional state per channel ($N=16$ in Mamba‑1), wrapped in a block made of in‑projection → short causal conv → SiLU → SSM → gate → out‑projection (drawn grey around the core).

**Manim notes:** the dials are `Arc` + `Line` needles rotated by `ValueTracker`s keyed to a toy Δ sequence. The scan tree is a `VGroup` of `Dot`s and `Line`s built level by level with `LaggedStart`. Badge: `SCHEMATIC` on the toy Δ values.
**⚠ Don't say:** "Mamba loops over tokens on the GPU". Training uses a **parallel** selective scan.

---

### Slide 10 ★ — Mamba‑1's problems (1:30)

**Signature frame:** a factory floor. On the left is a giant **stamping press** labelled *Tensor Cores (matmul)*, idle and gathering dust. On the right, a narrow conveyor of scan workers labelled *scalar / vector units* is jammed.

**Beats**
1. **The hardware fact.** A bar chart shows that on an A100, dense BF16 matmul runs at ≈312 TFLOP/s while non‑matmul FP32 runs at ≈19.5 TFLOP/s: **about 16×**. The selective scan is *not a matmul*, so it runs on the slow side. ⏸
2. **Consequence: the notebook stays small.** Manu's state per channel is only $N=16$ numbers. Making $N$ bigger makes the scan slower in proportion. ⏸
3. **Consequence: addressability.** Inject 4, 8, 16, 32 *person → place* bindings into one notebook. On the left, Theo's wall simply gets longer. On the right, Manu's page fills with **overlapping ink**. Ask "Noah?" and then "Leo?": Theo reweights distinct cards, while Manu has to decode both from the *same* page. A caption with a question mark: **"capacity / interference?"** ⏸

**Manim notes:** `BarChart` for the TFLOP/s values (`PROPERTY`, cite the NVIDIA A100 datasheet and the Mamba‑2 paper). The ink overlap is semi‑transparent strokes accumulating on a fixed `Rectangle`.
**⚠ Don't say:** "a fixed state must forget". At arbitrary precision it doesn't have to. The real issues are finite‑precision capacity, interference, and what training learns. Mamba *can* do associative recall; it is not "a summariser that can't recall".

---

### Slide 11 ★ — Mamba‑2: one knob per head (1:45)

**Signature frame:** a diagonal matrix of N different dials **melts into a single dial times the identity**, and the notebook immediately becomes a stack of square tiles that slide into the stamping press.

**Beats**
1. **The restriction.** $A_t=\operatorname{diag}(a_{t,1},\dots,a_{t,N})$ becomes $A_t=a_t I$. The N separate dials merge into one with `ReplacementTransform`. *"Every coordinate of this head's notebook fades at the same rate."* ⏸
2. **What this buys.** With a scalar decay, the recurrence can be rewritten as **matrix multiplications over chunks** (Act III proves it). The press starts up. N grows from 16 to **64–256** without the same cost. ⏸
3. **Other block changes** (small icons, no derivation):
   - $x, B, C, \Delta$ are all produced by **one parallel in‑projection** at the start of the block, which is friendly to tensor parallelism;
   - $B$ and $C$ are shared across heads, the SSM analogue of sharing K/V (the paper calls this a multi‑value / multi‑input pattern);
   - an extra norm before the out‑projection. ⏸
4. **The reported gain.** "SSD core: 2–8× faster than Mamba‑1's scan" (Dao & Gu 2024), shown as a citation card and clearly labelled *core‑layer measurement in their setup*, not an end‑to‑end claim. ⏸
5. The ladder now reads: sequential ✗ → content‑blind ✗ → slow scan / small state ✗. One new rung remains, unlit: *"addressability: still a fixed‑size state"*. Save it for Act V.

**Math**

$$
A_t^{\text{Mamba-1}}=\operatorname{diag}(a_{t,1},\dots,a_{t,N})\ \longrightarrow\ A_t^{\text{Mamba-2}}=a_t\,I_N,\qquad a_t=\exp(\Delta_t A),\ A<0
$$

**⚠ Don't say:** "Mamba‑2 is strictly more expressive". At fixed $N$ it is a *restriction*. The win is that it lets you afford a much larger $N$ and run on matmul hardware.

---

## Act III — Max's mathematics (SSD in detail)

> All identities in this act are `IDENTITY` badges, checked by `math_contracts/test_ssd_equivalence.py`.

### Slide 12 ★ — The state is an N×P sheet: fade, write, read (2:00)

**Signature frame:** a mint‑coloured grid $S_t$ with **N rows** (memory slots) and **P columns** (channels). Three animated operations happen on it: the whole sheet **dims** (×$a_t$), a **rank‑1 rectangle** is painted in by two thin vectors ($b_t$ down the side, $x_t$ across the top), and a **read vector** $c_t$ sweeps down the rows to produce the output $y_t$.

**Beats**
1. Show the sheet with `Brace`s: $N$ rows, $P$ columns. ⏸
2. **Fade:** $a_t S_{t-1}$. Every cell's opacity is multiplied by $a_t$ (a single scalar, which is the point of slide 11). ⏸
3. **Write:** $b_t x_t^\top$. A purple column vector $b_t$ (N) and a yellow row vector $x_t$ (P) sweep out an outer‑product rectangle and **add** it onto the sheet. Use the cell colour intensity $= b_{t,n}x_{t,p}$. ⏸
4. **Read:** $y_t = S_t^\top c_t$. A cyan column $c_t$ dots against each column of the sheet and produces the $P$‑vector $y_t$. ⏸
5. **Rename for comparison** (persistent lower‑third): $b_t \to k_t$ (key / write address), $c_t\to q_t$ (query / read address), $x_t\to v_t$ (value). *"Fade, write with a key, read with a query."* The audience should already feel the attention echo.

**Math**

$$
\underbrace{S_t}_{N\times P} = \underbrace{a_t}_{\text{scalar}}\,\underbrace{S_{t-1}}_{N\times P} + \underbrace{b_t}_{N\times 1}\underbrace{x_t^\top}_{1\times P},\qquad
\underbrace{y_t}_{P\times 1} = S_t^\top \underbrace{c_t}_{N\times 1}
$$

**Manim notes:** `StateSheet(N=4, P=6)` is a grid of `Square`s whose `fill_opacity` is bound by `always_redraw` to a NumPy array updated per beat. The outer product uses `LaggedStart(*[cell.animate.set_fill(...)])` in row‑major order. Keep $q/k/v$ labels **pinned in the same screen locations** for slides 12–16.
**Aside (one line on screen):** this is a **fast‑weight memory**: an outer‑product associative matrix (Schmidhuber 1992; Schlag et al. 2021).
**⚠ Don't say:** "$x_t$ is the token embedding". It is the SSD's inner input, produced after projections and conv.

---

### Slide 13 ★ — Unroll the notebook: it's a weighted sum over the past (2:00)

**Signature frame:** the current sheet $S_t$ splits into **translucent ghost layers**, one per earlier write. Each ghost is veiled with opacity equal to its **survival product** $\prod a_u$. The query beam reads all the ghosts at once, and **Theo's attention picture from slide 2 slides in beside it**.

**Beats**
1. Expand three steps by hand. Each new line keeps the earlier terms anchored and multiplies them by the next $a$ (`TransformMatchingTex`):
   $S_1 = b_1x_1^\top$ ⟶ $S_2 = a_2 b_1x_1^\top + b_2x_2^\top$ ⟶ $S_3 = a_3a_2\, b_1x_1^\top + a_3\, b_2x_2^\top + b_3x_3^\top$. ⏸
2. **General form:** the ghost stack. The veil on ghost $s$ is $\prod_{u=s+1}^{t}a_u$, driven by a `ValueTracker` so you can scrub through time live. ⏸
3. **Read with $q_t$:** distribute the dot product into the sum. The boxed formula assembles, with three labelled underbraces. ⏸
4. **Side by side with attention:** left is $\sum_s \alpha_{ts}v_s$ (softmax weights); right is $\sum_s L_{ts}(q_t^\top k_s)\,v_s$. Point out the **two differences**: no softmax normalisation, and an extra **decay factor**. ⏸

**Math**

$$
S_t = \sum_{s=1}^{t}\Big(\prod_{u=s+1}^{t}a_u\Big)\,k_s v_s^\top
\quad\Longrightarrow\quad
\boxed{\,y_t = \sum_{s=1}^{t}\underbrace{\Big[\prod_{u=s+1}^{t}a_u\Big]}_{\text{survival}}\ \underbrace{(q_t^\top k_s)}_{\text{match}}\ \underbrace{v_s}_{\text{content}}\,}
$$

(With $S_0\neq 0$ there is an extra term $(\prod_{u\le t}a_u)\,S_0^\top q_t$. The test suite checks it.)

**⚠ Don't say:** "Mamba‑2 stores the past tokens and looks them up". This is an **algebraic decomposition**. The runtime stores only $S_t$. The coefficients are signed and not normalised.

---

### Slide 14 — The matrix form: two heatmaps that rhyme (2:00)

**Signature frame:** two $T\times T$ lower‑triangular heatmaps side by side. Left: blue $\operatorname{softmax}(QK^\top)$. Right: mint $L\odot QK^\top$. A shared row pointer scans both. Above the right one, the **decay mask $L$** fills row by row with live numbers.

**Beats**
1. Stack the per‑token vectors into $Q, K\in\mathbb R^{T\times N}$, $V\in\mathbb R^{T\times P}$ (rows slide into matrices). ⏸
2. **Build $L$ live** on a 4×4 example with $a_2=.8,\ a_3=.5,\ a_4=.75$. The diagonal is all 1s, the upper triangle is exactly 0, and each lower entry is the product of the $a$s *between* source and target. Pause on $L_{4,2}=a_3a_4=.375$ and $L_{4,1}=a_2a_3a_4=.3$ (both checked by `test_on_screen_4x4_mask`). ⏸
3. $M = L\odot QK^\top$, then $Y=MV$. Morph the 4×4 into a 16×16 heatmap. ⏸
4. **The rhyme and the differences.** Labels appear one after the other:
   - **"SSD ≠ softmax attention"** (no normalisation, signed);
   - **"same algebra ≠ same memory"** (Theo stores K/V for all T; Max stores one $N\times P$ sheet).

**Math**

$$
L_{ts}=\begin{cases}\prod_{u=s+1}^{t}a_u & s\le t\\ 0 & s>t\end{cases},\qquad
\boxed{\,Y=\big(L\odot QK^\top\big)V\,}
\qquad
L=\begin{bmatrix}1&0&0&0\\ .8&1&0&0\\ .4&.5&1&0\\ .3&.375&.75&1\end{bmatrix}
$$

**Manim notes:** `DecimalMatrix` for $L$, with entries revealed by `LaggedStart(*[Write(e) for e in row])`, one row per `⏸`. While each entry appears, highlight the contributing $a_u$ in the sequence strip above with `Indicate`. Heatmaps use `Heatmap(n=16, data=...)` coloured by a diverging map (mint positive, coral negative).

---

### Slide 15 — Why a T×T matrix can cost O(T): semiseparable structure (1:45)

**Signature frame:** the big mint heatmap is cut by **scissors** into $C\times C$ chunks. Diagonal blocks stay dense. **Each off‑diagonal block folds like paper into two skinny strips** ($C\times N$ and $N\times C$) joined by a single scalar bead $\gamma$.

**Beats**
1. **Chunk it.** Grid lines every $C$ tokens. Count the blocks: $(T/C)^2/2$. ⏸
2. **Zoom into one off‑diagonal block** (later chunk $I$, earlier chunk $J$). Split the survival product of any entry into three parts: *from the start of $I$ to $t$*, *across the gap between chunks*, and *from $s$ to the end of $J$*. Colour the three segments of the sequence strip differently. ⏸
3. **Fold.** The block equals $\gamma_{IJ}\,\operatorname{diag}(u_I)\,Q_I K_J^\top\,\operatorname{diag}(w_J)$, which is a product of a $C\times N$ and an $N\times C$ matrix. **Rank ≤ N.** The paper‑fold animation compresses the block into its two thin factors. ⏸
4. Name it: **$L$ is 1‑semiseparable, and $M$ is $N$‑semiseparable** (every block below the diagonal has rank ≤ N). Checked by `test_off_diagonal_block_rank_at_most_N`.

**Math**

$$
L_{ts}=\underbrace{\prod_{u=b_I}^{t}a_u}_{u_t}\;\underbrace{\prod_{u=e_J+1}^{b_I-1}a_u}_{\gamma_{IJ}}\;\underbrace{\prod_{u=s+1}^{e_J}a_u}_{w_s},
\qquad
M_{IJ}=\gamma_{IJ}\operatorname{diag}(u_I)\,Q_I\,K_J^\top\operatorname{diag}(w_J)
$$

($b_I$ is the first index of chunk $I$, $e_J$ the last index of chunk $J$; empty products are 1.)

**Manim notes:** scissors are an `SVGMobject` (original art) moving along `DashedLine`s. The fold is `Transform(block, VGroup(tall_strip, bead, wide_strip))`.

---

### Slide 16 ★ — The four SSD steps and the complexity bill (2:00)

**Signature frame:** a horizontal sequence split into $T/C$ chunks, each a mint box. **Four coloured passes** run over it: (1) green fills inside each box in parallel; (2) each box emits a small state sheet; (3) the sheets hand off left‑to‑right in a relay; (4) each box receives the incoming sheet and adds a correction. A cost table builds below.

**Beats**
1. **Intra‑chunk (diagonal blocks):** every chunk computes $(L\odot QK^\top)V$ **locally**, all at once, as tensor‑core matmuls. They assume the incoming state is zero. ⏸
2. **Chunk states:** each chunk summarises its own tokens into an $N\times P$ sheet (also a matmul). ⏸
3. **State passing:** a short **scan over $T/C$ chunks** (not $T$ tokens) carries the true state across boundaries: $S^{\text{in}}_{I+1} = (\prod_{u\in I}a_u)\,S^{\text{in}}_I + S^{\text{local}}_I$. ⏸
4. **Output correction:** each chunk reads the incoming state with its queries, decayed to each position, and adds it to step 1. Arrows show that the first token's contribution reaches the last output. ⏸
5. **The bill:** build the complexity table. Then show *both* runtimes converging to the **same output vector** for a toy input (live numbers from `ssd_chunked` vs `ssd_recurrent`, `test_chunked_equals_recurrent`).

**Math (one SSD head, core only)**

| | Cost | Note |
|---|---|---|
| Training / prefill (chunked) | $O\big(T\,[\,NP + C(N+P)\,]\big)$ | linear in $T$ for fixed $N,P,C$; mostly matmuls |
| Decode, per token | $O(NP)$ | constant in $T$ |
| State memory | $O(NP)$ (+ tiny conv buffer) | constant in $T$ |
| Compare: attention prefill | $O(T^2 d)$ | |
| Compare: attention decode | $O(Td)$ per token, $O(T)$ KV memory | |

**Manim notes:** four passes in four colours (green, purple, white relay, cyan), each a `LaggedStart` over the chunk boxes; step 3 must visibly be **sequential over chunks** (`Succession`). Toy numbers come from `math_contracts` imported at render time so the screen *cannot* disagree with the test.
**⚠ Don't say:** "SSD removes arithmetic". It **reorganises** the work into matmul‑shaped pieces. Projections, conv and gates cost extra. Wall‑clock gains depend on kernel and hardware.

---

## Act IV — The clinic, the conveyor belt, and real Transformers

### Slide 17 ★ — Why do we need FFNs, and why are they the expensive part? (2:15)

**Signature frame:** two halves of a layer. On the left, **mixing**: yellow arrows fly *between* token rails (attention or SSM). On the right, **thinking**: each token's vector, alone, enters a **refinery**. It expands into a wide bank of ~4d neurons, some of which light up, then contracts back to $d$.

**Beats**
1. **What mixing alone can't do.** Attention's output is a weighted sum of values, which is **linear in V**, with weights computed from Q/K. Stack only mixers and you mostly re‑blend vectors. Something has to apply a *nonlinear, per‑token* transformation: "now that I know this is a money bank, recall facts about banks." ⏸
2. **The FFN as a memory of the training set** (Geva et al. 2021). Draw $W_{\text{up}}$ as a stack of **key rows** and $W_{\text{down}}$ as **value columns**. The token dots with every key row, a few neurons fire (the activation is the "match"), and their value columns are **summed** into the output.
   Callback: *"Attention is a key‑value memory over the **context**; the FFN is a key‑value memory over the **training data**."* Theo's wall and the refinery's rows are drawn in the same visual style. ⏸
3. **The parameter bill.** A per‑layer pie chart:
   - attention projections $W_Q,W_K,W_V,W_O$: $4d^2$;
   - FFN: $8d^2$ (either $2\cdot d\cdot 4d$, or SwiGLU $3\cdot d\cdot\tfrac{8}{3}d$).

   **The FFN is about ⅔ of the layer.** FLOPs per token follow parameters (≈2 FLOPs per weight), so at short context the FFN is about ⅔ of the compute as well. ⏸
4. **The crossover** (honest caveat). For one decode step at context $T$, attention scores plus value mixing cost ≈ $4Td$ FLOPs, against $16d^2$ for the FFN. They are equal at **$T\approx 4d$** (about 16 k tokens for $d=4096$). Show a slider on $T$ where the pie flips. *"The FFN dominates until the context gets long, and then the KV wall takes over."* ⏸
5. **The question that drives MoE:** *"Why does every token have to pay for the whole refinery?"*

**Math**

$$
\operatorname{FFN}(r)=W_{\text{down}}\,\sigma(W_{\text{up}}r),\qquad
\operatorname{SwiGLU}(r)=W_{\text{down}}\big[\operatorname{SiLU}(W_{\text{gate}}r)\odot W_{\text{up}}r\big]
$$

$$
\operatorname{FFN}(r)=\sum_{i=1}^{d_{ff}}\underbrace{\sigma(\mathbf k_i^\top r)}_{\text{match to key } i}\;\mathbf v_i
\qquad(\mathbf k_i=\text{row } i \text{ of } W_{\text{up}},\ \mathbf v_i=\text{column } i \text{ of } W_{\text{down}})
$$

**Manim notes:** the refinery is a `NeuralLayer` of `Circle`s (d → 4d → d). Neurons glow by `set_fill(opacity=act)`. The pie is a `VGroup` of `AnnularSector`s whose angles are bound to a `ValueTracker(T)`.
**⚠ Don't say:** "attention retrieves, the FFN stores facts". That is a useful first cut, but both can do both. And "FFN is always the bottleneck" is false at long context.

---

### Slide 18 ★ — Mixture‑of‑Experts: the specialist clinic (2:30)

**Primary metaphor: a hospital.** Every token is a **patient** carrying its chart (the residual vector). A **triage nurse** (the router) reads the chart and refers the patient to the **top‑k specialist rooms** (experts, each an FFN). The specialists' notes are **weighted by the nurse's confidence** and written back into the chart. Rooms that are closed use no electricity (no FLOPs), **but they still occupy floor space** (all expert weights sit in GPU memory).

**Signature frame:** a hospital corridor of 8 doors seen from above, with a gold triage desk at the entrance. A "bank (money)" patient walks in, two doors light up, two gold paths go in and come back, and the notes are merged onto the chart. Behind it, a "bank (river)" patient takes **different doors**.

**Beats**
1. **Split the refinery.** The big FFN from slide 17 slides apart into 8 smaller rooms. ⏸
2. **Triage.** The router computes a score per room ($z = W_r r$) and draws them as gold bars over the doors. **Top‑2** doors open. ⏸
3. **Treatment and merge.** Two paths light up, two outputs come back, and they are weighted by $g_e$ and **added to the residual**. The other 6 doors stay dark. ⏸
4. **A second patient, a different route.** Show that routing is **per token, per layer**: the same word can see different rooms at different depths. ⏸
5. **The GP everyone sees: shared experts.** Add one always‑open room (DeepSeekMoE style) for common knowledge, so specialists don't each have to relearn it. ⏸
6. **Sub‑specialties: fine‑grained experts.** Split each room into 4 smaller ones and open 4× as many. The cost stays the same, but the **number of possible referral combinations explodes**: $\binom{16}{2}=120 \to \binom{64}{8}\approx 4.4\times10^9$ (the DeepSeekMoE argument). ⏸
7. **The bill:** a three‑counter overlay: **stored** experts (memory), **active** experts per token (FLOPs), **dispatched** tokens (communication).

**Math**

$$
z=W_r\,r,\qquad \mathcal I(r)=\operatorname{TopK}(z,k),\qquad
\operatorname{MoE}(r)=\sum_{e\in\mathcal I(r)} g_e(r)\,\operatorname{FFN}_e(r)\;+\;\sum_{j\in\mathcal S}\operatorname{FFN}^{\text{shared}}_j(r)
$$

with $g$ a softmax (or sigmoid, as in DeepSeek‑V3) renormalised over the selected experts.

**Secondary metaphors (one beat each, pick per audience):**
- **Orchestra:** a full orchestra is stored on stage, but in each bar only a few sections play. Capacity is large while the sound per bar is cheap.
- **Ambulance hub:** shown in slide 19 for all‑to‑all.

**Manim notes:** `Clinic(n_experts=8, k=2)` (Appendix A). Router bars are `BarChart` with `TOY` weights. Paths are `CubicBezier`s with `ShowPassingFlash` for "in flight". The combination counter is an `Integer` that animates its value.
**⚠ Don't say:** "expert 3 is the finance expert". Experts are *learned*, distributed and not semantically pure. Colour them by abstract glyphs, never by topic labels. MoE selects FFNs **inside** one model; it is not an ensemble of separate LLMs.

---

### Slide 19 — How modern LLMs actually run MoE (2:00)

**Signature frame:** a map of several **hospital buildings (GPUs)**, each hosting some of the specialist rooms. Patients are put into **ambulances** at a central hub, driven to whichever building holds their specialists, treated, and driven back. An administrator watches occupancy dials over every room.

**Beats**
1. **The popularity problem.** Without balancing, 100 patients enter and most go to room 3. A queue forms while the other rooms are idle (**routing collapse**). ⏸
2. **Capacity and dropping** (Switch Transformer). Each room has a fixed number of seats (the *capacity factor*). Overflow patients **skip treatment and pass through unchanged on the residual belt**. This is where the residual connection saves the day. ⏸
3. **Balancing strategies over time:**
   - auxiliary load‑balancing loss (Shazeer 2017; GShard 2020; Switch 2021);
   - **auxiliary‑loss‑free balancing** (DeepSeek‑V3): a per‑expert *bias* is added only for *selection*, and it is nudged up for under‑used rooms and down for over‑used ones. Animate the administrator turning small dials. ⏸
4. **Expert parallelism and all‑to‑all.** Ambulances: every GPU sends each token to the GPU that owns its experts, and they all come back. Draw the two all‑to‑all phases as ambulance waves. **Communication is a real cost.** DeepSeek‑V3 limits each token to a few nodes to keep ambulance trips short. ⏸
5. **Why decode is memory‑bound.** At small batch, each token opens different rooms, so the GPU must **read many experts' weights from HBM for very few FLOPs**. Larger batches amortise this. ⏸
6. **Timeline strip** (cite; numbers marked *verify against the model card*):

| Year | System | Routing idea | Scale (verify) |
|---|---|---|---|
| 2017 | Sparsely‑Gated MoE (Shazeer et al.) | noisy top‑k, aux loss | LSTM‑era, up to 137 B |
| 2020 | GShard (Lepikhin et al.) | top‑2, expert parallelism | 600 B translation |
| 2021 | Switch Transformer (Fedus et al.) | **top‑1**, capacity factor, dropping | up to 1.6 T |
| 2023/24 | **Mixtral 8×7B** (Mistral AI) | top‑2 of 8 | 47 B total / 13 B active |
| 2024 | **DeepSeekMoE** | fine‑grained + shared experts | 16 B |
| 2024 | **DeepSeek‑V3** | 256 routed + 1 shared, top‑8, sigmoid gate, aux‑loss‑free bias, node‑limited routing | 671 B / 37 B active |
| 2025 | Qwen3‑MoE, Llama 4, gpt‑oss | many experts, small top‑k (with or without shared expert) | see model cards |

**Manim notes:** GPUs are rounded `Rectangle`s in a 2×2 grid. Ambulances are small `Triangle`s moving along `ArcBetweenPoints`. Occupancy dials reuse the `Dial` mobject from slide 9.
**⚠ Don't say:** "top‑2 of 8 means ¼ of the FLOPs". It means 2 expert FFNs per token versus 8 stored. Whether that is cheaper than a dense baseline depends on what you compare at **matched** stored or active parameters.

---

### Slide 20 ★ — The residual stream: how every layer talks to every other (2:30)

**Signature frame:** a tall white **conveyor belt / shared whiteboard** under each token. Every layer (blue attention, coral/mint SSM, gold FFN/MoE) **reads** by projecting a slice of the belt through its own lens and **writes** by adding a coloured arrow back. At the top, the unembedding reads the final sum.

**Beats**
1. **Write = add.** The belt carries $r^{(\ell)}$. Each block reads $\operatorname{Norm}(r^{(\ell)})$ and *adds* its output. Nothing is ever overwritten; later layers can only add (or add a cancelling vector). ⏸
2. **The sum of all writes.** Unroll the belt: $r^{(L)}_t = E[\tau_t] + \sum_\ell u^{(\ell)}_t$. Each arrow is still visible in the stack. *"The residual stream is a shared communication channel"* (Elhage et al. 2021). ⏸
3. **Read = project.** A layer's $W_Q$, $W_K$, $W_V$, $W_{\text{up}}$, $B$, $C$, $\Delta$ are all **linear reads** from the belt. Two layers communicate when one writes into a subspace that the other reads. Animate a 2D subspace "slot" lighting up in layer 3's write and layer 9's read. ⏸
4. **Why this enables interpretability.** Because writes add linearly, we can:
   - apply the **unembedding to any intermediate belt** (the *logit lens*) and watch "bridge" become more likely layer by layer;
   - attribute logits to individual writes (*direct logit attribution*). ⏸
5. **More features than dimensions.** $d$ is finite but the model wants many more features. A toy inset (Elhage et al. 2022): 5 features in 2D arranged as a pentagon, which is **superposition**. *"The belt is crowded, so features share directions."* ⏸
6. **Two axes, two memories (key hybrid picture).** Horizontal mint arrows show the **Mamba state $S_t^{(\ell)}$ carrying information across time within a layer**. Vertical white arrows show the **residual $r_t^{(\ell)}$ carrying information up through depth**. An attention layer reaches **sideways into earlier positions' residual‑derived K/V**. Draw all three kinds of arrows on one grid. ⏸
7. **A real hybrid stack.** Assemble block glyphs in an interleave pattern (for example M E M E A E M E …, M = Mamba‑2, A = attention, E = MoE/FFN). Grounded examples (verify the layer order from the checkpoint config before animating it literally):
   - **Jamba** (AI21, 2024): one attention layer per 8, MoE every other layer;
   - **Nemotron‑H** (NVIDIA, 2025): mostly Mamba‑2 with a few attention layers;
   - **Nemotron 3 Nano 30B‑A3B**: Mamba‑2 + GQA + MoE hybrid;
   - also Zamba, Samba, Falcon‑H1, and Qwen3‑Next (linear attention + full attention), as a "the field is converging on hybrids" collage.

**Math**

$$
r^{(\ell+1)}_t = r^{(\ell)}_t + F_\ell\!\big(\operatorname{Norm}(r^{(\ell)})\big)_t,\qquad
r^{(L)}_t = E[\tau_t] + \sum_{\ell=0}^{L-1}u^{(\ell)}_t,\qquad
\text{logit-lens}_\ell = W_U\,\operatorname{Norm}(r^{(\ell)}_t)
$$

**Manim notes:** `ResidualRail(n_layers=12)` per token. Reads are translucent `Polygon` "lenses" attached to the rail. Writes are `Arrow`s that animate `GrowArrow` then fold into the running sum. The logit‑lens readout is a mini `BarChart` (labelled `SCHEMATIC`) at the side of the rail, updating per layer.
**⚠ Don't say:** "the Mamba state *is* the residual stream". The state moves across **time** inside one layer; the residual moves across **depth** at one position.

---

### Slide 21 — "Transformers aren't that naive": the real toolbox (2:30)

**Signature frame:** a giant attention heatmap plus a KV‑cache wall, carved down **panel by panel**. Each technique is a new tool that cuts away a different part. Under each panel sit three small bars that show **what it shrinks**: prefill FLOPs, decode work, KV memory.

| Technique | Introduced by | Flagship use | Animation | Shrinks |
|---|---|---|---|---|
| **Multi‑Query Attention (MQA)** | Shazeer, *Fast Transformer Decoding* (2019) | PaLM, Falcon | all query heads share **one** K/V card stack; drawers collapse | KV memory and bandwidth ↓↓ |
| **Grouped‑Query Attention (GQA)** | Ainslie et al., EMNLP 2023 | Llama‑2‑70B, Llama 3, Mistral, Qwen | 8 query heads share 2 K/V groups (the $g(h)$ from slide 4) | KV memory ↓ (tunable) |
| **Sliding‑window attention** | Longformer (Beltagy et al. 2020); Child et al. 2019 sparse patterns | **Mistral 7B** (window 4096, rolling‑buffer cache) | the triangle becomes a **band**; old cards slide out of a rolling buffer | prefill $T^2\to TW$, KV $T\to W$ |
| **Interleaved local/global** | Gemma 2 (1:1), Gemma 3 (5 local : 1 global) | Gemma family; gpt‑oss alternates banded/dense (verify) | layers alternate band / full triangle, stacked in depth | KV ↓ on most layers, keeps some global reach |
| **Attention sinks / StreamingLLM** | Xiao et al. 2023 (ICLR 2024) | streaming deployments; learned sinks in some models | first few cards stay pinned while the window slides | stable infinite‑stream decoding |
| **Multi‑head Latent Attention (MLA)** | DeepSeek‑V2 (2024) | DeepSeek‑V2/V3/R1, Kimi K2 (verify) | each K/V card is **compressed into a small latent card** and expanded on read | KV memory ↓↓ while keeping per‑token history |
| **FlashAttention 1/2/3** | Dao et al. 2022; Dao 2023; Shah et al. 2024 | ubiquitous | the huge matrix is never materialised; tiles stream through a small SRAM box with an online softmax | memory traffic and activation memory ↓; **FLOPs unchanged ($T^2$)** |
| **PagedAttention / vLLM** | Kwon et al., SOSP 2023 | vLLM, most serving stacks | logical card rows mapped onto non‑contiguous physical pages, like OS virtual memory | fragmentation ↓, batch size ↑ |
| **Sparse / learned‑sparse attention** | BigBird (Zaheer et al. 2020); **NSA** (DeepSeek, 2025) | long‑context research; DeepSeek sparse attention variants (verify) | a few global and selected blocks light up in the band | prefill/decode work ↓ for long T |
| **RoPE + context extension** | RoPE (Su et al. 2021); Position Interpolation (Chen et al. 2023); YaRN (Peng et al. 2023) | Llama, Qwen, Mistral | Q/K vectors rotate by angle ∝ position; stretching the rotation frequencies extends the context | enables long context (not a cost reduction) |

**Beats:** one `⏸` per row, with a running tally of the three bars. End on a summary caption: **"These change the *cost* of keeping the card wall. GQA, MQA, MLA and Flash keep every card addressable. Windows give up distant cards."** Then: *"Hybrids take a different route: replace most walls with notebooks."*

**Manim notes:** one `Heatmap` reused and masked through each panel via `Transform` of its mask. The KV wall is `IndexCardWall` with `merge_heads(groups=2)`. Flash tiles use a small `SurroundingRectangle` "SRAM" box sweeping the matrix in tile order.
**⚠ Don't say:** "FlashAttention makes attention subquadratic". It doesn't: it is **exact** attention with less memory IO. And GQA still caches **O(T)** positions.

---

## Act V — Theory, the tunnel, and mechanisms

### Slide 22 ★ — The efficiency–expressivity frontier (2:00)

**Signature frame:** a 2D conceptual map.
- **x axis:** *memory the mixer keeps as context grows*, from constant to linear.
- **y axis:** *direct addressability of individual past tokens*.
- **Placed glyphs:** full attention (top right), sliding window (middle), Mamba‑1/Mamba‑2 (bottom left), hybrids (in between, near the top‑left "sweet spot").

The MoE clinic then **rises out of the page as a third axis**: stored capacity vs active compute.

**Beats**
1. Place the glyphs one at a time with qualitative arrows. Badge: `SCHEMATIC — qualitative positions, not benchmarks`. ⏸
2. **Evidence cards** fly in and pin to regions of the map (all `PROPERTY`, cite on screen):
   - **Recall needs state.** *Zoology* (Arora et al., ICLR 2024): multi‑query associative recall (MQAR) explains most of the quality gap between attention and gated‑convolution/SSM models. *Based* (Arora et al., ICML 2024): a **recall–throughput trade‑off**, where state size buys recall. Draw a schematic Pareto curve.
   - **Copying needs state that grows.** *Repeat After Me* (Jelassi et al., ICML 2024): a fixed‑state model can't copy arbitrarily long strings, while a small Transformer can.
   - **State tracking is hard for both.** *The Illusion of State* (Merrill, Petty & Sabharwal, ICML 2024): standard Transformers and common SSM parameterisations sit in the same low circuit class (TC⁰), so neither can track composed permutations in general. Grazzi et al. (ICLR 2025): allowing **negative** decay eigenvalues (here, $a_t<0$) lets linear RNNs solve parity. The notebook "flip" is a cute micro‑animation.
   - **Empirically, hybrids win.** Waleffe et al. (NVIDIA, 2024): an 8B Mamba‑2‑hybrid with a few attention layers matched or beat the Transformer baseline on the reported tasks.
   - **Theory of hybrids.** Cooper et al. (ICML 2026, *verify*): expressivity/efficiency separations for specific task families. ⏸
3. **The sequence‑mixer complexity table** (big‑O in $T$ only, everything else held fixed):

| Mixer | Prefill vs T | Decode per token vs T | Memory vs T | Addressability |
|---|---:|---:|---:|---|
| Dense softmax attention | $T^2$ | $T$ | $T$ | explicit per token |
| GQA / MQA / MLA | $T^2$ | $T$ | $T$ (smaller constant) | explicit |
| Sliding window $W$ | $TW$ | $W$ | $W$ | inside the window |
| Mamba‑1 / Mamba‑2 | $T$ | $1$ | $1$ | implicit, through the state |
| Hybrid | mix | attention layers $T$, SSM layers $1$ | KV for attention layers only | mixed |

*"O(1)" means constant in sequence length, not free.*

**Manim notes:** `Axes` with no ticks, glyphs are the brothers' props in miniature, and the third axis is a `ThreeDSlide` camera tilt (`self.move_camera(phi=60*DEGREES)`) as the clinic rises.
**⚠ Don't say:** "hybrids dominate". Results are **task‑family‑specific**.

---

### Slide 23 — Towards universality: two meanings of one word (2:00)

**Signature frame:** a split screen.
- **Left (A):** a "function landscape" that one architecture covers completely like a blanket. This is *expressive* universality.
- **Right (B):** three **constellations of features** (Theo's, Manu's, Max's) that, after a rotation, **line up partly**. This is *mechanistic* universality.

**Beats**
1. **(A) Expressive universality.**
   - Transformers are universal approximators of continuous sequence‑to‑sequence functions on compact domains (Yun et al., ICLR 2020).
   - RNNs (Siegelmann & Sontag 1995) and Transformers (Pérez et al. 2021) are Turing‑complete *under idealised assumptions* (unbounded precision or steps).
   - Caveats in grey: fixed size, finite precision, learnability, and depth are not covered. ⏸
2. **(B) Mechanistic universality: the hypothesis.** Do different networks trained on similar data **discover the same features and circuits**?
   - Lineage: Li et al. 2015 (convergent learning); Olah et al. 2020 (*Zoom In*: the universality claim); Kornblith et al. 2019 (CKA similarity).
   - Huh et al. 2024 (*Platonic Representation Hypothesis*, ICML) at the scale of whole representations. ⏸
3. **Across *these* architectures.** *Towards Universality* (Wang et al., ICLR 2025):
   - SAE features in Transformers and Mamba are largely similar;
   - **induction circuits** are structurally analogous;
   - Mamba shows an **"off‑by‑one" motif**: information is written one position later, attributed to its short convolution.

   Animate this as a one‑step‑delayed arrow in Manu's induction circuit. Also: Paulo et al. 2024 (*Does Transformer Interpretability Transfer to RNNs?*): steering, the tuned lens and similar tools largely carry over to Mamba and RWKV. ⏸
4. **The sharp question:** *"Can we find the same computational variable carried by different memories with the same causal consequences?"* That is what slide 25 is designed to test.

**Manim notes:** constellations are `Dot` clouds with `Line`s between matched features. Animate `Rotate` of one cloud until the matched pairs overlap (Procrustes alignment as a visual). Badge: `SCHEMATIC`.
**⚠ Don't say:** "Mamba and Transformers learn the same algorithm". Feature similarity is **not** identity of mechanism.

---

### Slide 24 ★ — The tunnel: deriving the attention form of Mamba‑2 from both sides (3:30)

> **The showpiece.** In Act III, Max dug from the **recurrence side**: he unrolled his notebook and found a matrix. Now Theo digs from the **attention side**. *Two tunnel crews start at opposite sides of a mountain. Do they meet?*

**Signature frame:** a mountain cross‑section. The left tunnel is labelled with the recurrence $S_t = a_tS_{t-1}+k_tv_t^\top$, the right tunnel with $\operatorname{softmax}(QK^\top)V$. Each excavation step is an equation, and **the two bores meet at one glowing boxed identity**: $Y=(L\odot QK^\top)V$.

**Beats: the right tunnel (Theo digs), one equation per ⏸**

1. **Start from causal softmax attention.** Row $t$:
   $$o_t=\frac{\sum_{s\le t}\exp(q_t^\top k_s)\,v_s}{\sum_{s\le t}\exp(q_t^\top k_s)}$$ ⏸
2. **Replace the exponential by a kernel feature map** (Katharopoulos et al. 2020, *Transformers are RNNs*): $\exp(q^\top k)\approx\phi(q)^\top\phi(k)$. For the slide, take $\phi=\mathrm{id}$ and drop the normaliser (Mamba‑2 also has no softmax denominator):
   $$o_t=\sum_{s\le t}(q_t^\top k_s)\,v_s\qquad\Longleftrightarrow\qquad O=\operatorname{tril}(QK^\top)\,V$$ ⏸
3. **Associativity: move the brackets.** Animate the parentheses physically hopping:
   $$o_t^\top=\sum_{s\le t}q_t^\top k_s\,v_s^\top=q_t^\top\underbrace{\Big(\sum_{s\le t}k_sv_s^\top\Big)}_{S_t\ \in\ \mathbb R^{N\times P}}$$
   *The $T\times T$ matrix was never needed. Theo's whole card wall collapses into one $N\times P$ sheet.* ⏸
4. **The prefix sum is a recurrence.** $S_t=S_{t-1}+k_tv_t^\top$, so **linear attention is an RNN with a matrix state** (checked: `test_linear_attention_is_ssd_with_no_decay`). The problem: nothing ever fades, so the sheet saturates. ⏸
5. **Add forgetting.** Multiply the past by a decay $a_t$ before each write. The causal mask of ones becomes the survival mask $L$:
   $$S_t=a_tS_{t-1}+k_tv_t^\top\quad\Longleftrightarrow\quad Y=(L\odot QK^\top)V$$
   **The bores meet.** The right tunnel's last equation is exactly the left tunnel's result from slides 13–14. Flash, then the boxed identity with badge `IDENTITY` (`test_recurrent_equals_matrix`). ⏸
6. **Live proof by numbers.** A toy $T=6$, $N=2$, $P=3$ input, with outputs from both routes printed side by side (imported from `math_contracts` at render time). They match to $10^{-10}$. ⏸

**Beat 7: the family portrait** (the same template, different masks and write rules):

| Model | State update | Mask / structure |
|---|---|---|
| Linear attention (Katharopoulos et al. 2020) | $S_t=S_{t-1}+k_tv_t^\top$ | $L=$ causal ones |
| RetNet (Sun et al. 2023) | $S_t=\gamma S_{t-1}+k_tv_t^\top$ | $L_{ts}=\gamma^{t-s}$ (fixed, Toeplitz) |
| **Mamba‑2 / SSD** (Dao & Gu 2024) | $S_t=a_tS_{t-1}+k_tv_t^\top$, $a_t$ input‑dependent scalar | $L$ 1‑semiseparable |
| GLA (Yang et al. 2024) | $S_t=\operatorname{diag}(\alpha_t)S_{t-1}+k_tv_t^\top$ | per‑key‑dimension decay |
| Mamba‑1 (Gu & Dao 2023) | diagonal $\bar A_t$ per channel | tensor‑valued mask (no single $T\times T$ $L$) |
| DeltaNet (Schlag et al. 2021; Yang et al. 2024) | $S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ | **overwrite** the old value at key $k_t$ instead of just adding |
| Gated DeltaNet (Yang et al., ICLR 2025) | decay + delta rule | used in recent hybrids (e.g. Qwen3‑Next, verify) |
| Softmax attention | the "state" is the whole KV cache | $\exp$ corresponds to an infinite‑dimensional feature map, so it can't be folded into a finite $N\times P$ sheet. That is *why* Theo has to keep every card |

**Manim notes:** the mountain is a `Polygon`. Each tunnel segment is a `Rectangle` that grows by `.animate.stretch` as each equation is `Write`‑n beside it. The two final equations are placed at the meeting point and merged with `TransformMatchingTex` into one box. The bracket hop in beat 3 is `TransformMatchingTex` with `key_map` so the parentheses visibly move.
**⚠ Don't say:** "Mamba‑2 *is* attention". It equals a **masked, unnormalised, linear** attention‑shaped operator. It never computes a softmax, and it never stores the cards.

---

### Slide 25 — Interpretability designs: from pictures to interventions (3:00)

**Signature frame:** a **switchboard** under a hybrid stack. A clean run (*Leo → bridge*) runs on top and a corrupted run (*Leo → cupboard*) on the bottom. Patch cables swap one component at a time, and a big **answer‑margin meter** (logit of *bridge* minus *cupboard*) swings, or doesn't.

**Beats: a gallery of designs, one tile each (⏸ per tile), every result marked `SCHEMATIC` unless real data is imported**
1. **Logit lens / tuned lens** (nostalgebraist 2020; Belrose et al. 2023). Read the belt at every layer with the unembedding (or a learned per‑layer probe), and watch "bridge" rise. *Cheap and descriptive.* ⏸
2. **Induction heads** (Elhage et al. 2021; Olsson et al. 2022). The pattern `[A][B] … [A] → [B]`, drawn as a previous‑token head feeding an induction head. Manu's version uses the off‑by‑one arrow from slide 23. ⏸
3. **Implicit attention of SSMs** (Ali, Zimerman & Wolf 2024, *The Hidden Attention of Mamba Models*; Zimerman et al. 2024). Slide 24's identity lets us **draw Mamba‑2's $L\odot QK^\top$ as a heatmap** and use it for attribution. Caveat on screen: *signed, descriptive, not causal*. ⏸
4. **Sparse autoencoders, transcoders, attribution graphs** (Bricken et al. 2023; Cunningham et al. 2023; Templeton et al. 2024; Dunefsky et al. 2024; Ameisen et al. 2025). The belt vector decomposes into a few active dictionary features, and a feature graph is traced from input to answer. Note: a residual SAE **does not see inside the SSM state**. Decomposing $S_t$ itself is a separate design. ⏸
5. **Causal interventions.**
   - Activation and path patching (Meng et al. 2022, ROME causal tracing; Wang et al. 2022, IOI; Heimersheim & Nanda 2024, best practices).
   - **Distributed Alignment Search** (Geiger et al. 2023/2024): swap only a learned subspace, $r\leftarrow r_{\text{base}}+UU^\top(r_{\text{src}}-r_{\text{base}})$.
   - For Mamba: causal tracing of factual recall (Sharma et al. 2024). ⏸
6. **The hybrid experiment: "Where did the answer come from?"** At the query token, just before an attention layer, three candidate paths light up:
   - (A) the **current Mamba state**;
   - (B) **source K/V** at the *bridge* position;
   - (C) other routes.

   The patch grid:

| Patch | If the answer flips, it suggests… |
|---|---|
| recurrent state at the query | the state already holds the binding |
| query residual | the answer is assembled at the query |
| **correct** source K/V | attention reads the source directly |
| **wrong** source K/V (control) | specificity check |
| knock out the source → query edge | necessity of that path |

   ⏸
7. **The decision tree for claims** (persistent on the final frame):
   *behaviour solved?* → *variable decodable (probe $D_Y$)?* → *intervention changes the answer?* → *via which path?* → *replicates under load and controls?*
   Under it: **decodable ≠ used**, and **probe accuracy, interchange accuracy and answer margin are three different numbers.**

**Math (lower third)**

$$
\Delta_Y^{(\ell)} = D_Y\big(r_q^{(\ell+1)}\big)-D_Y\big(r_q^{(\ell)}\big),\qquad
r_{\text{patched}} = r_{\text{base}}+UU^\top\big(r_{\text{src}}-r_{\text{base}}\big)
$$

**Manim notes:** `Switchboard` mobject (Appendix A). Cables are `CubicBezier`. The margin meter is an `Arrow` needle on an `Arc` driven by a `ValueTracker` set to **mock** values with the watermark `HYPOTHESIS TO TEST`.
**⚠ Don't say:** "attention retrieves the fact from the Mamba state". That is a **hypothesis**, and it competes with direct source‑KV reading. The source K/V were themselves computed from residuals that earlier Mamba layers had already shaped.

---

### Slide 26 ★ — Closing: different memories, sometimes similar computations (1:00)

**Signature frame:** Theo's card wall, Manu's notebook and Max's ledger each send one arrow into a **single residual vector** on the belt. The clinic's gold arrow joins it. Caption: **"Different memories. Sometimes similar computations. Causality decides."**

**Beats**
1. Recap ribbon, one icon per act: card wall → notebook → matrix identity → clinic → belt → tunnel. ⏸
2. Final narration (suggested):
   > "Theo keeps every card. Manu and Max carry the past forward in a notebook that never grows, and Max showed that notebook is a masked attention matrix in disguise. The clinic decides which transformation each token gets. And all of them talk through one shared belt. But resemblance isn't mechanism: to know what a model really uses, we have to trace the source, change the computation, and watch the answer." ⏸
3. Sources slide (Appendix D, as two scrolling columns).

---

## Appendix A — Reusable mobject library

All constructors take a `palette` module and default to the colours in §0.2. Each one is a `VGroup` subclass so it composes with `.animate`, `arrange` and `next_to`. **Every scene must build its own instances** (no hidden state across scenes), and the entry and exit positions are declared at the top of each scene file for compositing.

| Class | Signature (sketch) | Built from | Used in |
|---|---|---|---|
| `Brother` | `Brother(name, color, prop: Literal["cards","notebook","ledger"])` | `SVGMobject` (original line art) + `Text` | 1, 6, 10, 26 |
| `TokenCard` | `TokenCard(text, key_color=K_COL, val_color=V_COL)` | `RoundedRectangle`, `Text` | 1–4 |
| `IndexCardWall` | `IndexCardWall(n, heads=1, kv_groups=None)` with `.add_card()`, `.merge_heads(groups)`, `.grey_future(t)` | `VGroup` of `TokenCard` | 2–4, 6, 21 |
| `QueryBeam` | `QueryBeam(source, targets, scores: ValueTracker)` | `Line`, `always_redraw` bars | 2, 4 |
| `Heatmap` | `Heatmap(n, data, cmap, causal=True)` with `.set_mask(kind)` | grid of `Square` | 3, 4, 14–16, 21, 25 |
| `ResidualRail` | `ResidualRail(n_layers, token)` with `.write(vec, color)`, `.read_lens(layer)` | `Line`, `Arrow`, `Polygon` | 5, 20, 25 |
| `Notebook` | `Notebook(N, P, color)` with `.fade(a)`, `.write(b, x)` | `Rectangle` grid | 9, 10 |
| `StateSheet` | `StateSheet(N, P)` bound to a NumPy array | `Square` grid + `Brace` | 12, 13, 16 |
| `Dial` | `Dial(tracker, label)` | `Arc`, `Line` needle | 9, 11, 19 |
| `ScanTree` | `ScanTree(T)` | `Dot`, `Line` levels | 9 |
| `ChunkPipeline` | `ChunkPipeline(T, C)` with `.pass_(i)` for steps 1–4 | `Rectangle`s + arrows | 16 |
| `Clinic` | `Clinic(n_experts, k, shared=0)` with `.route(token, toy_scores)` | `Rectangle` doors, `CubicBezier` paths | 18, 19 |
| `GPUMap` | `GPUMap(n_gpu, experts_per_gpu)` with `.all_to_all(tokens)` | `RoundedRectangle`, `ArcBetweenPoints` | 19 |
| `Switchboard` | `Switchboard(components)` with `.patch(name)` | `CubicBezier`, `Arc` meter | 25 |
| `EvidenceBadge` | `EvidenceBadge(kind: Literal["IDENTITY","PROPERTY","SCHEMATIC","MEASURED"])` | `RoundedRectangle` + `Text` (dotted border for SCHEMATIC) | all |

**Hero snippet: the live decay mask of slide 14** (pattern only; adapt to the current ManimCE API):

```python
from manim import *
from manim_slides import Slide
import numpy as np
from math_contracts.test_ssd_equivalence import decay_mask
from visual.palette import *

class S14MatrixForm(Slide):
    def construct(self):
        self.camera.background_color = BG
        a = np.array([0.9, 0.8, 0.5, 0.75])            # a_1 unused when S0 = 0
        L = decay_mask(a)                              # same function the tests check
        strip = VGroup(*[MathTex(fr"a_{i+1}={a[i]:g}") for i in range(4)]).arrange(RIGHT).to_edge(UP)
        mat = DecimalMatrix(np.zeros((4, 4)), element_to_mobject_config={"num_decimal_places": 3})
        for e in mat.get_entries():
            e.set_opacity(0)
        self.play(FadeIn(strip), FadeIn(mat.get_brackets()))
        self.next_slide()
        for t in range(4):
            anims = []
            for s in range(4):
                e = mat.get_entries()[4 * t + s]
                e.set_value(L[t, s]).set_color(MAMBA2 if s <= t else FADE)
                anims.append(e.animate.set_opacity(1))
            contributing = [strip[u] for u in range(1, t + 1)]
            self.play(LaggedStart(*anims, lag_ratio=0.15),
                      *[Indicate(m, color=FADE) for m in contributing])
            self.next_slide()
```

**Hero snippet: tip‑to‑tail residual writes (slide 5)**

```python
class S05BankResidual(ThreeDSlide):
    def construct(self):
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3])
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        start = np.array([0.6, 0.4, 0.2])               # E[bank], identical in both lanes
        deltas = {"river": [[-.3, .5, .2], [-.4, .3, .1]],    # TOY
                  "money": [[.5, -.3, .2], [.4, -.4, .1]]}    # TOY
        self.add(axes)
        for lane, color in (("river", ATTN), ("money", MOE)):
            tip = start.copy()
            arrow = Arrow3D(ORIGIN, tip, color=V_COL)
            self.play(Create(arrow))
            for d in deltas[lane]:
                new_tip = tip + np.array(d)
                self.play(Create(Arrow3D(tip, new_tip, color=color)))
                tip = new_tip
            self.play(Create(Arrow3D(ORIGIN, tip, color=RESID)))
        self.next_slide()
        self.begin_ambient_camera_rotation(rate=0.08)
        self.next_slide(loop=True)
        self.wait(6)
```

---

## Appendix B — File map and render commands

```text
mamba-colloquium/
├── PRESENTATION_VISUAL_SCRIPT.md         # this file
├── agent_skill/SKILL.md                  # brief for coding agents
├── math_contracts/test_ssd_equivalence.py
├── reference/original_plan/              # original storyboard + skill (math §4, sources §9)
├── visual/                               # palette.py + Appendix A mobjects
└── slides/
    ├── act1_attention.py      # S01ColdOpen … S06LongContextTax
    ├── act2_notebooks.py      # S07RNN … S11Mamba2Knob
    ├── act3_ssd.py            # S12StateSheet … S16FourSteps
    ├── act4_clinic_belt.py    # S17WhyFFN … S21TransformerToolbox
    └── act5_theory.py         # S22Frontier … S26Closing
```

| Slide | Class | Slide | Class |
|---|---|---|---|
| 1 | `S01ColdOpen` | 14 | `S14MatrixForm` |
| 2 | `S02QKVLibrary` | 15 | `S15Semiseparable` |
| 3 | `S03PrefillKVCache` | 16 | `S16FourSteps` |
| 4 | `S04MultiHead` | 17 | `S17WhyFFN` |
| 5 | `S05BankResidual` (3D) | 18 | `S18Clinic` |
| 6 | `S06LongContextTax` | 19 | `S19ServingMoE` |
| 7 | `S07RNN` | 20 | `S20ResidualStream` |
| 8 | `S08LinearSSM` | 21 | `S21TransformerToolbox` |
| 9 | `S09MambaSelection` | 22 | `S22Frontier` (3D) |
| 10 | `S10MambaProblems` | 23 | `S23Universality` |
| 11 | `S11Mamba2Knob` | 24 | `S24Tunnel` |
| 12 | `S12StateSheet` | 25 | `S25Interpretability` |
| 13 | `S13Unroll` | 26 | `S26Closing` |

```bash
# environment (system deps: FFmpeg, libpango1.0-dev, LaTeX with dvisvgm, e.g. texlive-latex-extra + texlive-science)
python -m venv .venv && source .venv/bin/activate
pip install "manim==0.19.0" manim-slides numpy pytest

# gate: math must pass before any Act III / slide 24 render
python -m pytest math_contracts

# iterate on one slide, then look at it
scripts/render.sh l slides/act3_ssd.py S14MatrixForm
scripts/contact_sheet.sh media/videos/act3_ssd/480p15/S14MatrixForm.mp4 /tmp/s14.png 12

# full video: every slide at 720p30, concatenated in talk order
scripts/build_video.sh m          # → media/final/three_brothers_720p30.mp4

# live click-stepped deck (Deck.pause() becomes next_slide())
MANIM_DECK=slides manim-slides render slides/act1_attention.py S01ColdOpen S02QKVLibrary S03PrefillKVCache S04MultiHead S05BankResidual S06LongContextTax
manim-slides present S01ColdOpen S02QKVLibrary S03PrefillKVCache S04MultiHead S05BankResidual S06LongContextTax
manim-slides convert S01ColdOpen ... deck.html          # self-contained HTML backup
```

**Implementation status.** All 26 slides are implemented in `slides/act1_attention.py` … `slides/act5_theory.py`, with shared helpers in `visual/common.py`. The Appendix A classes are implemented there as lightweight functions (`brother`, `token_card`, `heatmap`, `evidence_badge`, …). Every slide was rendered at 480p and checked frame by frame with contact sheets (looking for clipping, overlaps and text off the frame) before the 720p build. The video export compresses the live talk: each `⏸` becomes a short hold instead of a click.

---

## Appendix C — Math contract additions

The authoritative contract is `reference/original_plan/TRANSFORMERS_MAMBA2_MOE_MANIM_STORYBOARD.md` §4. This script adds the following, all tested in `math_contracts/test_ssd_equivalence.py`:

1. **Chunked SSD (slide 16).** For chunk $I$ with positions $b_I..e_I$ and incoming state $S^{\text{in}}_I$:
   $$Y_I = (L_I\odot Q_IK_I^\top)V_I + \operatorname{diag}\!\Big(\textstyle\prod_{u=b_I}^{t}a_u\Big)_{t\in I} Q_I S^{\text{in}}_I,\qquad
   S^{\text{in}}_{I+1} = \Big(\textstyle\prod_{u\in I}a_u\Big)S^{\text{in}}_I + \sum_{s\in I}\Big(\textstyle\prod_{u=s+1}^{e_I}a_u\Big)k_sv_s^\top$$
   Tested for $C\in\{1,2,4,5,16\}$, including non‑divisible $T$ (`test_chunked_equals_recurrent`).
2. **Linear attention as the special case $a\equiv1$ (slide 24):** $Y=\operatorname{tril}(QK^\top)V$, plus associativity $(QK^\top)V=Q(K^\top V)$ (`test_linear_attention_is_ssd_with_no_decay`).
3. **Initial state:** $Y \mathrel{+}= \operatorname{diag}(\prod_{u\le t}a_u)\,QS_0$ (`test_nonzero_initial_state`).
4. **Rank of the off‑diagonal blocks is ≤ N** (`test_off_diagonal_block_rank_at_most_N`).
5. **Decode FLOP crossover (slide 17):** per layer and token, FFN $\approx 16d^2$ FLOPs (2 FLOPs/MAC, $8d^2$ weights) vs attention scores+mixing $\approx 4Td$. They are equal at $T\approx4d$. *Order of magnitude only.* It ignores GQA, which reduces bytes but not query FLOPs, and kernel effects.
6. **KV‑cache bytes:** $M_{KV}\approx b\cdot L_A\cdot T\cdot H_{kv}\cdot(d_k+d_v)$.

---

## Appendix D — References

Every external claim on a slide cites one of these. Anything marked *verify* must be checked against the primary source or model card before the talk.

**Attention and Transformers**
- Bahdanau, Cho & Bengio (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.* arXiv:1409.0473
- Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762
- Shazeer (2019). *Fast Transformer Decoding: One Write‑Head is All You Need* (MQA). arXiv:1911.02150
- Ainslie et al. (2023). *GQA.* EMNLP 2023. <https://aclanthology.org/2023.emnlp-main.298/>
- Child et al. (2019). *Generating Long Sequences with Sparse Transformers.* arXiv:1904.10509
- Beltagy, Peters & Cohan (2020). *Longformer.* arXiv:2004.05150
- Zaheer et al. (2020). *Big Bird.* arXiv:2007.14062
- Jiang et al. (2023). *Mistral 7B.* arXiv:2310.06825
- Gemma Team (2024, 2025). *Gemma 2*, arXiv:2408.00118; *Gemma 3*, arXiv:2503.19786
- Xiao et al. (2023). *Efficient Streaming Language Models with Attention Sinks.* arXiv:2309.17453
- DeepSeek‑AI (2024). *DeepSeek‑V2* (MLA). arXiv:2405.04434
- Yuan et al. (2025). *Native Sparse Attention.* arXiv:2502.11089
- Dao et al. (2022). *FlashAttention.* arXiv:2205.14135 · Dao (2023). *FlashAttention‑2.* arXiv:2307.08691 · Shah et al. (2024). *FlashAttention‑3.* arXiv:2407.08608
- Kwon et al. (2023). *PagedAttention / vLLM.* SOSP 2023, arXiv:2309.06180
- Su et al. (2021). *RoFormer (RoPE).* arXiv:2104.09864 · Chen et al. (2023). *Position Interpolation.* arXiv:2306.15595 · Peng et al. (2023). *YaRN.* arXiv:2309.00071

**Recurrence and state spaces**
- Elman (1990). *Finding Structure in Time.* · Hochreiter & Schmidhuber (1997). *LSTM.* · Cho et al. (2014). *GRU*, arXiv:1406.1078 · Bengio, Simard & Frasconi (1994). *Learning long‑term dependencies with gradient descent is difficult.*
- Gu et al. (2020). *HiPPO.* arXiv:2008.07669 · Gu, Goel & Ré (2022). *S4.* ICLR 2022, arXiv:2111.00396
- Gu & Dao (2023). *Mamba: Linear‑Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
- Dao & Gu (2024). *Transformers are SSMs (Mamba‑2 / SSD).* ICML 2024, arXiv:2405.21060, with blog series at <https://goombalab.github.io/blog/2024/mamba2-part1-model/>
- Katharopoulos et al. (2020). *Transformers are RNNs.* arXiv:2006.16236
- Schmidhuber (1992). *Learning to control fast‑weight memories.* · Schlag, Irie & Schmidhuber (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174
- Sun et al. (2023). *RetNet.* arXiv:2307.08621 · Yang et al. (2024). *Gated Linear Attention.* arXiv:2312.06635 · Yang et al. (2024). *Parallelizing Linear Transformers with the Delta Rule.* arXiv:2406.06484 · Yang, Kautz & Hatamizadeh (2025). *Gated Delta Networks.* ICLR 2025, arXiv:2412.06464

**FFN and MoE**
- Geva et al. (2021). *Transformer Feed‑Forward Layers Are Key‑Value Memories.* EMNLP 2021, arXiv:2012.14913
- Shazeer (2020). *GLU Variants Improve Transformer* (SwiGLU). arXiv:2002.05202
- Shazeer et al. (2017). *Sparsely‑Gated Mixture‑of‑Experts.* arXiv:1701.06538
- Lepikhin et al. (2020). *GShard.* arXiv:2006.16668
- Fedus, Zoph & Shazeer (2021). *Switch Transformers.* arXiv:2101.03961
- Jiang et al. (2024). *Mixtral of Experts.* arXiv:2401.04088
- Dai et al. (2024). *DeepSeekMoE.* arXiv:2401.06066
- DeepSeek‑AI (2024). *DeepSeek‑V3 Technical Report.* arXiv:2412.19437
- Qwen3, Llama 4 and gpt‑oss model cards / technical reports (*verify* the expert counts and top‑k)

**Hybrids**
- Lieber et al. (2024). *Jamba.* arXiv:2403.19887
- Waleffe et al. (2024). *An Empirical Study of Mamba‑based Language Models.* arXiv:2406.07887
- NVIDIA (2025). *Nemotron‑H* (arXiv:2504.03624) and *Nemotron 3 Nano 30B‑A3B* (architecture: <https://github.com/NVIDIA-NeMo/Nemotron>), *verify* the layer schedule
- Cooper et al. (ICML 2026). *Expressivity‑Efficiency Tradeoffs for Hybrid Sequence Models* (cited in the original plan, *verify*)

**Expressivity and universality**
- Yun et al. (2020). *Are Transformers universal approximators of sequence‑to‑sequence functions?* ICLR 2020, arXiv:1912.10077
- Siegelmann & Sontag (1995). *On the computational power of neural nets.* · Pérez, Barceló & Marinkovic (2021). *Attention is Turing‑Complete.* JMLR
- Arora et al. (2024). *Zoology.* ICLR 2024, arXiv:2312.04927 · Arora et al. (2024). *Simple linear attention language models balance the recall‑throughput tradeoff (Based).* ICML 2024, arXiv:2402.18668
- Jelassi et al. (2024). *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML 2024, arXiv:2402.01032
- Merrill, Petty & Sabharwal (2024). *The Illusion of State in State‑Space Models.* ICML 2024, arXiv:2404.08819
- Grazzi et al. (2025). *Unlocking State‑Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025, arXiv:2411.12537
- Li et al. (2015). *Convergent Learning.* arXiv:1511.07543 · Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited (CKA).* arXiv:1905.00414 · Olah et al. (2020). *Zoom In.* Distill · Huh et al. (2024). *The Platonic Representation Hypothesis.* ICML 2024, arXiv:2405.07987
- Wang et al. (2025). *Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures.* ICLR 2025, arXiv:2410.06672

**Interpretability**
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.* · Olsson et al. (2022). *In‑context Learning and Induction Heads.* · Elhage et al. (2022). *Toy Models of Superposition.* (Transformer Circuits Thread)
- nostalgebraist (2020). *interpreting GPT: the logit lens.* · Belrose et al. (2023). *Tuned Lens.* arXiv:2303.08112
- Meng et al. (2022). *Locating and Editing Factual Associations in GPT (ROME).* arXiv:2202.05262 · Wang et al. (2022). *Interpretability in the Wild (IOI).* arXiv:2211.00593 · Heimersheim & Nanda (2024). *How to use and interpret activation patching.* arXiv:2404.15255
- Geiger et al. (2024). *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS).* arXiv:2303.02536
- Bricken et al. (2023). *Towards Monosemanticity.* · Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features.* arXiv:2309.08600 · Templeton et al. (2024). *Scaling Monosemanticity.* · Dunefsky et al. (2024). *Transcoders.* arXiv:2406.11944 · Ameisen et al. (2025). *Circuit Tracing: Revealing Computational Graphs in Language Models.*
- Ali, Zimerman & Wolf (2024). *The Hidden Attention of Mamba Models.* arXiv:2403.01590 · Zimerman, Ali & Wolf (2024). *Explaining Modern Gated‑Linear RNNs via a Unified Implicit Attention Formulation.* arXiv:2405.16504
- Sharma et al. (2024). *Locating and Editing Factual Associations in Mamba.* arXiv:2404.03646 · Paulo, Marshall & Belrose (2024). *Does Transformer Interpretability Transfer to RNNs?* arXiv:2404.05971
- The research‑programme report used in the original plan (§9.1): hypotheses and experimental design only, **not** measured results.

**Tooling**
- Manim Community docs: <https://docs.manim.community/en/stable/>
- manim‑slides: <https://github.com/jeertmans/manim-slides>
- Agent skill: <https://github.com/edoardob90/manim-skills> (third party, ManimCE 0.20.1)

---

## Appendix E — QA checklist

**Math and complexity**
- [ ] `python -m pytest math_contracts` passes before any Act III or slide‑24 render. Toy numbers on screen are imported from the same functions.
- [ ] Slide 3 separates **prefill** (fills the cache, $O(T^2)$), **decode** ($O(T)$ per token), and **cache memory** ($O(T)$). Cached decode is never shown as O(1).
- [ ] Slides 12–16 and 24: dimensions are labelled, $a_t$ is a scalar per head, the upper triangle of $L$ is exactly 0, and **no softmax in any SSD equation**.
- [ ] Slide 17: the "FFN ≈ ⅔" claim carries the long‑context caveat.
- [ ] Slide 21: FlashAttention is described as exact and IO‑aware, **not** subquadratic.

**Science**
- [ ] Heads (slide 4), experts (slides 18–19) and features (slides 23, 25) are never given guaranteed semantic labels.
- [ ] Every toy or invented number carries a `SCHEMATIC` badge. Every cited number has its source on screen, and model‑card numbers are re‑verified.
- [ ] Slide 10: "fixed state ⇒ must forget" is never stated. Capacity and interference are posed as a question.
- [ ] Slide 9: Mamba‑1 training is shown as a **parallel** scan.
- [ ] Slide 20: state across *time* vs residual across *depth* are two distinct arrows.
- [ ] Slide 25: decodability, interchange accuracy and answer margin are distinguished. The hybrid "retrieval from state" story is labelled **HYPOTHESIS**.

**Design**
- [ ] Palette is consistent (Q cyan, K purple, V yellow, residual white, attention blue, Mamba coral, Mamba‑2 mint, MoE gold).
- [ ] Horizontal = time and vertical = depth, everywhere.
- [ ] ≤1 primary equation and ≤25 words on screen. No clipped LaTeX at 1080p and 1440p.
- [ ] Every slide renders independently at `-ql` before the final render. Stills are exported for slides 2, 3, 5, 12, 14, 16, 18, 20, 21, 24, 25.
- [ ] Characters and art are original. Inspired by 3Blue1Brown's pedagogy, with no copied branding or assets.
