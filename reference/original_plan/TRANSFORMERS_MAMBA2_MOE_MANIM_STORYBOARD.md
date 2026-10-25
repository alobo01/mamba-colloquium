# From Index Cards to Recurrent Memory
## A mathematically faithful, 3Blue1Brown-inspired Manim film about Transformers, Mamba, Mamba-2, MoE, and mechanistic interpretability

**Status:** production specification / research-verified storyboard, **not** a rendered video.  
**Audience:** ML researchers comfortable with Transformer attention, linear algebra, and asymptotics; no prior state-space model knowledge assumed.  
**Language:** English.  
**Target:** ~41 minutes, with a chapterable ~25-minute core cut; 16:9 2560×1440 or 3840×2160 at 30 fps; vector-first in Manim Community Edition.  
**Style goal:** intuitive, geometric, rhythmically revealed mathematics inspired by the pedagogical approach of *3Blue1Brown*, with original artwork, character designs, diagrams, typography, and animation choices—not a copy of the channel's proprietary branding or distinctive assets.  
**Deliverable for a coding agent:** one cohesive film assembled from independently renderable Manim scenes, narration, and a mathematical QA test suite.

> **The governing visual thesis:** A Transformer preserves addressable past representations, Mamba recurrently updates an integrated state, Mamba-2 admits two mathematically equivalent computational views, and MoE selects *which token-wise computation to run*. The residual stream is their shared communication interface. These distinctions concern *mechanisms*, not claims that a model has an anthropomorphic memory.

---

## 0. One-minute creative pitch

Three brothers are asked to follow a long story about keys, parcels, places, and money:

- **Theo / Transformer**, the archivist, files an **index card at every position** and later retrieves the relevant cards using the current query. His strength is *source-addressable retrieval*.
- **Manu / Mamba**, the notebook keeper, continuously updates a **fixed-size notebook**. He learns when to write, retain, attenuate, or overwrite. His strength is *efficient recurrent processing*.
- **Max / Mamba-2**, Manu's brother, maintains a related notebook but realises the same computation can be carried out using **structured blocks of matrix multiplication**. His key improvement is the *state-space duality (SSD) algorithm* and a more GPU-friendly parameterisation.

All three consult the **same workshop of specialist experts** when the architecture uses a Mixture of Experts (MoE): a token is routed to selected FFNs, not given its own omniscient human specialist. The workshop metaphor is not a fourth memory architecture.

The story starts with a deliberately ambiguous sentence:

> “The brothers watched fish near the bank.”  
> “The brothers deposited money at the bank.”

At the embedding stage, an identical illustrative **bank** token begins with the *same token embedding* in both sentences (before contextual processing; positional handling depends on the architecture). Its residual representation evolves into different context-sensitive states. Later, a memory test asks:

> “Leo left the green key under the bridge. Noah put the red key in the cupboard. … Where is Leo's key?”

The **bank** example explains *representation*; the **brothers' keys** example explains *cross-token memory and retrieval*. Keep these distinct and reuse both later in the interpretability experiments.

### The three big reveals

1. **Transformer attention can retrieve old tokens but its exact causal history grows with context.** A KV cache avoids recomputing *past* keys and values; it does not eliminate attention over cached positions.
2. **Mamba-2 is an SSM and also an attention-like structured matrix.** A recurrent outer-product update unrolls into a causal, decay-masked, *unnormalised* attention-like operation. These two implementations are mathematically equivalent for the core SSD mapping.
3. **A hybrid can exploit both forms of memory while MoE changes compute capacity along an orthogonal axis.** What the trained model actually uses attention for must be established by intervention.

**Explicit misconceptions to prevent:** (i) a KV cache is not the same as prefill, (ii) cached attention decoding is not O(1) in context length, (iii) a recurrent state is not a guaranteed lossy summary, (iv) Mamba-2 does **not** implement ordinary softmax attention, (v) MoE selects FFNs, not whole independent LLMs, and (vi) a colourful SAE feature or probe is not proof of a causal mechanism.

---

## 1. Visual language and durable continuity

### 1.1 Palette and mapping (consistent across every shot)

| Visual object | Proposed colour | Meaning |
|---|---|---|
| Residual stream / shared workspace | white `#F2F2F2` | Representations moving upward through depth |
| Tokens / neutral information | cream `#D8D5CD` | Discrete sequence positions |
| Queries `q` | turquoise `#3AD5D0` | What is being sought |
| Keys `k` / addresses | purple `#B78CFF` | Matching / source identity |
| Values `v` / payloads | warm yellow `#FFD166` | Information returned |
| Transformer / Theo / index cards | blue `#4F8EF7` | Addressable source-indexed memory |
| Mamba-1 / Manu / notebook | coral `#F28C6B` | Recurrent update and selective forgetting |
| Mamba-2 / Max / blocks | mint `#69D6B4` | Same recurrence, efficient blockwise computation |
| MoE / router | gold `#F2C14E` | Selection of token-wise experts |
| Fading, suppression, zero ablation | subdued grey `#747D8C` | Reduction/removal |
| Hypothetical interpretability data | dotted outlines | Predictions/schematics, **not measurements** |

Use a very dark midnight background (`#11141C`), fine off-white strokes, spacious negative space, smoothly interpolated motion, and occasionally a chalklike construction grid. Equations should enter **after** the physical metaphor produces the need for them.

### 1.2 Characters, props and spatial grammar

- Brothers are minimal line-art busts with distinctive accessories: Theo has blue card sleeves; Manu carries a red notebook; Max carries a green block ledger. Avoid complicated face animation.
- **Vertical axis** always means **network depth**. **Horizontal axis** means **token position / sequence time**. Interventions should never conflate these axes.
- Residual stream: a pale vertical stack of token-specific rails. Each block adds a coloured vector arrow to a rail.
- Theo's memory: a growing row of cards, one per token **per attention layer**. Do not imply a single global universal KV cache shared by all layers.
- Manu/Max's state: fixed *shape* for each recurrent layer. Token arrivals change the *contents* but not the number of slots. It is not literally an ordinary semantic-summary paragraph.
- Specialized FFNs: workshops with different geometric tool icons, *not* labelled “law expert”, “math expert”, etc. as though actual expert specialisation were guaranteed. Use **illustrative specialisations** with an on-screen disclaimer.

### 1.3 Recurring visual grammar

| Operation | Reusable animation idiom | Manim candidates |
|---|---|---|
| Token arrives | card slides into a timeline | `LaggedStart`, `FadeIn`, `.animate.shift` |
| Attention reads old token | `q` beam scans `k` labels, then yellow `v` flows back | `CurvedArrow`, `Flash`, `Line`, `ValueTracker` |
| Update residual | component arrow tip-to-tail, then resulting vector replaces previous | `Vector`, `Arrow`, `TransformMatchingTex` |
| Selective state update | orange ink enters notebook while previous glyphs fade | `ValueTracker`, `always_redraw`, `FadeOut` |
| SSD algebra | identities remain anchored while terms morph | `TransformMatchingTex`, `Indicate`, `SurroundingRectangle` |
| Matrix multiplication | row/column sweep; output tile illuminates | `Matrix`, `VGroup`, `LaggedStart` |
| MoE dispatch | token fans out to top-`k` workshops and weighted replies reconverge | `Arrow`, `TracedPath`, `AnimationGroup` |
| Counterfactual patch | clean and corrupt streams run side by side, one wire swaps | `Transform`, `ReplacementTransform`, `DashedLine` |

**Legibility constraints:** reserve the centre ~75% for each key mathematical construction. Display at most one primary equation or one labelled diagram plus a compact equation. Annotation font ≥38 px at 1440p in master renders; avoid more than ~25 on-screen words in an explanatory frame. Every equation must have explicit dimensions available on an adjacent reveal or lower-third.

### 1.4 Source / evidence labels

Use three distinct overlays: **MATHEMATICAL IDENTITY**, **ARCHITECTURAL PROPERTY**, and **HYPOTHESIS / SCHEMATIC**. A fourth overlay **MEASURED RESULT — SOURCE REQUIRED** is permitted only if genuine experiment outputs are imported. In the present plan, all interpretability heatmaps are explicitly hypothetical.

---

## 2. Chapter arc and timing

| Act | Scenes | Time | Narrative question |
|---|---|---:|---|
| I. The address book | S01–S05 | 0:00–8:10 | How does a Transformer look up, cache, and contextualise information? |
| II. Why the notebook? | S06–S09 | 8:10–14:10 | Why recurrent models, what did Mamba change, and what did it not fix? |
| III. The mathematical reveal | S10–S14 | 14:10–24:40 | How does Mamba-2 relate exactly to attention, and why is it fast? |
| IV. Specialists and compromises | S15–S19 | 24:40–33:15 | Why FFNs, why MoE, and why aren't modern Transformers naive? |
| V. Mechanisms, not metaphors | S20–S23 | 33:15–41:20 | What can we interpret, what can we prove, and what remains open? |

**Core 25-minute cut:** S01–S04, S05 abbreviated, S06–S08, S10–S13, S15–S17, S19, S20/S23 abbreviated. Move SSD chunk internals (S14) and causal interpretability procedures (S21–S22) into extended edition. Each scene is independently renderable and has a descriptive start/end state.

---

## 3. Full scene-by-scene storyboard

Each scene contract specifies: **concept**, **single visual hook**, **math**, **beats**, **narration intent**, **transition**, **research guardrail**. Scene IDs are stable interface names for implementation.

### S01 — Cold open: the brothers and the key (~1:10)

**Visual hook:** three brothers receive a never-ending ribbon of cards. Theo adds every card to a wall of labelled drawers, Manu rewrites a notebook page, Max does the same but groups incoming notes into square batches. A question card arrives: “Where is Leo's key?” Three methods glow.

**Beat order:**
1. Reveal three identical-looking question cards above three different memory devices.
2. Flash back to “Leo → green key → bridge” and “Noah → red key → cupboard”.
3. Theo pulls the **bridge** source card directly. Manu's notebook emits a candidate answer. Max's notebook responds via its current state.
4. Freeze on the question: **“Do they remember the same way?”**

**Do not** claim the recurrent outputs will necessarily be correct. Make this a neutral setup. Transition by morphing the physical story cards into model tokens.

### S02 — Q, K, V, and why several heads? (~2:20)

**Concept:** `r_t` denotes the current token representation (we will unpack its evolution in S04); query–key scores determine which source values to combine; value and key have different roles; multi-head allows different independently learned projections and computations.

**Visual:** Theo asks “**Which key belongs to Leo?**”. Every index card has a purple title (`k_s`) and a yellow payload (`v_s`). A cyan query `(q_t)` passes over purple titles. Matching is a graded **inner product**, not a symbolic exact-match database lookup. Yellow payloads flow to the query as a weighted sum. Duplicate the head into 3 conceptual heads: **entity association**, **coreference**, **positional pattern**. Call these **illustrations of possible learned behaviours**, not guaranteed specialisations.

**Equations, revealed left to right:**

\[
 q_t^{(h)}=W_Q^{(h)}r_t,\quad k_s^{(g)}=W_K^{(g)}r_s,\quad v_s^{(g)}=W_V^{(g)}r_s,
\]
\[
 \alpha_{ts}^{(h)}=\frac{\exp\bigl((q_t^{(h)})^\top k_s^{(g(h))}/\sqrt{d_k}+b_{ts}^{(h)}\bigr)}{\sum_{j\le t}\exp\bigl((q_t^{(h)})^\top k_j^{(g(h))}/\sqrt{d_k}+b_{tj}^{(h)}\bigr)},
\]
\[
 o_t= W_O\operatorname{concat}_h\left(\sum_{s\le t}\alpha_{ts}^{(h)}v_s^{(g(h))}\right).
\]

The group mapping `g(h)` supports MHA (`g(h)=h`) and anticipates GQA. Optional bias `b` accounts for position mechanisms where applicable; use 0 in the simplified cartoon. Show masked future cards greyed out.

**Question answered explicitly:** why not simply use a single vector? Because Q/K separate *selection* from V *content* and multiple heads can learn distinct relations/subspaces before their outputs are mixed. No assertion that heads map one-to-one onto semantic roles.

**Transition:** Theo starts answering a second, then a third autoregressive token; duplicated computation begins to look wasteful.

### S03 — Prefill is not the KV cache (~1:45)

**Concept:** causal self-attention under autoregressive inference has two phases. The KV cache stores previously computed attention-layer keys and values, avoiding recomputation; it does not make the new query independent of the prefix.

**Split shot:** left = **PREFILL**: a triangular `T×T` attention grid is painted in large tiles for a prompt. Right = **DECODE**: one new query row scans `T` cached columns. Previously computed K/V cards stay in memory; only new token K/V cards are appended. Add a horizontal memory-meter growing linearly with cached tokens.

**Equations (attention-core terms, per layer, fixed model dimensions):**

\[
 \begin{aligned}
 \text{full causal prefill:}\quad &O(T^2d_h H)\ \text{attention work},\\
 \text{one decode step at context }T:\quad &O(Td_h H)\ \text{attention work},\\
 \text{KV cache size:}\quad &2Tn_{\rm kv}d_h\ \text{numbers per layer}.
 \end{aligned}
\]

Include QKV/projection and FFN work as *additional terms*. **Total** work for generating `M` tokens starting at prompt length `T`: `O(MT+M²)` attention-score/value work (for fixed head widths), **not** simply `O(M)` or `O(T)` for the whole generation. On-screen correction: **“Cached decode is O(context length) per token; the cache itself is O(context length).”**

**Terminology:** user wording “profiling” is likely **prefilling**. In narration, use **prefill**. Explicitly show that KV caching and prefill are distinct concepts.

**Transition:** pause on the contextual `bank` card from the cache; ask what vector actually feeds its Q/K/V projections.

### S04 — The evolving meaning of “bank” (~1:55)

**Concept:** token embeddings are an initial representation; contextual computation changes the residual states at successive layers.

**On-screen equations:**

\[
 r_t^{(0)}=E[\tau_t] \quad\text{(plus any applicable input-position term)},
 \qquad r_t^{(\ell+1)}=r_t^{(\ell)}+u_t^{(\ell)}.
\]

**Animation:** split screen with “fish near the **bank**” above “money at the **bank**.” Align the same bank token between sentences. Initially, identical yellow embedding arrows are overlaid; after each block, different small component vectors get added and the bank residual becomes a visibly different resultant direction. Finish with a **2D projection** of token states on an illustrative semantic axis (river-related vs finance-related), labelled **cartoon projection, not empirically measured geometry**.

**Subtlety:** contextual disambiguation need not happen only at the bank position and need not yield perfectly disentangled directions. Exact tokenizer segmentation and the architecture's position handling matter: modern RoPE Transformers typically introduce position in queries/keys rather than literally adding position vectors to initial residuals. Show only the *token-embedding* guarantee; keep positional encoding a side label, not an assumption.

**Narration anchor:** “The embedding is a starting point, not the meaning the model ultimately uses.”

**Transition:** pull back from one contextual token to thousands of cached token positions; show the long-context memory and compute tax.

### S05 — The long-context tax (~1:00)

**Visual hook:** a grid grows from 32×32 to 128×128 to 512×512, then into a virtual horizon. Two counters run: **compute in full prefill ~quadratic** and **stored K/V in decode ~linear**. Draw notional growth curves, clearly *asymptotics*, not measured latency. A small “memory bandwidth” meter empties whenever a decode step rereads the KV wall.

**Intuition:** the same indexed memory that makes exact lookup flexible carries a growing per-layer cache. FlashAttention can reduce intermediate attention-memory traffic without changing the quadratic dense-attention arithmetic; GQA/MLA/windowing change cache size or scope, to be revisited.

**Transition:** Theo asks “What if the notebook never grew?” Match-cut to Manu's fixed-size state.

### S06 — Why recurrence came first (~1:20)

**Concept:** ordinary RNNs have used fixed-dimensional hidden state for decades; they're natural for streaming. The recurrent update has challenges with gradient propagation and sequential training dependencies.

**Math:**

\[
 h_t=\tanh(Wh_{t-1}+Ux_t+b),\qquad
 \frac{\partial h_T}{\partial h_s}=\prod_{t=s+1}^{T}\frac{\partial h_t}{\partial h_{t-1}}.
\]

**Animation:** the notebook compresses many arriving words into a single glowing state. A faint gradient signal moves backwards through gates multiplying or shrinking; in another shot it amplifies. A calendar slams down: **LSTM** and **GRU** mitigated aspects of long-range retention. Standard recurrent dependencies also limit straightforward parallelisation across time.

**Accuracy:** vanishing/exploding gradients and memory interference are *risks*, not universal impossibility theorems. Do not say “RNNs were unable to process long sequences”; RNNs (including LSTMs) have long been used for sequence applications.

**Transition:** swap RNN nonlinear state to a structured linear-state update with learnable dynamics.

### S07 — From S4 to selective state spaces and Mamba-1 (~2:05)

**Concept:** structured SSMs encode temporal dynamics; fixed time-invariant dynamics are efficiently convolutional but weak at content-dependent selection. Mamba makes SSM parameters input dependent and introduces an efficient selective scan.

**Animation beats:**
1. **Time-invariant filter (S4 intuition):** each arriving card pushes a uniform ripple through a memory pipe. Its dynamics don't change because one token says “ignore me”.
2. **Selective update:** the tokens “Leo”, “green key”, “um”, “ignore previous clue” pass the notebook. `Δ_t, B_t, C_t` controls brighten/attenuate differently for different inputs.
3. **Scan:** rather than animated Python for-loop, represent updates as associatively composable transition operators; a tree combines updates in parallel for training, and a single state updates in streaming inference.

**Generic selective SSM equations:**

\[
 s_t=\bar A_t s_{t-1}+\bar B_t u_t,\qquad
 y_t=C_t^\top s_t,
\]

with `\bar A_t,\bar B_t,C_t` depending on current input and `\bar A_t` structured (diagonal for Mamba-1's core S6). Show a local depthwise convolution preceding the selective SSM and a multiplicative gate after its output as transparent peripheral pieces, not central derivation.

**Important guardrail:** Mamba-1 already has a hardware-aware *parallel selective scan* in training; never suggest it simply loops over all tokens on GPU.

**Transition:** Manu can process a long ribbon, but when he needs to store many independent exact arbitrary bindings, his notebook faces a different kind of pressure.

### S08 — One state, many bindings: the limitation is addressability (~1:25)

**Visual:** inject 4, 8, 16, 32 independent `{person→location}` bindings into the same notebook. Compare Theo's **position-indexed card wall** with Manu's **fixed-shape matrix state**. Query **Noah** then **Leo**. Theo can directly weight distinct cards by the new query; Manu must decode from the same aggregate state. Shift the query while holding prefix fixed to emphasise current-query sensitivity.

**Key caveat:** a fixed-dimensional real state is **not mathematically forced to forget** at arbitrary precision. The relevant concerns are finite-precision capacity, interference, learned generalisation, and access patterns. Mamba can implement associative retrieval; it is not “a summariser that cannot recall”.

**On-screen question:** “Can we keep constant state *and* make recurrent computation as GPU-friendly as matrix multiplication?”

**Transition:** Max places the notebook on a grid of square tiles.

### S09 — Why Mamba-1 needed a sequel (~1:10)

**Visual:** GPU tensor cores are stylised as a giant array of synchronised matrix-multiplication cells. Mamba-1's scan operations look like a narrower, scan-shaped path that does not saturate the array as effectively. Max restricts part of the dynamics **within each head** and re-expresses the computation as block matrix operations. A caption says: **“Change the parameterisation to change which hardware kernels are practical.”**

**Math preview:**

\[
 A_t^{\mathrm{Mamba1}}=\mathrm{diag}(a_{t,1},\ldots,a_{t,N})\quad \longrightarrow\quad
 A_t^{\mathrm{Mamba2}}=a_t I_N \quad\text{(per head)}.
\]

This **restricts** the class of state transitions at fixed `N` but permits efficient SSD algorithms and practical use of larger `N`. Research results in the source paper report a 2–8× speedup for the *core layer* under its measured setups; do not present this as a generic end-to-end system speedup.

**Transition:** the simplest possible Mamba-2 recurrence occupies the centre of the screen. Everything else dims.

### S10 — Mamba-2 state, dimensions, and what is actually stored (~2:05)

**Central frame:** a matrix `S_t` with **N rows × P columns**, one per recurrent head. Show `N` memory coordinates (rows), `P` channels (columns); `B_t` is an `N`-vector, `x_t` is a `P`-vector, `C_t` is an `N`-vector, and `a_t` is a **scalar within the head**.

**Core equations (no omitted dimensions):**

\[
 \underbrace{S_t}_{N\times P}
 =\underbrace{a_t}_{\text{scalar}}\underbrace{S_{t-1}}_{N\times P}
 +\underbrace{b_t}_{N\times1}\underbrace{x_t^\top}_{1\times P},
 \qquad
 \underbrace{y_t}_{P\times1}=S_t^\top\underbrace{c_t}_{N\times1}.
\]

**Animated proof-by-objects:** the previous state fades/scales by `a_t`; `b_tx_tᵀ` paints a rank-one outer-product rectangle; output vector appears by multiplying current state with `c_t`. Say **“forget/retain, write, read”**. More accurately: when `0<a_t<1` it attenuates the old state; the write/read vectors determine what is integrated and exposed. The model has additional input projections, causal short convolution, gates and normalisation outside this simplified SSD core.

**Notation bridge:** `b_t` is *SSD write/key*, `c_t` is *SSD read/query*, and `x_t` is the inner SSD *value/input*, not the original text-token embedding. We will rename `k_t=b_t`, `q_t=c_t`, and `v_t=x_t` for comparison with attention **only within the derivation**.

**Transition:** fix `S_t`; unfold it into ghostly contributions from previous positions.

### S11 — First derivation: recurrent memory becomes an implicit attention sum (~2:20)

**Start with 3 positions**; show every algebraic expansion as a physical stacking of prior outer products:

\[
 \begin{aligned}
 S_1 &= a_1S_0+b_1v_1^\top,\\
 S_2 &= a_2a_1S_0+a_2b_1v_1^\top+b_2v_2^\top,\\
 S_t &= \left(\prod_{u=1}^t a_u\right)S_0+
 \sum_{s=1}^{t}\left(\prod_{u=s+1}^{t}a_u\right)b_sv_s^\top.
 \end{aligned}
\]

Set `S_0=0` for the main storyboard (explain the omitted initial-state term). Multiply `S_tᵀ` by `c_t=q_t`:

\[
 \boxed{y_t=\sum_{s=1}^{t}\underbrace{\left[\prod_{u=s+1}^{t}a_u\right]}_{\text{selective temporal survival}}\underbrace{(q_t^\top k_s)}_{\text{content compatibility}}\underbrace{v_s}_{\text{source contribution}}.}
\]

**Visual:** each earlier orange write has an adjustable transparent veil proportional to later forget factors, then a cyan query dots into purple keys. Yellow source value vectors mix together. Theo's attention weighted-sum shot from S03 returns beside it, generating the “aha”.

**Important qualification:** coefficients here can be signed and are not softmax probabilities. `q_t^T k_s` is a dot product; no general normalisation to sum 1. The expression is an **algebraic decomposition**, not evidence that the runtime separately stores and directly looks up each `(k_s,v_s)`.

**Transition:** pack all the implicit coefficients into a square causal matrix.

### S12 — Second derivation: the attention-shaped matrix (~2:30)

**Visual:** a blackboard displays two tiled `T×T` heatmaps side by side. Left: Transformer `softmax(QKᵀ/√d + mask)` with row-wise normalisation. Right: SSD `L⊙(QKᵀ)`, where `L` is a lower-triangular product-of-decays matrix. A row pointer scans the same **position** in each heatmap. Their visual resemblance is the punchline; the different normalisation and memory are immediately annotated.

**Definitions:**

\[
 Q=\begin{bmatrix}q_1^\top\\\vdots\\q_T^\top\end{bmatrix},\quad
 K=\begin{bmatrix}k_1^\top\\\vdots\\k_T^\top\end{bmatrix}\in\mathbb{R}^{T\times N},\quad
 V=\begin{bmatrix}v_1^\top\\\vdots\\v_T^\top\end{bmatrix}\in\mathbb{R}^{T\times P},
\]
\[
 L_{ts}=\begin{cases}\prod_{u=s+1}^t a_u,&s\le t,\\0,&s>t,\end{cases}
\qquad
 \boxed{Y=\bigl[L\odot(QK^\top)\bigr]V.}
\]

Show a **4×4 explicit example** and derive two entries: `L_{4,2}=a_3a_4` and `L_{4,4}=1`. The matrix `L` is a *1-semiseparable* lower-triangular structure; the masked mixer `M=L⊙(QKᵀ)` has off-diagonal blocks with rank at most `N` (the feature dimension). This is exactly the **core SSD mapping** under the stated assumptions.

**A/B comparison:**

\[
 \begin{array}{c|c}
 \text{causal softmax attention}&\text{SSD / Mamba-2 core}\\\hline
 \operatorname{softmax}_{\rm row}(QK^\top/\sqrt{N}+\mathcal M)V & [L\odot(QK^\top)]V\\
 \text{source-indexed K/V cache}&\text{contracted fixed-shape recurrent state}
 \end{array}
\]

Use simple labels **“algebraic relation ≠ same runtime memory”** and **“SSD ≠ softmax attention”**.

**Transition:** the `T×T` SSD matrix looks quadratic and expensive—Max grabs scissors and slices it into blocks.

### S13 — How can a `T×T` formula have linear-time computation? (~1:45)

**Visual:** the matrix from S12 is divided into square `C×C` chunks. For blocks below the diagonal, thin low-rank conduits replace huge dense blocks. Show `S_t` traveling across chunk boundaries, while local chunk computations happen simultaneously on many GPU tiles.

**Exact algebra for separated chunks:** for a later row `t` in chunk `I` and earlier column `s` in chunk `J`, `I>J`, factor

\[
 L_{ts}=\underbrace{\prod_{u=b_I}^{t}a_u}_{u_t}\;
 \underbrace{\prod_{u=e_J+1}^{b_I-1}a_u}_{\gamma_{I,J}}\;
 \underbrace{\prod_{u=s+1}^{e_J}a_u}_{v_s^{\mathrm{decay}}}.
\]

Here `b_I` is the first position of later chunk `I` and `e_J` the last of earlier chunk `J`; empty products equal 1. This gives the token-mixing off-diagonal block

\[
 M_{I,J}=\gamma_{I,J}\,\operatorname{diag}(u_I)\,Q_I\,
 K_J^\top\,\operatorname{diag}(v_J^{\mathrm{decay}}),
\qquad \operatorname{rank}(M_{I,J})\le N.
\]

**Visual proof:** the full off-diagonal block folds into two thin factors `C×N` and `N×C`, connected by a scalar. Emphasise **low-rank off-diagonal organisation**, not that *every* triangular submatrix is low rank independently of N.

**Transition:** take the physical chunks to the tensor-core grid, introducing the four-pass SSD execution in S14.

### S14 — The four steps of SSD and cost accounting (~1:50)

**Animated 4-part pipeline**, with a single `T`-token row divided into `T/C` chunks:

1. **Local outputs:** inside each chunk, compute the causal within-chunk contribution **assuming incoming state is zero**.
2. **Chunk writes:** compute each chunk's outgoing-state contribution from its own tokens **assuming incoming state is zero**.
3. **State passing:** combine chunk state transforms in order via a scan over **`T/C` chunks**, not `T` tokens.
4. **Final correction:** use each true incoming chunk state to add its contribution to that chunk's output.

Use unique time-consistent colours for the four components; the initial input's contribution must visibly reach the output. The local calculation can use quadratic work in **chunk size** `C` but does not construct the full sequence `T×T` matrix.

**Complexity (one head, SSD core, ignoring projections/gates/convolution):**

\[
 \begin{aligned}
 \text{streaming per-token time:}&\quad O(NP),\\
 \text{streaming recurrent state:}&\quad O(NP),\\
 \text{SSD blockwise full-sequence work:}&\quad O\bigl(T[NP+C(N+P)]\bigr)\quad\text{for a representative fixed-}C\text{ implementation},\\
 \text{hence for fixed }N,P,C:&\quad O(T)\text{ in sequence length}.
 \end{aligned}
\]

The displayed blockwise count is **schematic arithmetic work for a standard chunked contraction**, not a universal kernel/runtime bound. Some reference implementations do a quadratic chunk-level scan (`O((T/C)^2)`) for simplicity and therefore cease to be asymptotically linear for unbounded `T`; an associative scan restores long-context linear scaling. Model-level linear projections/gates add cost, and full-model throughput is not determined by these SSD terms alone.

**GPU point:** many intra-chunk operations are high-throughput matmuls. SSD's advantage is **work organisation**, not magic elimination of arithmetic.

**Transition:** dissolve the chunks and return to one token's residual stream; ask what happens *after* positions exchange information.

### S15 — Why FFNs, and why do they take so much compute? (~1:40)

**Visual:** a Transformer attention block moves yellow information *between positions*. Then an FFN “refinery” independently processes each token's residual vector via expansion, nonlinear gating and contraction. A toy before/after vector reveals “**attention brings information; FFN transforms it**,” but explain both modules can implement more nuanced computations, and retrieval/knowledge cannot be assigned exclusively to one.

**Math:** standard two-layer FFN

\[
 \mathrm{FFN}(r)=W_{\rm down}\,\sigma(W_{\rm up}r+b)+b',
\]

and a commonly used gated alternative

\[
 \mathrm{SwiGLU}(r)=W_{\rm down}\left[\mathrm{SiLU}(W_{\rm gate}r)\odot(W_{\rm up}r)\right].
\]

For gated width `d_ff`, show **~`3d d_ff` FFN weights** per block versus **~`4d²` attention-projection weights** in ordinary equal-width MHA (subject to GQA, dimensions, biases and implementation). For `d_ff≈8d/3`, this is roughly `8d²` FFN weights versus `4d²` attention projections. Corresponding dense matrix FLOPs per token scale roughly with these parameter counts (each multiply-add convention should be stated). **Caveat:** at sufficiently long context, the attention-score/value calculations and KV bandwidth can dominate instead. Don't state FFN is always the largest runtime bottleneck.

**Transition:** the refinery expands into eight equally large workshops. Why force every token to use all of them?

### S16 — The expert workshop: sparse MoE routing (~1:45)

**Visual:** for an illustrative MoE layer, 8 workshops sit in a semicircle. An incoming bank residual vector is scored by a gold **router**, selecting top-2 workshops. Two yellow paths illuminate. The returned vectors are *weighted and summed* into the token's residual. Show a second token taking a different route; still only its own FFN-style transformation happens in that layer.

**Math:**

\[
 z= W_{\rm router}r,\quad I(r)=\operatorname{TopK}(z,k),\quad
 \mathrm{MoE}(r)=\sum_{e\in I(r)}g_e(r)\,\mathrm{FFN}_e(r),
\]

where selected `g_e` are nonnegative router weights, commonly normalised over selected experts; precise gating varies by model. Some models add always-on shared experts:

\[
 \mathrm{MoE}_{\rm shared}(r)=\sum_{e\in I(r)}g_e(r)\,\mathrm{FFN}_e(r)
 +\sum_{j\in\mathcal S}\mathrm{FFN}^{\rm shared}_j(r).
\]

**Illustrative numbers:** top-2 of 8 similarly sized experts means **8 experts stored** and **2 experts evaluated per token**, **not** one quarter of all *model* FLOPs, and not necessarily cheaper than one expert of the same individual width. Sparse MoE increases total model capacity relative to active per-token compute when compared at **matched stored parameter budgets**.

**Critical visual correction:** no specialist is literally assigned to “financial bank” from the text itself: learned experts can encode distributed behaviour and routing is dynamic. Use colour-coded abstract operations, never claim semantic expert purity.

**Transition:** the workshop becomes very popular; everyone queues for one doorway.

### S17 — What serving a modern MoE really costs (~1:30)

**Visual:** 100 token dots enter the router. Without balancing, most select workshop #3; it develops a queue and other workshops sit idle. The router's balancing mechanism redistributes traffic. Then animate physical GPUs: tokens are dispatched over links to whichever GPU owns the selected experts, worked on, and returned. Highlight both arithmetic and communication costs.

**Show three mechanisms:**
- **Top-k sparse activation**, router gradients and selection.
- **Balancing/capacity management:** historical auxiliary losses / overflow / token dropping as in Switch-style systems versus later routing/bias strategies such as auxiliary-loss-free balancing in DeepSeek-V3. Model-specific, not universal.
- **Expert parallelism** and **all-to-all dispatch** with batch efficiency effects, plus memory storage for inactive parameters.

**Research references:** Shazeer et al. (2017) sparsely gated MoE; Switch Transformer (2021/2022) top-1 routing; Mixtral 8×7B (2024) top-2 of 8; DeepSeekMoE (2024) fine-grained/shared experts; DeepSeek-V3 (2024/2025) sparse large-scale routing/balancing. Each date is attributed to original release or publication in reference appendix.

**Transition:** MoE workshops snap onto *both* Theo's attention layers and Max's recurrent layers; it is an orthogonal mechanism.

### S18 — Modern Transformers are less naive than the caricature (~1:55)

**Visual:** cut down an enormous attention heatmap through successive architecture techniques **without conflating their savings**. One panel per method; name origin and example model where appropriate.

| Technique | Animated transformation | What it fundamentally changes | Introduced / associated models |
|---|---|---|---|
| **Sliding-window attention** | triangular mask becomes banded strip | local attention connections and often bounded per-layer KV history; may reduce distant exact access | Longformer (Beltagy et al., 2020), **Mistral 7B** (2023) decoder deployment |
| **Multi-query attention (MQA)** | many query heads share one K/V stack | sharply reduces cached K/V heads; query multiplicity remains | Noam Shazeer, *Fast Transformer Decoding* (2019) |
| **Grouped-query attention (GQA)** | 8 query heads share 2 K/V groups | adjustable quality/cache trade-off | Ainslie et al., EMNLP 2023; Mistral 7B |
| **FlashAttention** | huge attention matrix is streamed through tiny SRAM tiles | IO/activation-memory efficiency for **exact** attention, not asymptotically subquadratic dense-attention FLOPs | Dao et al., NeurIPS 2022; follow-ups FlashAttention-2/3 |
| **PagedAttention / vLLM** | logical K/V cards stored in non-contiguous physical pages | allocation efficiency, fragmentation, batching | Kwon et al., SOSP 2023, vLLM |
| **Multi-head latent attention (MLA)** | cached K/V width compresses into latent cards | reduces per-token KV storage while preserving source-indexed cached history | DeepSeek-V2 (2024) |
| **Sparse/global attention variants** | select important global slots among local windows | global access for some positions without full dense everywhere | Longformer (2020), BigBird (2020) |

Emphasise **FlashAttention and PagedAttention are systems/algorithm improvements**, not different query-addressing semantics at the modelling level. GQA still has linearly many cached time positions. Sliding-window local receptive fields can expand across stacked layers but do not guarantee global exact addressability.

**Transition:** zoom out to compare full architectures on a common graph rather than claim one always wins.

### S19 — The efficiency–expressivity frontier (~1:45)

**Visual:** a two-axis conceptual map: x = *streaming memory growth with context*, y = *flexibility of direct token-addressable retrieval*. Points are **qualitative**, not performance benchmarks. Full attention at high addressability/linear KV memory; Mamba-2 at fixed per-layer state/less explicit source addressability; hybrids between. MoE adds a **third dimension: stored capacity vs active token compute**—animate workshops rising perpendicular to the map. Put a separate visual strip for measured-model outcomes if/when real data is supplied.

**Comparative table (core sequence mixer, not full LLM):**

| Architecture / technique | Prefill mixer work vs T (fixed widths) | Decode mixer per token vs current T | Mixer state/KV across context | Source addressability |
|---|---:|---:|---:|---|
| Dense causal softmax attention | `O(T²)` | `O(T)` | `O(T)` | explicit per-token K/V |
| GQA / MQA | `O(T²)` | `O(T)` | `O(T)` but fewer KV channels | explicit, shared K/V groups |
| Fixed-window attention (W fixed) | `O(TW)` | `O(W)` | `O(W)` per layer with eviction | window-local |
| Mamba-1 selective recurrence | `O(T)` using scan | `O(1)` in T | `O(1)` in T | implicit via recurrent state |
| Mamba-2 core SSD | `O(T)` via appropriate chunk algorithm | `O(1)` in T | `O(1)` in T | implicit via recurrent state |
| Hybrid | architecture dependent | attention layers `O(T)`; recurrent layers `O(1)` | K/V for attention layers; fixed states for recurrent | mixed |

*All big-O statements suppress model width, layer count, batch size, kernel-specific constants, projection costs, and MoE communication. “O(1)” means constant **with respect to sequence length**, not zero cost.*

**Theory anchor:** Cooper et al. (ICML 2026) establish task-family-specific expressivity/efficiency separations and construct efficient hybrids for selective copying and associative recall. Do **not** transform this into an unqualified theorem that all hybrids dominate all Transformers or SSMs.

**Transition:** assemble a concrete hybrid architecture from alternating coloured block glyphs.

### S20 — A real hybrid: residual stream as a common workspace (~1:40)

**Visual:** build an illustrative stack `M, E, M, E, A, E, M, E ...` around horizontal tokens, where `M` = Mamba-2, `A` = attention, `E` = MoE. Then reveal a **grounded example**: **NVIDIA Nemotron 3 Nano 30B-A3B**, a 52-layer hybrid consisting of 23 Mamba-2 layers, 6 GQA layers, and 23 MoE layers in the reported model architecture. Those totals refer to architectural layer types, not 52 repetitions of a single Mamba→MoE→attention triplet. Check the exact arrangement against the selected checkpoint's config before animating a *literal* block order. Avoid using the simplified `M M M A` as the actual model's full schedule.

**Always-visible residual equation:**

\[
 r_t^{(\ell+1)}=r_t^{(\ell)}+F_{\ell}\bigl(\operatorname{Norm}(r^{(\ell)})\bigr)_t.
\]

Inside `F`: either a cross-position attention computation, a recurrent Mamba-2 computation, or a token-wise MoE transformation depending on architecture. More complicated blocks can have multiple residual additions; explicitly say this is an **abstract update view**, not a claim every checkpoint has identical block internals.

**Crucial two-axis illustration:** Mamba recurrent state `S_t^(ℓ)` carries information **forward across time within one layer**. Residual vector `r_t^(ℓ)` carries token-specific information **upward across depth**. Each token position has a residual vector at each layer. A hybrid attention layer can use *earlier positions' residual-derived K/V*, not only decode the current position's Mamba state. Show two separate arrows.

**Transition:** freeze on a query token just before an attention boundary. What exactly was learned/stored, and by which path did it arrive?

### S21 — Interpretability design I: storage, addressability, re-instantiation (~1:55)

**Based on attached report:** distinguish world state `G`, binding `B_i`, source position `P_i`, query `q` and answer `Y=B_q`. Don't display “attention restores a fact from Mamba” as an established result.

**Animation:** pair two counterfactual story variants:

- Original: “Leo: bridge; Noah: cupboard. **Where is Leo's key?**”
- Query swap: identical prefix, now “**Where is Noah's key?**”

The pre-query recurrent state is identical by causal construction. Later representations at the queried token can differ. Draw four overlays: `G`, `B_i`, `P_i`, `Y`. On the hybrid stack, animate a **hypothetical** jump in target `Y` decodability across attention, but ask whether `G` or individual `B_i` was already present in the state and whether the jump is causally used.

**Report-backed equation:**

\[
 \Delta_Y^{(\ell)}=D_Y(r_q^{(\ell+1)})-D_Y(r_q^{(\ell)}).
\]

`D_Y` is held-out probe performance with matched procedure across layers; high performance alone does not prove the model causally uses the variable. Optional still: `D_Y` heatmap over **layer × token position**, watermark **SCHEMATIC**.

**Transition:** a hypothetical “Leo → bridge” feature lights up after attention. Ask whether it is a real circuit component or a measurement artefact.

### S22 — Interpretability design II: three lenses, one causal question (~2:05)

**Visual triptych:** the **same** latent target `Y` is viewed with (1) **SAE feature bars**, (2) **linear/DAS subspace projection**, (3) **representational geometry**. All three view the same source/target bindings and the same before/after hybrid boundary. Under them, place one large **causal switchboard** for activation and edge patching.

**SAE:**

\[
 z=f(W_{\rm enc}(r-b)),\quad \hat r=W_{\rm dec}z+b.
\]

Illustrate **candidate** source-associated feature becoming large at query; test feature ablation and answer change. Distinguish feature similarity and feature *identity* across independent SAEs; match by decoder cosine + activation co-occurrence, not merely index.

**Linear/causal subspace:** show that a linear classifier decoding the location is not causal proof. DAS-style intervention transplants a learned subspace:

\[
 r_{\rm patched}=r_{\rm base}+UU^\top(r_{\rm source}-r_{\rm base}).
\]

**Geometry:** a cyclic `month` toy dataset projects onto a circle; the diagram is explicitly **latent target geometry**, not a statement that the model performs computation by physically rotating around that circle. Show circle probe + label-shuffled control + corresponding counterfactual intervention.

**Causal switchboard:** clean = `Leo → bridge`; corrupt = `Leo → cupboard`. Compare patching (a) recurrent state at query, (b) query residual, (c) correct source KV, (d) wrong source KV, plus (e) correct source-to-query knockout. Make the *answer logit margin* visibly change when a source is truly causal. Use **illustrative arrows and no measured percentages**.

**Guardrails from report:** direct source-KV use and state-only decoding are competing explanations. In a hybrid, old token K/V may already have been transformed by earlier Mamba layers. Report result as “supported by targeted interventions” only once measured. Negative SAE result does not by itself falsify the representation hypothesis.

**Transition:** the possible matches across very different architectures become constellations of similarly shaped, but not identical, circuits.

### S23 — Towards universality: same task, similar computation, different memory? (~2:25)

**Visual:** Theo, Manu, and Max repeatedly solve the *same* little induction or entity-tracking problem. Their external outputs align; inside, traces form partially corresponding circuits. Some patterns align; other temporal offsets appear (a one-position-delayed arrow illustrates the **off-by-one motif** reported for Mamba). Put an SAE feature correspondence beside a **causal intervention graph**. Then split into two distinct meanings of universality:

**A. Universal approximation / formal expressive power.** Under specified assumptions and problem domains, Transformers have universal-approximation results (e.g. Yun et al., ICLR 2020). This does **not** say a fixed finite model solves every algorithmic problem or that every architecture has identical asymptotic memory. Hybrid expressivity results (Cooper et al., 2026) concern particular task families, conditions, and resource bounds.

**B. Mechanistic universality.** A research hypothesis asks whether networks trained on similar tasks independently discover similar internal features/circuits. Wang et al. (ICLR 2025), *Towards Universality*, report SAE-feature similarities and structurally analogous induction circuits across Transformer and Mamba families, while noting an off-by-one Mamba motif. **Cross-architecture feature similarity is not proof of identical mechanisms or necessary causal function.**

**Final narrated question:** “Can we identify the same computational variable, transported through different memories, with the same causal consequences?”

**Final visual:** three siblings' routes merge into a residual-space vector. The closing caption is a question rather than a triumphalist ranking: **“Different memory architectures. Sometimes similar computations. Causality decides.”** Fade to sources.

---

## 4. The mathematical appendix that must be built into the film's source code

This section is the authoritative equation contract for animators. Keep all symbolic conventions consistent across scenes.

### 4.1 Residual-stream conventions

Let `t∈{1,…,T}` index token positions, `ℓ` block depth, `d` residual width, and `r_t^(ℓ)∈R^d`. A simplified pre-normalised update is

\[
 r_t^{(\ell+1)}=r_t^{(\ell)}+u_t^{(\ell)},\qquad
 u_t^{(\ell)}=F_{\ell}(\mathrm{Norm}(r^{(\ell)}))_t.
\]

For a conventional attention+FFN Transformer block with two residual additions,

\[
 a_t^{(\ell)}=r_t^{(\ell)}+\operatorname{Attention}_{\ell}(\operatorname{Norm}(r^{(\ell)}))_t,
\]
\[
 r_t^{(\ell+1)}=a_t^{(\ell)}+\operatorname{FFN}_{\ell}(\operatorname{Norm}(a_t^{(\ell)})).
\]

A hybrid architecture may instead alternate distinct block types. `r_t^(0)` should be called the *input residual representation*, with input embeddings and any architecture-dependent position incorporation. RoPE alters projected Q/K rather than requiring additive positional embeddings in the residual.

### 4.2 Attention with MHA, MQA, GQA

For attention head `h` and KV group `g(h)`, `q_t^h,k_s^{g(h)}∈R^{d_k}`, `v_s^{g(h)}∈R^{d_v}`. Let `H_q` be query heads and `H_kv` K/V heads. In MHA `H_q=H_kv`; in MQA `H_kv=1`; in GQA `1<H_kv<H_q`. Per-layer caching of a prompt of length `T`, ignoring batch/bytes, is `2T H_kv d_k` elements if keys/values have the same width `d_k` (general case `T H_kv(d_k+d_v)`). With `L_A` attention layers and `b` bytes per element, `M_KV ≈ b L_A T H_kv(d_k+d_v)`. Add batch size and padding/storage overhead as needed. Head-specific softmax computation continues to scan eligible positions even with grouped K/V.

### 4.3 Mamba-2 core duality, unrolled rigorously

For one SSD head,

\[
 a_t\in\mathbb{R},\quad b_t,c_t\in\mathbb{R}^N,\quad v_t,y_t\in\mathbb{R}^P,\quad S_t\in\mathbb{R}^{N\times P}.
\]
\[
 S_t=a_tS_{t-1}+b_tv_t^\top,\qquad y_t=S_t^\top c_t.
\]

Assuming `S_0=0`, by induction,

\[
 S_t=\sum_{s=1}^{t}\left(\prod_{u=s+1}^{t}a_u\right)b_sv_s^\top.
\]

Substituting gives

\[
 y_t=\sum_{s=1}^{t}\left(\prod_{u=s+1}^{t}a_u\right)(c_t^\top b_s)v_s.
\]

Define `Q[t,:]=c_tᵀ`, `K[t,:]=b_tᵀ`, `V[t,:]=v_tᵀ` and causal decay mask `L_{ts}=1_{s≤t}∏_{u=s+1}^{t}a_u`. Then **exactly**

\[
 Y=(L\odot QK^\top)V.
\]

This equality holds for **the defined recurrence**, not for full softmax attention and not by itself for the complete Mamba-2 block with all additional projections, short convolutions, activation functions, normalisation and gates. The Mamba-2 architecture learns the coefficients and includes those surrounding operations; SSD is its central sequence mixer.

In a typical Mamba-2 parameterisation, discretised/head-wise decay factors are positive and often take the form `a_t=exp(\Delta_t A)` with `A<0`, `Δ_t>0`, so `0<a_t<1`. Do not hardcode this exact parameterisation into abstract proof tests; specify the coefficients used. The SSD's temporal products are thus amenable to log-space / segment-sum implementations, while naïve subtraction of very long cumulative sums can be numerically unstable.

**A second proof (attention → recurrence):** define an *incrementally updatable sufficient state*

\[
 S_t=\sum_{s=1}^{t}\lambda_{t,s}k_sv_s^\top,\qquad
 \lambda_{t,s}=\prod_{u=s+1}^{t}a_u.
\]

Then

\[
 S_t=a_t\sum_{s=1}^{t-1}\lambda_{t-1,s}k_sv_s^\top+k_tv_t^\top
 =a_tS_{t-1}+k_tv_t^\top,
 \quad y_t=S_t^\top q_t.
\]

Show these as two animated routes to the *same* final expression, not two approximate algorithms.

### 4.4 Semiseparable factorisation and SSD chunking

For `T` tokens partitioned into blocks of `C`:

- Within each block: compute masked bilinear token-token interaction. Arithmetic roughly `O(C² N+C² P)` per chunk, `O(TC(N+P))` over the sequence.
- Chunk output state / incoming state contributions: outer-product contractions and projections roughly `O(TNP)` for the sequence.
- Chunk scan: O(`T/C`) sequential state-composition steps if sequential (or logarithmic parallel depth with associative scan), with a `N×P` recurrent state per head. Beware conflating **span/depth**, **total arithmetic work**, and **physical GPU latency**.
- This yields the representative arithmetic accounting `O(T[NP+C(N+P)])` for fixed-head fixed-chunk SSD kernels, neglecting surrounding network operations. Matrix block implementations can differ; derive exact kernel-specific constants only from actual chosen implementation.

**Tiny 4×4 numeric example** for the screen (safe, deterministic): choose scalar key/query vectors (`N=1`) and scalar values (`P=1`) and `a_2=0.8,a_3=0.5,a_4=0.75`, with `S_0=0`. Then

\[
 L=\begin{bmatrix}
 1&0&0&0\\
 .8&1&0&0\\
 .4&.5&1&0\\
 .3&.375&.75&1
 \end{bmatrix}.
\]

The two screen-check entries `.375=a_3 a_4=.5×.75` and `.3=a_2 a_3 a_4=.8×.5×.75` must be tested in Python. Use separate colour channels for input contribution and decay so arithmetic is visible.

### 4.5 Complexity: never omit the scope

Define `T` input length; `M` generated tokens; `d` model width; `H_q` query heads; `H_kv` KV heads; `N` SSD state dimension; `P` per-SSD-head channel width; `C` SSD chunk size; `W` local attention window. Keep all three cost concepts separate:

1. **Prefill FLOPs / arithmetic vs input length**: dense causal attention `O(T² H_q d_k)` plus projection/FFN costs; state-space mixer linear in T with fixed N,P,C; window `O(TW H_q d_k)`.
2. **One-token generation computation vs available context**: full attention `O(T H_q d_k)` plus all dense projections/FFNs; SSM recurrence core `O(NP)`; local window `O(W H_q d_k)`.
3. **Live recurrent/KV memory vs length**: full/GQA/MQA `O(T H_kv d_k)` elements; Mamba-2 state `O(NP)` per head plus finite local-convolution buffer; local-window cache `O(W H_kv d_k)` if correctly evicted.

Full *wall-clock* comparisons additionally depend on load level, kernels, bandwidth, quantisation, batch, whether the model is compute- or memory-bound, implementation of MoE communications, and GPU architecture. Do not overlay synthetic 2×/5× speedups as measured data. If charts show cost curves, label **asymptotic illustrations**.

### 4.6 FFN and MoE accounting

Dense SwiGLU per token, ignoring biases: `3d d_ff` parameter multiplications (~`6d d_ff` FLOPs under 2-FLOP multiply-add convention), compared with `4d²` projection weights in ordinary Q/K/V/O attention (GQA modifies K/V projections). With `E` equal-width MoE experts and top-`k`, the expert weight storage per layer is approximately `E×3d d_ff`, while expert matmul work per token approximately `k×3d d_ff` multiply-adds, excluding routers/shared experts/communication. These are **relative to individual expert width**; compare like-for-like architectures carefully.

---

## 5. Shot specifications: reusable hero animations

### Hero A — Residual “bank” vector in 3D (S04)

- Camera rotates only slightly so viewer recognises a vector in feature space, **not** an epistemically accurate map of actual learned language geometry.
- Start: one cream `bank` box feeding the same vector in two separate sentence panels.
- Animation: small colour-coded arrows are added by successive blocks. Label diagram “illustrative latent projection”.
- Maintain sentence context text at top while the vector rotates; the viewer must never lose which bank sense is being shown.
- Final still suitable for presentation: title **“Embeddings start a representation; residual updates contextualise it.”**

### Hero B — KV cache as index-card wall (S02–S05)

- Draw `T=6`, then `T=12`, then `T=24` card slots. Query beam reads all eligible past K/V positions by soft weights and returns values.
- Prefill shows triangular full interaction and cache construction *within* the prompt. Decode shows only one new row but still touches eligible cached positions.
- Counter increments **cached positions** and **heads × K/V groups** separately.
- The wrong version would show the query opening exactly one drawer and magically retrieving without scanning/softmax; use this only as a *human metaphor* and visibly return to fuzzy weighted retrieval.

### Hero C — Notebook with input-dependent ink (S06–S08)

- Notebook perimeter retains exactly the same pixel dimensions as token count grows.
- Show a single state matrix behind it so the human cartoon cannot be confused with a semantic English synopsis.
- For a filler token, illustrate a smaller learned write / different decay only as a possible policy, not universal behaviour.
- For hard random bindings, show overlapping traces and an interrogative “capacity/interference?” instead of incorrectly drawing guaranteed lost facts.

### Hero D — Recurrence-to-matrix metamorphosis (S10–S14)

- Draw three terms `a_t S_{t-1}`, `b_t v_t^T`, `S_t^T c_t` as physical operations on the SAME labelled dimensions.
- Keep matching `q/c`, `k/b`, `v/x` terms in anchored screen locations as they move through unrolling and matrix form.
- Rows fill one at a time in `L`: diagonal 1; lower triangular multiplied decay; upper triangle exactly zero.
- After the `T×T` matrix fills, the off-diagonal block factorises into skinny factors, and tiles detach into parallel processor cells.
- End with two representations (fixed-state scan and chunked matrix computation) converging on the **same output vector** for a numeric toy case.

### Hero E — MoE dispatch map (S16–S17)

- Use eight abstract workshops, gold top-k router. Per-token routing probabilities indicated by *bar lengths*, but use invented numeric examples labelled `toy weights`.
- Active expert highlights only run for selected top-k. Idle workshops are still present (weights still occupy storage).
- Second shot makes expert traffic skew and balancing visible. Third shows all-to-all across four GPUs; make communication an explicit cost rather than unearned “free speed”.
- End with a 2×2 conceptual plot: **memory mixer choice** (addressed vs recurrent) × **FFN choice** (dense vs MoE). All four combinations can exist.

### Hero F — A mechanistic test, not an interpretation cartoon (S21–S23)

- Top track: clean run `Leo→bridge`. Bottom: matched corruption `Leo→cupboard`.
- Three candidate information paths into query `r_q`: (A) current Mamba recurrent state, (B) correct source K/V, (C) other routes. Represent both A and B as potentially active.
- Patch correct KV, patch wrong-source KV, patch state separately. Show output probabilities only as **mock demonstration, not empirical observation** unless real logged scores are imported.
- Contrast visible **decodability**, actual **causal interchange accuracy**, and the **answer margin**; they are three different quantities.
- A hypothetical large attention-boundary jump needs the watermark **HYPOTHESIS TO TEST**.

---

## 6. Optional research-grade visual chapters and still figures

1. **Implicit attention of a recurrent layer:** visualise `\|c_{t\leftarrow s}\|` or signed scalar influences from Mamba's unrolled recurrence beside softmax attention. Label `descriptive influence; not causal attribution`. Signed cancellation matters.
2. **Retrieval scaling stress test:** animated `n_bindings∈{4,8,16,32,64}` and source–query distance grid for both families; separate *number of bindings* and *distance*. Only populate measured curves if experiment results are provided.
3. **Source support entropy:** `p_s ∝ \|c_{t←s}\|`, `N_eff=exp(-∑p_s log p_s)`; clearly explain that broad contribution-norm support is not evidence of broad causal reliance.
4. **Mamba off-by-one motif:** highlight token information written into recurrent state one position later in selected mechanistic studies; do not generalise to every feature/head/model.
5. **State/feature site matters:** residual stream SAE versus Mamba **internal recurrent-state** decomposition (e.g., matrix-atom approaches such as WriteSAE). A residual SAE does not expose the entire SSM state.
6. **Quantisation and systems sidebar:** BF16/FP8/INT4 throughput depends on hardware and kernel; avoid conflating lower parameter bytes, actual dense tensor-core FLOPs, expert routing and KV bandwidth.
7. **Long-context failure-mode panel:** lost-in-middle retrieval, copy/repeat tasks, MQAR; use *counterfactual key–value tasks* to avoid semantic knowledge confounds.
8. **Decision-tree slide for experimental claims:** `behaviour solved? → representation decodable? → causal intervention works? → correct path? → replication under load?`.

Recommended keynote still exports: `bank_residual.png`, `qkv_heads.png`, `prefill_vs_decode.png`, `kv_memory_growth.png`, `rnn_vs_ssm.png`, `mamba2_outer_product.png`, `ssd_matrix_identity.png`, `ssd_four_stages.png`, `ffn_vs_moe.png`, `modern_attention_optimisations.png`, `hybrid_residual_flow.png`, `causal_source_paths.png`, `universality_two_meanings.png`.

---

## 7. Agent implementation plan (Manim Community Edition)

### 7.1 External Manim agent skill to reference

**Verified existing community agent skill:** [`edoardob90/manim-skills`](https://github.com/edoardob90/manim-skills), containing `manim/SKILL.md` plus separate scene, mobject, plotting, animation, rendering, and gotcha guides. Its README documents installation by `npx skills add edoardob90/manim-skills`. It was authored for ManimCE 0.20.1, so an agent must cross-check version-sensitive APIs against the **current [Manim Community documentation](https://docs.manim.community/)** rather than assume the skill's version is installed or current. **This is an external suggestion, not an installed skill in this environment.**

The accompanying local [`agent_skill/SKILL.md`](agent_skill/SKILL.md) is a project-specific *agent contract*. It is **not** a replacement for the verified community Manim skill; it connects that reference to this film, requires mathematical tests and scene rendering, and forbids speculative interpretation being shown as data.

```bash
# Optional for an agent runtime that supports skills (review repository first)
npx skills add edoardob90/manim-skills

# Python environment for rendering (not installed by this document)
python -m venv .venv
source .venv/bin/activate
python -m pip install 'manim>=0.20,<1' 'manim-voiceover>=0.3,<1' numpy scipy
# Install system-level dependencies (FFmpeg, LaTeX) as required by Manim docs.
```

**Render commands once project files exist:**

```bash
manim -ql scenes/act_03_ssd.py S12SSDMatrixDuality
manim -qm scenes/act_03_ssd.py S12SSDMatrixDuality
manim -qh scenes/act_03_ssd.py S12SSDMatrixDuality
```

Use [`TransformMatchingTex`](https://docs.manim.community/en/stable/reference/manim.animation.transform_matching_parts.TransformMatchingTex.html) for algebraic reveals; `MathTex` for equations; `ValueTracker`/`always_redraw` for live changing factors; Manim Voiceover for narration tied to scene beats. Audio can also be recorded and edited externally. Don't force auto-generated narration.

### 7.2 Proposed source tree

```text
manim-explainer/
├── TRANSFORMERS_MAMBA2_MOE_MANIM_STORYBOARD.md   # this document
├── agent_skill/SKILL.md                          # instructions for coding agents
├── pyproject.toml                                # Manim CE and dependencies
├── scenes/
│   ├── act_01_attention.py                       # S01–S05
│   ├── act_02_mamba.py                           # S06–S09
│   ├── act_03_ssd.py                             # S10–S14
│   ├── act_04_moe.py                             # S15–S19
│   └── act_05_interpretability.py                # S20–S23
├── visual/
│   ├── palette.py
│   ├── brothers.py
│   ├── token_track.py
│   ├── residual_rail.py
│   ├── memory_cards.py
│   ├── state_matrix.py
│   ├── heatmaps.py
│   ├── moeworkshops.py
│   └── captions.py
├── math_contracts/
│   ├── ssd_reference.py                           # NumPy implementations of both views
│   ├── attention_reference.py
│   └── test_ssd_equivalence.py
├── narration/
│   ├── s01.md ... s23.md                           # segment-by-segment VO scripts
│   └── pronunciation.md
├── tests/
│   ├── test_source_math.py
│   ├── test_dimensions.py
│   ├── test_scene_smoke.py
│   └── test_caption_compliance.py
├── stills/
├── output/                                        # preview and final media
└── references/                                    # sources and licence details
```

### 7.3 Build order and verification gates

**Milestone A — animatic and continuity (no polished rendering).** Create a black-and-white animatic for S02, S03, S04, S10, S12, S16, S22. Check that time, depth, and memory states never silently swap coordinate meanings; review motion order and runtime against narration. **Deliverable:** seven low-quality clips and a single composite storyboard strip.

**Milestone B — mathematical core.** Implement tiny NumPy recurrence and matrix-duality tests, then S10–S14. Use random seeds and small dimensions to numerically compare `(L⊙QKᵀ)V` against a time-unrolled SSD recurrent loop. Add a nonzero initial-state test with the appropriate extra term. Check scalar `a` and matrix dimensions in every intermediate rendering. **Deliverable:** math unit tests plus a rendered SSD “aha” transition.

**Milestone C — narrative system.** Build reusable tokens, three brothers, residual rails, card wall, notebook, and MoE workshops. Preserve object identity across cuts (same bank embedding, same key-card contents, same query/corrupt examples). **Deliverable:** S01–S19 at preview quality.

**Milestone D — epistemic guardrails.** Render S20–S23 only after every simulated probe/heatmap/route annotation has `SCHEMATIC`, `TOY`, or `MEASURED` provenance in scene metadata. A real numerical chart requires recorded source data and an explicit provenance/metric label. **Deliverable:** extended cut with scientifically compliant interpretation section.

**Milestone E — film finishing.** Record VO, synchronise beats, master music safely below narration, add subtitles, QA the 25-minute core cut and full edition, export stills and thumbnails, and run accessibility checks. **Deliverable:** master MP4, chaptered cut, SRT, still figure pack, and render manifest.

### 7.4 Objective acceptance criteria

- **Mathematical:** All displayed dimensions align; SSD recurrent and matrix forms match to tolerance on random toy inputs (float64 `rtol=1e-10`, `atol=1e-10` for short tests); causal masks and decay directions correct; no softmax in SSD equation.
- **Complexity:** Pre/decode/memory cost labels are separated; per-token O(T) caveat for attention; no blanket GPU speedup from asymptotics; finite chunk vs full-sequence matrices clear.
- **Science:** Every specific empirical claim is attributed, including origin models and dates; unmeasured interpretability plots are watermarked; no guaranteed semantic MoE specialists; no fixed-state *must-forget* theorem.
- **Pedagogy:** A viewer can explain query vs key vs value, prefill vs cache, Mamba's selective update, Mamba-2's `a_t I` restriction, SSD matrix identity, and MoE top-k routing without watching an equation-heavy appendix.
- **Design:** At least 12 polished, self-contained still frames; no clipped math or overlapping labels; stable colour semantics; meaningful transition into every equation; avoid excessive decorative particles.
- **Render:** all S01–S23 run individually at preview resolution before final encoding; no missing font/glyph/LaTeX errors; version/asset manifests stored alongside output.

### 7.5 Two tiny math tests agents must perform before animation

```python
import numpy as np

def ssd_recurrent(a, K, Q, V):
    T, N = K.shape
    P = V.shape[1]
    S = np.zeros((N, P), dtype=np.float64)
    Y = np.zeros((T, P), dtype=np.float64)
    for t in range(T):
        S = a[t] * S + np.outer(K[t], V[t])
        Y[t] = Q[t] @ S
    return Y

def ssd_matrix(a, K, Q, V):
    T = len(a)
    L = np.zeros((T, T), dtype=np.float64)
    for t in range(T):
        survival = 1.0
        for s in range(t, -1, -1):
            L[t, s] = survival
            survival *= a[s]
    return (L * (Q @ K.T)) @ V

rng = np.random.default_rng(2026)
for T in (1, 3, 8, 17):
    N, P = 5, 7
    a = rng.uniform(0.1, 0.95, size=T)
    K, Q = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    V = rng.normal(size=(T, P))
    np.testing.assert_allclose(ssd_recurrent(a, K, Q, V),
                               ssd_matrix(a, K, Q, V), rtol=1e-10, atol=1e-10)
```

The update order in this test uses `a[t]` to scale the previous state, while the new write at time `t` has factor one. The matrix generator accordingly applies `a[s]` after recording the contribution from source `s`; it uses subsequent factors for earlier sources, as the analytic derivation requires. Test the 4×4 numeric `L` displayed in S12/S14 separately.

---

## 8. Short voiceover outline (designed to become full script, not the final narration)

| Chapter | Narration premise | Required contrast |
|---|---|---|
| 1 | “One word begins as an embedding and ends as a contextual representation.” | bank/riverside vs finance |
| 2 | “Attention tells us where to look and what to copy.” | Q matches K; V transports content |
| 3 | “Caching prevents unnecessary repetition, not growing context costs.” | prefill vs decode vs KV memory |
| 4 | “Recurrence trades indexed history for a fixed-shaped evolving state.” | addressability vs integrated state |
| 5 | “Mamba learned *when* to update; Mamba-2 changed *how* to compute efficiently.” | S4 / Mamba-1 / SSD |
| 6 | “The same operation admits a recurrent and an attention-shaped equation.” | exact SSD equivalence, no softmax |
| 7 | “Even after tokens communicate, something must transform their representations.” | attention/SSM vs FFN |
| 8 | “MoE buys capacity by activating only some FFNs per token.” | stored vs active vs dispatched |
| 9 | “Fast Transformers already use clever memory, window, and IO tricks.” | GQA/SWA/Flash/Paged/MLA |
| 10 | “Interpretability starts where the cartoon stops: we must intervene.” | state-patch vs source-KV-patch |
| 11 | “Similar learned circuits are possible even with different memory substrates.” | universality of approximation vs mechanisms |

### Suggested final narration (last 20 seconds)

> “Theo keeps the old cards. Manu and Max carry the past forward in recurrent states. The workshop decides which transformations to apply. In a hybrid network, all of them can write into the same residual stream. But resemblance is not mechanism: to discover what the model actually used, we must trace the source, change the computation, and watch the answer.”

---

## 9. Source and claim audit

### 9.1 Provided report — how it is used

**Provided primary context:** *From Recurrent Memory to Addressable Retrieval: A Mechanistic Interpretability Research Programme for Hybrid Mamba-2–Attention Language Models* (9 September 2026), uploaded PDF, 50 PDF pages. The film uses its formulation as a **hypothesis and experimental design**, not as evidence of observed behaviour:

- Report ch. 1–2, PDF pp. 7–12: notebook versus index cards; query-independent `G`, bindings `B_i`, source positions `P_i`, selected target `Y`; residual shared workspace; recurrent-vs-token-indexed memory; crucial alternative current state / past source KV / both.
- Ch. 3, PDF pp. 13–16: prior Mamba interpretability, universality SAEs, recall deficits, and already-known attention-layer recall probes; research novelty and caution.
- Ch. 4–5, PDF pp. 17–21: Entity Tracking, binding/query/position swaps, MQAR, calendar geometry, behaviour-first gates, and matched probing.
- Ch. 6–8, PDF pp. 22–27: SAEs, linear/DAS, geometry, and causal controls.
- Ch. 9–11, PDF pp. 28–34: causal path attribution and stress tests.
- Appendices, PDF pp. 41–46: state unrolling, architectural memory growth, patch definitions, leakage controls and intervention details.

These are summarised/reinterpreted for teaching. **Any apparent claim “attention retrieves from the Mamba state” must remain a question until the current-state and source-KV contributions are disentangled.**

### 9.2 External primary research and implementation references

1. Vaswani et al. (2017), **Attention Is All You Need** — standard multi-head attention / Transformer. <https://arxiv.org/abs/1706.03762>
2. Gu & Dao (2023/2024), **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — input-dependent selective SSM and parallel selective scan. <https://arxiv.org/abs/2312.00752>
3. Dao & Gu (ICML 2024), **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality** — rigorous SSD and Mamba-2. <https://proceedings.mlr.press/v235/dao24a.html>
4. Dao & Gu (2024), **State Space Duality Part I: The Model**, **Part II: Theory**, **Part III: Algorithm** — original researcher derivations, scalar-identity structure and chunk procedure. <https://goombalab.github.io/blog/2024/mamba2-part1-model/>; <https://goombalab.github.io/blog/2024/mamba2-part2-theory/>; <https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems>
5. Shazeer (2019), **Fast Transformer Decoding: One Write-Head is All You Need** — MQA. <https://arxiv.org/abs/1911.02150>
6. Ainslie et al. (EMNLP 2023), **GQA** — grouped-query attention and uptraining. <https://aclanthology.org/2023.emnlp-main.298/>
7. Beltagy, Peters & Cohan (2020), **Longformer** — sliding + global attention. <https://arxiv.org/abs/2004.05150>
8. Mistral AI (2023), **Mistral 7B** — GQA and sliding-window attention in a decoder LLM. <https://mistral.ai/news/announcing-mistral-7b/>
9. Dao et al. (NeurIPS 2022), **FlashAttention** — IO-aware exact attention. <https://papers.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html>
10. Kwon et al. (SOSP 2023), **Efficient Memory Management for Large Language Model Serving with PagedAttention** — vLLM. <https://arxiv.org/abs/2309.06180>
11. DeepSeek-AI (2024), **DeepSeek-V2** — MLA and sparse MoE. <https://arxiv.org/abs/2405.04434>
12. Shazeer et al. (2017), **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. <https://arxiv.org/abs/1701.06538>
13. Fedus, Zoph & Shazeer (2021/2022), **Switch Transformers** — token-level top-1 routing and capacity balancing. <https://arxiv.org/abs/2101.03961>
14. Jiang et al. (2024), **Mixtral of Experts** — sparse top-2 of 8 expert FFNs. <https://arxiv.org/abs/2401.04088>
15. Dai et al. (2024), **DeepSeekMoE** — fine-grained experts and shared experts. <https://arxiv.org/abs/2401.06066>
16. DeepSeek-AI (2024/2025), **DeepSeek-V3 Technical Report** — large-scale MoE training and load balancing. <https://arxiv.org/abs/2412.19437>
17. NVIDIA (2025), **Nemotron 3 Nano 30B-A3B** technical report / architecture reference. <https://github.com/NVIDIA-NeMo/Nemotron/blob/main/skills/nemotron-nano3/paper/architecture.md>
18. Cooper et al. (ICML 2026), **Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models** — task-specific theoretical separations. <https://proceedings.mlr.press/v306/cooper26a.html>
19. Yun et al. (ICLR 2020), **Are Transformers universal approximators of sequence-to-sequence functions?** — formal universality under assumptions. <https://research.google/pubs/are-transformers-universal-approximators-of-sequence-to-sequence-functions/>
20. Wang et al. (ICLR 2025), **Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures** — feature/circuit similarities, off-by-one motif. <https://proceedings.iclr.cc/paper_files/paper/2025/hash/e3c3473812173643147170188ef2b141-Abstract-Conference.html>
21. Kim & Schuster (ACL 2023), **Entity Tracking in Language Models**; Arora et al. (ICLR 2024), **Zoology / MQAR**; Liu et al. (TACL 2024), **Lost in the Middle**. <https://aclanthology.org/2023.acl-long.213/>; <https://openreview.net/forum?id=LY3ukUANko>; <https://aclanthology.org/2024.tacl-1.9/>
22. Manim Community official docs, **example gallery / rendering / voiceovers / TransformMatchingTex**. <https://docs.manim.community/en/stable/>; <https://docs.manim.community/en/stable/guides/add_voiceovers.html>
23. Community agent resource (third party), **edoardob90/manim-skills**. <https://github.com/edoardob90/manim-skills>

**Source QA:** before recording narration, validate current primary-paper details and the exact selected Nemotron checkpoint config. A technical report about a *proposed experiment* does not establish measured claims. Do not insert arbitrary empirical numbers in the presentation.

---

## 10. Final visual message and scope boundaries

The film should answer four independent questions, in this order:

1. **What representation is available?** Token embeddings are transformed into contextual, layer-dependent residual vectors.
2. **How does a layer exchange information across positions?** Attention provides query-dependent access to indexed K/V representations; Mamba updates an internal recurrent state; Mamba-2's SSD core admits both recurrent and masked-attention-like formulations.
3. **Where is computation spent?** Dense FFNs are often a large parametric/FLOP component; MoE sparsifies token-wise FFN compute while adding routing/memory/communication trade-offs; actual efficiency depends on GPU and workload.
4. **What does the trained model really do?** Architecture motivates hypotheses about memory; probes, sparse features, geometry, and especially targeted causal interventions are needed to identify mechanisms.

**Do not rank the brothers as “smartest”.** The lesson is that these architectural choices distribute *addressability*, *state*, *parallelism*, and *conditional computation* differently. The scientific ending is **an intervention experiment the viewer now understands how to design**, rather than a simplistic winner.
