# The Three Brothers: a Manim talk on Transformers, Mamba, Mamba-2 and MoE

A visual explanation for researchers who already know Transformers, in 24 slides. It runs about 7 minutes as a video and about 30 minutes as a live talk. It covers attention and the KV cache; Mamba and Mamba-2, with the full SSD derivation; FFNs and MoE; the efficiency–expressivity trade-off; universality; and what interpretability finds inside both families.

| Path | What it is |
|---|---|
| `TALK_SCRIPT.md` | What to say on every slide, click by click, plus a formula cheat sheet, likely questions and sources |
| `slides/sNN_name.py` | **One file per slide.** Each file has one scene class, and every animation beat is one `animate_*` function with a typed signature and a docstring |
| `visual/` | The shared toolkit, imported with `from visual import *` (see below) |
| `math_contracts/test_ssd_equivalence.py` | NumPy checks behind every formula on screen: recurrence = matrix form = chunked algorithm = linear attention, plus parity with negative decays |
| `agent_skill/SKILL.md` | Brief for coding agents (references the `edoardob90/manim-skills` Manim skill) |
| `reference/original_plan/` | The original storyboard this talk grew out of |

**`visual/`**

| Module | Contents |
|---|---|
| `theme.py` | palette, fonts, `T()` text, `M()` plain maths |
| `equations.py` | `ceq()` colour-coded maths; `morph()` (one substitution step); `collapse()` (replace a derivation by its result); `residual_eq()` |
| `shapes.py` | `card`, `pill`, `chip`, `token_card`, `heat`, `triangle`, `sheet`, `bars`, `glow`, `arrow`, … |
| `characters.py` | `brother()` |
| `deck.py` | the `Deck` base class: backdrop, progress dots, `title()`, `pause()` (one click), `clear_body()`, `clear_all()`, and the `DECK_CHECK=1` layout checker |

**Colour code for equations.** Every symbol keeps one colour on every slide:
- query `q Q` cyan
- key `k K` purple
- value `v V` yellow
- state `S h` coral
- decay and mask `a L γ Δ` mint
- attention and windows `W` blue
- outputs white

Write equation parts atomically, as in `ceq("S_t", "=", "a_t", "S_{t-1}", "+", "k_t", r"v_t^\top")`, and the colours follow automatically.

## Slides

| File | Scene | Slide | Animation functions |
|---|---|---|---|
| `slides/s01_intro.py` | `S01Intro` | Three brothers, three memories. | `animate_title`, `animate_story`, `animate_memories`, `animate_question` |
| `slides/s02_qkv.py` | `S02QKV` | Attention in one picture: query, key, value, softmax, heads. | `animate_projections`, `animate_scores`, `animate_softmax`, `animate_blend`, `animate_heads` |
| `slides/s03_kv_cache.py` | `S03KVCache` | Prefill → KV cache → decode. | `animate_prefill`, `animate_decode`, `animate_comparison` |
| `slides/s04_bank.py` | `S04Bank` | "bank": the same embedding drifts to two meanings in the residual stream. | `animate_start`, `animate_layers` |
| `slides/s05_ladder.py` | `S05Ladder` | Long context, then the ladder RNN → Mamba → Mamba-2 (each fixes the last one's problem). | `animate_growing_wall`, `animate_rnn`, `animate_mamba`, `animate_mamba2` |
| `slides/s06_ssd_state.py` | `S06SSDState` | Mamba-2 in one picture: an N×P sheet that fades, gets a rank-1 write, and is read. | `animate_fade`, `animate_write`, `animate_read` |
| `slides/s07_ssd_unroll.py` | `S07SSDUnroll` | Unroll the notebook one step at a time, then replace the steps by the closed form. | `animate_unroll`, `animate_token_weights`, `animate_read`, `animate_matrices`, `animate_softmax_contrast` |
| `slides/s08_ssd_chunks.py` | `S08SSDChunks` | Fast to train: chunks, low-rank far blocks, the four SSD steps and the cost. | `animate_chunked_matrix`, `animate_low_rank`, `animate_steps_and_cost` |
| `slides/s09_ffn.py` | `S09FFN` | Why a feed-forward network, and why it is the expensive part. | `animate_mixing`, `animate_per_token`, `animate_cost_split` |
| `slides/s10_moe.py` | `S10MoE` | MoE as a hospital: a triage nurse (router) opens 2 of 8 specialist rooms per patient (token). | `animate_routing` |
| `slides/s11_moe_production.py` | `S11MoEProduction` | How modern LLMs run MoE: shared + fine-grained experts, load balancing, expert parallelism. | `animate_fine_grained`, `animate_balancing`, `animate_expert_parallel` |
| `slides/s12_bus.py` | `S12Bus` | The residual stream is one shared bus; the Mamba state runs along time instead. | `animate_belt`, `animate_two_axes` |
| `slides/s13_toolbox.py` | `S13Toolbox` | Real Transformers are cleverer: ten tricks, who introduced them, who uses them. |  |
| `slides/s14_measured_cost.py` | `S14MeasuredCost` | The bill, measured: Transformer ÷ Mamba memory and time as the context doubles. | `animate_bars`, `animate_oom`, `animate_why` |
| `slides/s15_recall.py` | `S15Recall` | Recall needs memory that grows: a phone book into a wall vs into a fixed notebook. | `animate_stream`, `animate_store`, `animate_query`, `animate_bound` |
| `slides/s16_hybrids.py` | `S16Hybrids` | Neither pure model is cheap: function composition and why hybrids win. | `animate_task`, `animate_window`, `animate_hybrid`, `animate_results` |
| `slides/s17_state_tracking.py` | `S17StateTracking` | Tracking state: a blind spot for both, and the one-line fix for parity. | `animate_shell_game`, `animate_parity`, `animate_formula` |
| `slides/s18_universality.py` | `S18Universality` | Universality: all three families can approximate anything — the question is the size. | `animate_proofs`, `animate_price`, `animate_bounds` |
| `slides/s19_derivation.py` | `S19Derivation` | Deriving Mamba-2's attention form, starting from softmax attention. | `animate_linear`, `animate_regroup`, `animate_rnn`, `animate_fade`, `animate_meet`, `animate_check`, `animate_family` |
| `slides/s20_shared_features.py` | `S20SharedFeatures` | Same features, stretched depth (Wang et al., ICLR 2025, "Towards Universality"). | `animate_stacks`, `animate_scores`, `animate_example` |
| `slides/s21_induction.py` | `S21Induction` | Same algorithm, different wiring: the induction circuit in a Transformer and in Mamba. | `animate_pattern`, `animate_transformer`, `animate_mamba`, `animate_off_by_one` |
| `slides/s22_representation_flow.py` | `S22RepresentationFlow` | Where the representations differ: token diversity across depth, and what attention looks like. | `animate_curves`, `animate_causes`, `animate_hidden_attention` |
| `slides/s23_tools_transfer.py` | `S23ToolsTransfer` | The interpretability toolbox transfers from Transformers to Mamba. | `animate_tools` |
| `slides/s24_closing.py` | `S24Closing` | Closing: three memories merge into one residual vector. | `animate_merge` |

## Build

```bash
python -m venv .venv && . .venv/bin/activate
pip install "manim==0.19.0" manim-slides numpy pytest   # also needs FFmpeg, pango dev headers, LaTeX + dvisvgm, the Inter font
python -m pytest math_contracts
scripts/render.sh l slides/s07_ssd_unroll.py     # one slide, 480p15 (no files = all slides, in parallel)
scripts/build_video.sh m                         # all slides at 720p30 → media/final/three_brothers_720p30.mp4
DECK_CHECK=1 scripts/render.sh l                 # also print overlapping / off-frame text into media/logs/*.log
```

For a live, click-stepped deck (each click matches a ▶ in `TALK_SCRIPT.md`):

```bash
MANIM_DECK=slides manim-slides render slides/s07_ssd_unroll.py S07SSDUnroll
manim-slides present S07SSDUnroll
```
