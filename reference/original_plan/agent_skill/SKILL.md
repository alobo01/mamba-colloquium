---
name: hybrid-llm-manim-film
summary: Implement the research-audited 3Blue1Brown-inspired Manim storyboard for Transformer, Mamba-1, Mamba-2, MoE, and mechanistic interpretability.
---

# Hybrid LLM architecture explainer — Manim production skill

## Before generating code

1. Read `../TRANSFORMERS_MAMBA2_MOE_MANIM_STORYBOARD.md` in full and treat **§4** as the mathematical contract and **§9** as the source/claim contract.
2. Consult the third-party Manim Community skill at <https://github.com/edoardob90/manim-skills> (`manim/SKILL.md`, `animations.md`, `plotting.md`, `rendering.md`, `gotchas.md`). This external resource targets ManimCE 0.20.1; verify changes in <https://docs.manim.community/en/stable/>. Do not claim the skill is already installed.
3. Identify whether ManimCE, FFmpeg, and LaTeX are installed. If absent, prepare environment declarations but do not claim successful renders.
4. Identify whether actual probing/intervention results exist. If not, force metadata label `SCHEMATIC` in S21–S23.

## Film-specific instructions

- Three brothers Theo/Manu/Max represent indexed attention memory / Mamba selective state / Mamba-2 blockwise SSD, respectively. MoE is a workshop **orthogonal** to the memory mixer.
- Use one colour per mathematical object across scenes: Q cyan, K purple, V yellow, residual white, attention blue, Mamba coral, Mamba-2 mint, MoE gold.
- Token `bank` starts as an illustrative identical embedding in two contexts and contextualises differently through residual updates. Do not misrepresent positional embeddings in a RoPE model.
- At each scene boundary, explicitly declare the persistent mobjects and their exit/entry positions for compositing. All scenes must render independently; avoid hidden dependence on previous scene object state.
- Do not present KV cache as prefill. Do not show cached attention decode as O(1) in prompt length.
- SSD core is `S[t]=a[t]*S[t-1]+outer(K[t],V[t])`, `Y[t]=Q[t]@S[t]`, identical to `(L*(Q@K.T))@V`. Never add softmax to the SSD identity.
- Use `MathTex` or TeX-aware transforms for identities, and run the NumPy equivalence tests before scene render.
- Distinguish model memory and compute costs from *sequence mixer only* costs. Handle grouped query heads in attention illustrations.
- Mark synthetic interpretability heatmaps and invented router weights as `TOY` or `HYPOTHESIS`; never narrate them as findings.
- Reuse scene IDs S01–S23 and class names such as `S12SSDMatrixDuality`. Produce source code, test logs, preview renders, and a manifests file of verified exports.

## Mandatory production loop

- Implement a small mobject primitive → render low quality (`manim -ql`) → inspect output frames for clipping, timing, and correctness → adjust → render medium → final high quality.
- All GPU/performance graphs must distinguish analytic order-of-growth from empirical benchmark measurements. Do not invent wall-clock points.
- Every primary mathematical identity receives a deterministic visual unit test or a companion reference check.
- Preferred built-ins: `Scene`, `VGroup`, `MathTex`, `TransformMatchingTex`, `AnimationGroup`, `LaggedStart`, `ValueTracker`, `always_redraw`; use Manim Voiceover only if narration production is requested and the package is available.

## Completion checklist

- [ ] S04 shows two different *contextualised* residuals and identical starting token embedding only when token identity actually matches.
- [ ] S02 gives Q/K/V distinct functions and includes multiple heads.
- [ ] S03 distinguishes prefill, cache, O(T) decode attention work and O(T) cache space.
- [ ] S07 acknowledges Mamba-1's parallel selective scan.
- [ ] S10–S14 match matrix dimensions, have exact numeric SSD equality and correct decay products.
- [ ] S16–S17 shows token-level expert top-k and real dispatch costs.
- [ ] S18 never claims FlashAttention eliminates quadratic dense attention FLOPs.
- [ ] S20 distinguishes Mamba state across tokens from residual stream across layers.
- [ ] S21–S23 distinguishes state-only retrieval, source KV retrieval and mixed mechanism, tests with interventions.
- [ ] All synthetic / schematic interpretations visibly labelled.
- [ ] Film content and still frames have sources/attributions and an original visual identity.
