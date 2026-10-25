# The Three Brothers: talk script

A talk on how Transformers, Mamba, Mamba-2 and mixture-of-experts remember, compute and scale. It then covers what theory says about their trade-off, and what interpretability finds inside them.
Every slide is one file in `slides/` (`s07_ssd_unroll.py` is slide 7). Inside each file, every animation is one function. The video export plays the same slides back to back.

**Length.** The full talk is about 30 minutes: 24 slides, roughly 75 seconds each.
- **Short talk, about 12 minutes.** Play slides **1, 2, 3, 5, 6, 7, 10, 15, 16, 18, 20, 21, 24**.
- **Theory-only add-on, about 10 minutes.** Play slides **14–23**.

**How to read it.** Each slide has:
- an *idea* in one line;
- what is *on screen*;
- the words to *say*.

A line starting with **▶** is a click (`next_slide`); what follows it is what you say once that click has happened. The *Say* text is written to be spoken, so put it in your own words rather than reading it.

**The story in one breath.**
- Attention keeps every token, so it remembers exactly but gets expensive as the context grows.
- Mamba keeps a small notebook instead. Mamba-2 reorganises that notebook so GPUs can train it fast, and the maths shows it is attention with a decaying memory.
- Feed-forward layers and mixtures of experts add knowledge. The residual stream connects everything.
- Theory says neither pure design is cheap for every task. Both are "universal", but at very different sizes, and hybrids get the best of both.
- Inside, the two families learn surprisingly similar features and circuits, wired slightly differently.

**The cast.** Theo keeps one card per token (attention). Manu keeps one small notebook (Mamba). Max keeps the same kind of notebook, organised into tiles that GPUs love (Mamba-2).

**Colour code.** Every symbol keeps its colour on every slide, so long equations can be read by colour:

| Colour | Means |
|---|---|
| cyan | query **q, Q** (Mamba's C) |
| purple | key **k, K** (Mamba's B) |
| yellow | value **v, V** (Mamba's x) |
| coral | recurrent state **S, h** (and Manu) |
| mint | decay / mask **a, L, γ, Δ** (and Max) |
| blue | attention, softmax, windows **W** (and Theo) |
| gold | experts and router (MoE) |
| white | outputs and the residual stream |

---

## Part 1 · Theo's archive

## 1 · Three brothers
**Idea:** Same question, three ways to remember.
**On screen:** Theo, Manu and Max. A sentence runs across the top. Question: *Where is Leo's key?*

**Say:**
"Leo hid the green key under the bridge. Then someone asks: where is Leo's key? Three brothers have to answer, and each remembers the story differently."

▶ "Theo pins every word on a wall, one card per token. Manu has a single notebook of fixed size, and every new word is written into it, so old words slowly fade. Max has the same kind of notebook as Manu, but organised in tiles so a GPU can fill it in bulk."

▶ "Watch the question. Theo walks to the card that says 'bridge' and reads it. Manu and Max read their notebooks. Theo is exact but his wall keeps growing. The other two have a fixed size but must compress. That is the whole talk: what does each memory cost, what does it lose, and do they end up computing the same thing?"

**Bridge:** "Let's start with Theo and remind ourselves how his wall works."

---

## 2 · Attention in one picture
**Idea:** Query asks, key advertises, value delivers. Heads let you ask several questions at once.
**On screen:** a token passes through three lenses; a row of index cards; bars that become percentages; three small attention maps and the formula.

**Say:**
"Every token is turned into three vectors by three learned lenses. The **query** says what I am looking for. The **key** says how I can be found, like the label on a card's spine. The **value** is what I hand over if you pick me."

▶ "Now a query goes along the wall. For each card we take the dot product of the query with its key: a score for how well they match."

▶ "Softmax turns the scores into shares that add up to a hundred percent."

▶ "The output is a blend of the values, weighted by those shares. Nothing is looked up exactly; it is a soft weighted average."

▶ "Why many heads? Each head has its own lenses, so one can follow who-did-what, another the nearby words, another something far back. We concatenate the heads and mix them with an output matrix. Notice the colours in the formula: cyan Q, purple K, yellow V. They stay the same for the rest of the talk."

**Bridge:** "How do we run this when we generate one token at a time?"

---

## 3 · Prefill → KV cache → decode
**Idea:** Cache the past keys and values. Each new token then costs O(T), not O(T²).
**On screen:** a triangle of scores, a row of cards (the cache), a memory bar, two pills.

**Say:**
"First the prompt goes in all at once. This is **prefill**: every token attends to all earlier tokens, about T-squared scores, but in parallel."

▶ "With a causal mask, past keys and values never change. So we keep them: that is the **KV cache**. Each new token computes its own query, key and value, appends to the cache, and sweeps one row across it. That is **decode**."

▶ "Without the cache we would redo the whole triangle for every token. With it, each token costs work proportional to T, and memory grows with T. It still grows. That is the problem the next slides try to solve."

**Bridge:** "Before we fix the cost, let me show what flows through these layers."

---

## 4 · Tokens evolve in the residual stream
**Idea:** A token starts as a lookup-table vector and is rewritten by every layer. Context makes the same word end up in different places.
**On screen:** a river bank and a money bank, both starting from the same arrow; layers add arrows; the ends diverge.

**Say:**
"A token begins as an embedding, a row of a lookup table. 'Bank' starts as the *same* vector in both sentences."

▶ "Every layer reads the current vector and adds an update. Blue is what attention writes, gold what the feed-forward block writes. The river bank drifts towards 'river', the money bank towards 'money'. The running sum is the **residual stream**: next stream = current stream + what the block computes from it."

**Bridge:** "That is Theo's world. Why would we want something else?"

---

## Part 2 · Notebooks: Mamba and Mamba-2

## 5 · From the growing wall to Mamba-2
**Idea:** Long context makes the wall too big. RNNs fix that but have problems. Mamba fixes those and has a new one, which Mamba-2 fixes.
**On screen:** Theo's wall and a cost curve; then three columns, RNN → Mamba → Mamba-2.

**Say:**
"Theo's wall keeps growing, and attention cost grows with the square of the context."

▶ "What if the memory had a *fixed* size? That is the old idea of recurrent networks."

▶ "**RNNs** have two problems. Training is sequential, like dominoes. And the learning signal fades over long distances."

▶ "**Mamba** fixes both. The step size Δ depends on the input, so the model chooses what to write and what to forget. A parallel scan trains all tokens at once. Its remaining problem: a scan is not a matrix multiplication, and GPUs are built for matrix multiplication. So the state had to stay small."

▶ "**Mamba-2** uses one decay number per head, which turns the computation into matrix multiplications. Bigger state; the paper measures the core layer as two to eight times faster."

**Bridge:** "Let's open Max's notebook."

---

## 6 · Mamba-2 in one picture
**Idea:** The memory is one N×P sheet. Each token fades it, writes a note, reads it.
**On screen:** the coral sheet, the colour-coded update rule, three labels: fade / write / read.

**Say:**
"Mamba-2's memory is a sheet of N by P numbers, coral like every state in this talk, and it never grows."

▶ "**Fade.** Multiply the sheet by the mint number a-t, between zero and one."

▶ "**Write.** Add the purple key as a column times the yellow value as a row: a rank-one note."

▶ "**Read.** The cyan query reads the sheet and produces the output. Since the sheet has a fixed size, decoding costs the same at token ten and at token one million."

**Bridge:** "Now let's unroll this recurrence, one step at a time."

---

## 7 · Unroll it: attention with a decay mask
**Idea:** Substitute the recurrence into itself, step by step. The closed form is Y = (L ⊙ QKᵀ)V: attention whose mask is a decay, not a softmax.
**On screen:** S₁, S₂, S₃ written by substitution, with three tokens dimmed by their surviving weight. The lines collapse into one boxed sum. Then the read-out, the matrix form, and the L, QKᵀ and masked heatmaps.

**Say:**
"Let's just run the notebook. After the first token, the state is its note, k₁ v₁ᵀ. For the second token, fade and write: a₂ S₁ + k₂ v₂ᵀ. Now substitute S₁ (watch it fly in): a₂ k₁ v₁ᵀ + k₂ v₂ᵀ. Third token, same move: a₃ times all of that, plus the new note. Look at the mint factors. The first note has been faded twice, a₃a₂; the second once; the newest not at all. On the right, the three tokens are dimmed by exactly those weights: 0.4, 0.5 and 1."

▶ "The pattern is obvious, so replace all three lines with the general rule: the state is a sum of every past note, each faded by the product of the decays since it was written."

▶ "Now read it with the query. y-t is S-t transpose q-t, so each past value comes back weighted by two things: the mint decay product, which we call L-ts, and the usual query-key match. Do that for every row at once and it is a matrix product. Then collapse again: Y equals L element-wise times Q K-transpose, times V."

▶ "Here are the matrices. L has ones on the diagonal and decays below. The bottom-left entry, 0.3, is just 0.8 times 0.5 times 0.75. Multiply element by element with the usual score matrix, and that is Mamba-2."

▶ "Compare with softmax attention, top right. Same recipe, different mask. Attention normalises with softmax and keeps every token. Mamba-2 uses a decay and keeps a fixed-size summary."

**Bridge:** "That is the maths. Why is it fast to train?"

---

## 8 · Fast to train: chunks and low rank
**Idea:** Chunk the sequence. Far-away blocks have low rank, so they collapse into a small state that is passed along.
**On screen:** a chunked triangular matrix, one block factorised, four numbered steps, the colour-coded cost.

**Say:**
"Cut the sequence into chunks. Inside a chunk the block is small and dense, so we just multiply matrices."

▶ "Blocks far from the diagonal have rank at most N. Each is a product of two skinny matrices: everything the later chunk needs from the earlier one fits in the small state."

▶ "Four steps: do each chunk, summarise it into a state, pass the states along, add the past back in. Training is linear in T; decoding is O(N·P) per token, independent of context length."

**Bridge:** "We've covered how tokens mix. What do the other layers do?"

---

## Part 3 · Workshops and the bus

## 9 · Why a feed-forward network?
**Idea:** Attention moves information between tokens. The FFN thinks about each token alone, and it holds most of the parameters.
**On screen:** arcs between tokens; an hourglass per token; a bar split 1/3 attention, 2/3 FFN.

**Say:**
"Attention lets tokens talk. After the talk, each token still has to do something with what it heard."

▶ "That is the FFN: expand to four times the width, a non-linearity, squeeze back, for each token alone."

▶ "Attention is about four d-squared parameters, the FFN about eight. So the FFN is two thirds of each layer. Attention is a memory of this *context*; the FFN is a memory of the *training data*."

**Bridge:** "Can we make the FFN much bigger without paying for it?"

---

## 10 · MoE: a hospital with specialist rooms
**Idea:** Build many experts, send each token to only a few.
**On screen:** eight rooms, a router, tokens routed to two rooms each, the gold gating formula.

**Say:**
"A hospital with eight specialist rooms and a triage nurse at the door: the **router**."

▶ "A patient-token arrives; the nurse picks the two best rooms; the answers are blended with the nurse's weights. The next patient goes elsewhere."

▶ "All eight rooms exist, so you store all their parameters. Only two work per token, so you pay for two. The specialisations emerge during training; they are not neat human specialities."

**Bridge:** "Here is how today's big models make it work."

---

## 11 · How modern LLMs run MoE
**Idea:** Many small experts plus a shared one, a balanced load, and experts spread over GPUs.
**On screen:** three panels.

**Say:**
"Many small experts plus one always-on shared expert. DeepSeek-V3: 256 routed, 1 shared, 8 per token. Mixtral was simpler: 8 experts, 2 per token."

▶ "Keep the rooms equally busy. DeepSeek-V3 nudges a bias in the router instead of adding a loss term."

▶ "Experts live on different GPUs, so tokens travel there and back. That all-to-all traffic is the real cost of MoE."

**Bridge:** "What connects all these blocks?"

---

## 12 · The residual stream: one shared bus
**Idea:** All blocks talk through one running sum. Mamba's state runs along time; the residual runs along depth.
**On screen:** a belt with four blocks; a stack of writes; a grid with coral (time) and white (depth) arrows.

**Say:**
"Attention, MoE and Mamba-2 never talk directly. Each reads the stream and adds its result back. The stream is the sum of every write."

▶ "Two directions. Within a layer, Mamba's state carries information along *time*. The residual carries it along *depth*. Hybrids interleave a few attention layers among many Mamba-2 layers, and they can because they share this stream."

**Bridge:** "And Transformers themselves are cleverer than the plain version."

---

## Part 4 · The trade-off, and what theory says

## 13 · Real Transformers are cleverer than that
**Idea:** A toolbox of tricks shrinks the cache, the attention cost or the memory traffic.
**On screen:** ten tiles: name, who introduced it, where it ships.

**Say:**
"The model tricks:
- **MQA** (Shazeer 2019, PaLM) and **GQA** (Ainslie et al. 2023, Llama 2 70B and Llama 3) share keys and values between heads, which shrinks the cache.
- **Sliding windows** (Longformer, Mistral 7B) let each token see only recent tokens. Gemma 2 and 3 mix local and global layers.
- **Attention sinks** (StreamingLLM) keep long streams stable.
- **MLA** (DeepSeek-V2) compresses keys and values into a small latent.

And the systems tricks:
- **FlashAttention** computes exactly the same attention with far less memory traffic.
- **PagedAttention** (vLLM) pages the cache.
- **Sparse attention** (BigBird, DeepSeek NSA) skips most pairs.
- **RoPE + YaRN** handle position and stretch the context."

**Bridge:** "So how much does the bill actually differ in practice?"

---

## 14 · The bill, measured
**Idea:** On real hardware, the Transformer/Mamba cost ratio roughly doubles every time the context doubles, until the Transformer no longer fits.
**On screen:** paired bars (memory in blue, time in cyan) for 512 to 8k tokens; a dashed red "out of memory" bar at 8k; N²/N = N.

**Say:**
"A 2026 benchmark by Koledoye and colleagues compared a 50-million-parameter LLaMA with a 50-million-parameter Mamba on one A100. The bars show Transformer cost divided by Mamba cost. At 512 tokens the Transformer needs 1.35 times the memory; at 4k, 12.5 times the memory and 10.7 times the time."

▶ "At 8k tokens the Transformer ran out of memory; Mamba did not."

▶ "Why does the ratio double when N doubles? Attention work grows like N squared and Mamba's like N, so their ratio grows like N. The curves on slide 5 are not just theory."

*(Presenter note, not on screen: these are small, randomly initialised models in FP32 using the standard attention implementation. FlashAttention would remove most of the N² memory, but not the N² compute. The paper's fitted scaling equations do not reproduce its own crossover numbers, so quote the table, not the fits. Its "representational" results were measured on untrained models, so we don't use them.)*

**Bridge:** "So Mamba is cheaper. What does it lose?"

---

## 15 · Recall needs memory that grows
**Idea:** To recall n facts exactly you need at least n·log|V| bits of memory. Theo's wall grows to pay that; a fixed notebook cannot.
**On screen:** six key→value pairs stream in. Theo's wall keeps every card. The notebook has four slots, so pairs five and six land on top of pairs one and two (red slots). The query "E?" gets "4" from the wall and "3 or 4?" from the notebook. Then the bound.

**Say:**
"A phone book: A is 3, B is 7, and so on. Purple keys, yellow values."

▶ "Theo pins every entry. The notebook has four slots, so the fifth entry must share a slot with the first, and the sixth with the second. Nothing is 'forgotten' on purpose; it simply doesn't fit, and entries interfere."

▶ "Ask for E. Theo answers 4, exactly. The notebook's slot holds both A's 3 and E's 4, smeared together."

▶ "This is not a training problem; it is counting. To recall n arbitrary entries you need n log V bits somewhere. Jelassi and colleagues ('Repeat After Me', 2024) proved this for copying. A two-layer Transformer can copy very long strings, while a state-space model needs a state that grows with the string, and in their experiments Transformers learned to copy with far less data. The Zoology paper (Arora et al.) found that most of the quality gap between efficient convolution-style models and attention on real text comes from exactly this kind of associative recall."

**Bridge:** "So should we just use attention? Not so fast: attention has its own bill."

---

## 16 · Neither pure model is cheap
**Idea:** On tasks that need a long-range *control* signal plus a local *lookup*, pure SSMs need a big state and pure (windowed) Transformers need a long window. A two-layer hybrid needs neither.
**On screen:** the selective-copy prompt (#3 far back, the answer three tokens from the end). Three cards with lower bounds; the hybrid trick drawn on the row; then accuracy bars.

**Say:**
"This is the 2026 paper by Cooper, Diakonikolas, Ma and Sala, the one in your pack. The task is **selective copying**. Somewhere far back there is a number, here 3. At the end, output the token three positions back. A control variable from far away, then a local lookup. Many real tasks look like this: a question whose meaning depends on something said long ago."

▶ "A **pure SSM** has to keep enough of the recent tokens in its state to answer for any possible number. Their theorem: the total state, summed over layers, needs at least N log M bits. The state grows with the task."

▶ "A **pure Transformer** with sliding windows has the opposite problem: its windows must reach all the way back to the number. Total window at least on the order of L, the whole context. Full attention is the special case where the window *is* the context."

▶ "The **hybrid**: one Mamba layer, then one attention layer. Mamba is good at carrying one small thing forward, so it stamps '3' onto every later token. Now attention only needs a window of three. The state is about the vocabulary size, the window about N, and the parameters grow only polylogarithmically. Best of both."

▶ "And it is not just a construction. Trained from scratch at about two thousand parameters, the SSM-then-attention hybrid gets 99.9% on selective copy; both pure models are near 30%. The pure models need about six times the parameters to reach 90%. Order matters too: attention-then-SSM is no better than the pure models. The authors also report better length generalisation and robustness for hybrids, which matches why production models like Jamba and Nemotron-H are hybrids."

**Bridge:** "Recall and copying are about how *much* you remember. What about *tracking* something that changes?"

---

## 17 · Tracking state: a blind spot for both
**Idea:** Composing many permutations, like a five-cup shell game, is beyond both fixed-depth Transformers and Mamba (they sit in the complexity class TC⁰). Even parity needs a decay that can go negative.
**On screen:** five cups, a ball, five swaps. Then parity: bits arrive, and two lanes show a coral state arrow. With a ∈ [0,1] it only shrinks; with a ∈ [−1,1] it flips on every 1. Then the product formula.

**Say:**
"Hide a ball under a cup and swap the cups again and again. To answer, you must *compose* every swap in order. Merrill, Petty and Sabharwal ('The Illusion of State', 2024) showed that Transformers and diagonal state-space models like Mamba both live in the same small complexity class, TC⁰. Composing swaps of five objects is outside that class (unless a long-standing conjecture fails). So a fixed-depth model cannot track it for arbitrarily long sequences. Mamba's 'state' does not buy state tracking."

▶ "The smallest example is parity: is the number of ones odd? Use a single state number. If the decay must stay between 0 and 1, the state can only fade; it never changes sign, so it cannot flip with every 1. Allow the decay to be −1 and each 1-bit flips it. Three ones: it ends pointing down, odd."

▶ "In symbols: the state after T steps is the product of the decays times the start, which is −1 to the number of ones. Sarrof and colleagues showed the non-negative restriction is why Mamba fails parity, and Grazzi and colleagues (ICLR 2025) showed that allowing negative eigenvalues fixes parity and improves state tracking, even at 1.3B scale. Our test file checks both lanes."

**Bridge:** "So each architecture has tasks it can't do cheaply. Then what does it mean that they are all 'universal'?"

---

## 18 · Universal… but at what size?
**Idea:** All three families are universal approximators, but the constructions differ, and universality says nothing about the size you need. A lossless notebook is as big as a KV cache.
**On screen:** the universality statement. Three cards:
- Transformers: tags plus a lookup table.
- Linear RNN plus MLP: compress, then decode.
- Selective SSMs: gates reading a path.

Then the notebook grows to the width of Theo's wall, a recap of the three lower bounds, and "universal ≠ efficient".

**Say:**
"Universality means: for any reasonable sequence function and any tolerance ε, some setting of the weights gets within ε. All three families have such a theorem."

▶ "Each uses a different mechanism:
- **Transformers** (Yun et al. 2020): attention gives every context a unique tag, and the feed-forward layers act as a lookup table from tags to outputs.
- **Linear recurrences followed by an MLP** (Orvieto et al. 2023): the linear state compresses the input without loss, and the MLP decodes whatever function you want from it.
- **Selective SSMs** (Cirone et al. 2024): input-dependent gates let the state compute rich features of the whole input path. That is what selectivity adds over fixed filters."

▶ "Now the catch. To compress *without loss*, the notebook must hold everything, so it grows with the input: state proportional to T times d. That is a KV cache. Universality is cheap to claim; you get it back by letting the memory grow."

▶ "And the previous three slides are exactly the lower bounds on that size: recall needs n log V bits, selective copy needs N log M, and state tracking is out of reach at fixed depth. Universal does not mean efficient."

*(Presenter note, not on screen: precise statements need assumptions such as compact input domains, positional encodings and continuity. "State ∝ T·d" summarises the lossless constructions, not a tight bound for every function.)*

**Bridge:** "One more piece of theory: where exactly does Mamba-2's attention formula come from if you start from attention?"

---

## 19 · Deriving Mamba-2's attention form
**Idea:** Start from softmax attention, drop the softmax, regroup, name the sum, and add a fade. You arrive at the formula slide 7 reached from the notebook.
**On screen:** five rows, each with a step label. They collapse into the boxed Y = (L ⊙ QKᵀ)V, which two arrows (from attention and from the notebook) point into. Then matching numbers and the family of decays.

**Say:**
"Start with softmax attention."

▶ "Drop the softmax and keep only the causal mask. That is linear attention, Katharopoulos et al. 2020."

▶ "Look at one row t: a sum over past tokens of the query-key match times the value. Regroup it: because there is no softmax, we can sum the key-value outer products *first* and apply the query *afterwards*."

▶ "That bracket is a running sum, so give it a name: S-t. Each token adds one outer product. Linear attention is an RNN whose state is S."

▶ "Now insert one mint number in front of the old state: a fade. That is Mamba-2's recurrence."

▶ "Collapse all five lines: the result is Y equals L element-wise times QKᵀ, times V, exactly where slide 7 arrived from the notebook. Two roads, one formula."

▶ "And the numbers agree: our test code runs the recurrence and the matrix form on the same random inputs, and they match digit for digit. One template covers the family: a = 1 is linear attention, a fixed γ is RetNet, a learned input-dependent a is Mamba-2, and a full diagonal per channel is Mamba-1."

**Bridge:** "If the maths is this close, are the learned *insides* close too? That is where interpretability comes in."

---

## Part 5 · Looking inside

## 20 · Same features, stretched depth
**Idea:** Sparse autoencoders find largely the same features in a Transformer and in Mamba, at proportional depths.
**On screen:**
- Pythia (12 layers) and Mamba (24 thinner layers) stacked side by side, with purple links from layer ℓ to layer 2ℓ.
- The similarity score ρ, and three bars: 0.81, 0.73 and 0.68.
- One matched pair, "more or less".

**Say:**
"This is Wang et al., ICLR 2025, 'Towards Universality', also in your pack. They took Pythia-160M and Mamba-130M, trained on the same data, and trained a sparse autoencoder on every layer's residual stream to split it into features. For each Pythia feature they find the Mamba feature that fires on the same tokens, and record the correlation. When a Pythia feature lives in layer ℓ of 12, its twin usually lives near layer 2ℓ of 24: the same hierarchy, stretched."

▶ "How similar is similar? Train two SAEs on the *same* model: 0.81. Two Pythias with different seeds: 0.73. Pythia versus Mamba: 0.68. So changing the whole architecture costs only a little more than changing the random seed. More than a quarter of the features have a twin above 0.95. Raw neurons, by contrast, barely match: the shared structure is in features, not individual units."

▶ "An example: a feature for the phrase 'more or less' sits in Pythia's layer 6 and Mamba's layer 12, correlation 0.83. The simple features match best; complex ones match on theme but less on exact tokens, which the authors show is a complexity effect, not an architecture-specific feature."

**Bridge:** "Features match. What about algorithms?"

---

## 21 · Same algorithm, different wiring
**Idea:** Both architectures implement induction (A B … A → B). Transformers use two heads; Mamba uses a short convolution and the state, writing B one token late.
**On screen:** the row A B x … A → B; blue arcs for the Transformer's two heads; a coral bracket, belt and arrows for Mamba; the off-by-one pill.

**Say:**
"Induction: if you've seen A followed by B, and A appears again, predict B. It's the core of in-context copying."

▶ "In a Transformer (Olsson et al. 2022) this takes two heads. A **previous-token head** lets B record 'I came after A'. Later, an **induction head** at the second A searches for 'the token after A' and copies B."

▶ "Mamba has no heads. Using path patching, Wang et al. found that its short convolution, which looks at a few neighbouring tokens, merges A and B at the position *after* B. That combination is written into the SSM state of layer 17 of 24. The second A reads it back from the state."

▶ "So the algorithm is the same: mix the previous token in, store it as a key-value, retrieve it. But Mamba writes B's information one position late, the 'off-by-one' motif. They saw the same preference in another circuit (indirect object identification). Nobody knows yet why."

**Bridge:** "Where do the two architectures actually differ inside?"

---

## 22 · Where the representations differ
**Idea:** Across depth, Transformers first blur tokens together and recover late, while SSMs stay sharp early and blur deep. And Mamba's 'attention' is a smooth decay, not a sharp lookup.
**On screen:** two schematic curves of token diversity against depth, two reasons, then a softmax map beside a Mamba hidden-attention map with its formula.

**Say:**
"Hoang and colleagues (2025) tracked how different the tokens' representations are at each layer. In Transformers, tokens become similar to each other quickly, a kind of over-smoothing, and diversity comes back only in the last layers. In SSMs it is the opposite: tokens stay distinct early and homogenise deep in the stack. These curves are schematic shapes of that trend, not data."

▶ "Why? For Transformers, they trace it to the architecture itself. For SSMs, to training: an untrained Mamba stack keeps tokens diverse, so the smoothing is learned. This is a real representational difference, and it predicts different failure modes as context grows."

▶ "The second difference is what 'attention' looks like. Ali, Zimerman and Wolf (2024) wrote each Mamba layer as an implicit attention matrix: cyan c, purple b, and the mint product of decays in between. It's the slide-7 formula with a matrix decay. A softmax map is sharp: a few bright cells. Mamba's is smooth and fades with distance. And there is one map per channel, about a thousand times more maps than a Transformer has heads. With those maps, attribution methods built for Transformers give comparably good explanations on Mamba."

**Bridge:** "Which means most of our toolbox carries over."

---

## 23 · The interpretability toolbox transfers
**Idea:** The standard Transformer tools work on Mamba, and they find facts in the same place.
**On screen:** five tiles (tuned lens, state steering, causal tracing + ROME, knockout, SAEs + circuits), then one gold pill.

**Say:**
- "**Tuned lens** (Paulo, Marshall and Belrose 2024): read the residual stream mid-network as a next-token guess. It works on Mamba and RWKV too.
- They also found **steering** works. Better, you can steer the compressed **state** directly, which Transformers don't have.
- **Causal tracing and ROME** (Sen Sharma, Atkinson and Bau 2024): corrupt the subject, restore one activation at a time, and see which restores the fact. Rank-one editing then rewrites facts in Mamba too.
- **Knockout** (Endy et al. 2025): block information flow between chosen tokens. It was adapted from attention knockout to Mamba-1 and Mamba-2 by blocking the hidden attention from slide 22.
- And **SAEs and path patching**, the last two slides."

▶ "The headline result: in Mamba, as in GPT, facts are recalled by middle layers at the last token of the subject, and later layers act at the last token of the prompt. Different memories, the same place for knowledge. Endy et al. also find some flow patterns that differ between the families, so 'similar' is not 'identical'."

**Bridge:** "Let me close."

---

## 24 · Closing
**Idea:** Three memories, one stream.
**On screen:** three coloured paths, each with a one-line summary, merging into one white arrow; the final line.

**Say:**
"Theo remembers everything and pays per token. Manu keeps a fixed notebook and chooses what to keep. Max keeps the same notebook, but matmul-fast. Theory says none of them is cheap for every task, and the best systems mix them. Inside, they learn surprisingly similar features and circuits. Different memories; often similar computations."

---

## Cheat sheet

| Idea | Formula |
|---|---|
| Attention | O = softmax(QKᵀ / √d) V |
| KV cache | per-token decode work ∝ T, cache memory ∝ T |
| Residual stream | r⁽ℓ⁺¹⁾ = r⁽ℓ⁾ + block(norm r⁽ℓ⁾) |
| Mamba-2 state | S_t = a_t S_{t−1} + k_t v_tᵀ, y_t = S_tᵀ q_t |
| Unrolled | S_t = Σ_{s≤t} (a_{s+1}⋯a_t) k_s v_sᵀ |
| Decay mask | L_ts = a_{s+1} ⋯ a_t |
| Matrix form | Y = (L ⊙ QKᵀ) V |
| Mamba-2 cost | training linear in T, decode O(N·P) per token |
| FFN | ≈ 8d² parameters vs ≈ 4d² for attention |
| MoE | y = Σ over the top-k experts of g_i · E_i(x) |
| Recall bound | bits of state ≥ n log \|V\| |
| Selective copy (pure SSM) | Σ_i log \|S_i\| ≥ N log M |
| Selective copy (windowed TF) | Σ_i W_i ≥ Ω(L) |
| Parity with a scalar state | s_T = (Π a_t) s_0 = (−1)^{#1} s_0 if a_t = −1 on 1-bits |
| Hidden attention (Mamba) | α̃_ts = c_tᵀ (Π_{u=s+1}^{t} Ā_u) b̄_s |
| Feature similarity | ρ_i = max_j corr(f_i^Pythia, f_j^Mamba) |

## Questions you may get

- **Is Mamba-2 just attention?** The algebra has the same shape, but the mask is a decay instead of a softmax, and the memory is a fixed-size state.
- **Does the cache make decoding free?** No. Each token still costs work proportional to T and the cache grows with T. The cache saves the quadratic recompute.
- **Is FlashAttention subquadratic?** No. It computes exactly the same attention with much less memory traffic. It removes the N² *memory* in benchmarks like slide 14, not the N² compute.
- **Do experts specialise by topic?** Often partly, but not reliably.
- **Why can't Mamba copy long texts perfectly?** The state has a fixed size; see slide 15's counting argument.
- **If attention can recall anything, why bother with hybrids?** Slide 16: on tasks that need a long-range control signal, windowed attention must widen its window to the whole context. The SSM layer passes that signal along cheaply.
- **Are Transformers better at state tracking?** No. Slide 17: both families sit in TC⁰. Neither can compose permutations of five objects at arbitrary length without growing depth.
- **Does 'universal' mean Mamba can do everything a Transformer does?** In principle, with enough size. Slide 18: lossless compression needs a state as big as the cache.
- **Do Transformers and Mamba learn the same features?** Largely, by SAE matching (slide 20), with similarity close to that between two seeds of the same architecture. Circuits are analogous but wired differently (slide 21).
- **Is the slide-14 benchmark definitive?** No. It uses small untrained models in FP32 and the standard attention kernel. The trend is real; the exact numbers are specific to that setup.

## Sources

**Attention and tricks**
- Vaswani et al. 2017 (attention)
- Shazeer 2019 (MQA); Ainslie et al. 2023 (GQA)
- Beltagy et al. 2020 (Longformer); Xiao et al. 2023 (StreamingLLM)
- Dao et al. 2022 (FlashAttention); Kwon et al. 2023 (vLLM)
- Su et al. 2021 (RoPE); Peng et al. 2023 (YaRN)
- DeepSeek-V2 and V3 2024

**SSMs**
- Gu et al. 2022 (S4); Gu & Dao 2023 (Mamba); Dao & Gu 2024 (Mamba-2 / SSD)
- Katharopoulos et al. 2020 (linear attention)

**FFN and MoE**
- Geva et al. 2021 (FFN as memory)
- Shazeer et al. 2017; GShard 2020; Switch 2021; Mixtral 2024; DeepSeekMoE 2024

**Trade-off**
- Koledoye, Unachukwu, Nwobu & Rana 2026, *Benchmarking the Computational and Representational Efficiency of SSMs against Transformers on Long-Context Dyadic Sessions* (arXiv 2601.01237)
- Jelassi et al. 2024, *Repeat After Me* (ICML)
- Arora et al. 2023, *Zoology*
- Cooper, Diakonikolas, Ma & Sala 2026, *Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models* (arXiv 2603.08859)
- Waleffe et al. 2024, *An Empirical Study of Mamba-based Language Models*
- Merrill, Petty & Sabharwal 2024, *The Illusion of State in State-Space Models* (ICML)
- Sarrof, Veitsman & Hahn 2024, *The Expressive Capacity of State Space Models: A Formal Language Perspective*
- Grazzi et al. 2025, *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (ICLR)

**Universality**
- Yun et al. 2020, *Are Transformers universal approximators of sequence-to-sequence functions?* (ICLR)
- Orvieto et al. 2023, *On the Universality of Linear Recurrences Followed by Non-linear Projections*
- Cirone et al. 2024, *Theoretical Foundations of Deep Selective State-Space Models*

**Interpretability**
- Wang et al. 2025, *Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures* (ICLR)
- Olsson et al. 2022 (induction heads)
- Hoang et al. 2025, *A Comparative Analysis of Contextual Representation Flow in State-Space and Transformer Architectures* (arXiv 2510.06640)
- Ali, Zimerman & Wolf 2024, *The Hidden Attention of Mamba Models* (ACL)
- Paulo, Marshall & Belrose 2024, *Does Transformer Interpretability Transfer to RNNs?* (AAAI 2025)
- Sen Sharma, Atkinson & Bau 2024, *Locating and Editing Factual Associations in Mamba*
- Endy et al. 2025, *Mamba Knockout for Unraveling Factual Information Flow* (ACL)
