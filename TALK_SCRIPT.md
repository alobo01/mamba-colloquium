# The Three Brothers: talk script

A talk on how Transformers, Mamba, Mamba-2 and mixture-of-experts remember, compute and scale.
The slides are in `slides/`, and the same slides are exported as a 5-minute video. This file tells you what to say on each slide.

**Length.** Full talk about 20 minutes (16 slides, roughly a minute each). Short talk about 10 minutes: play only slides **1, 2, 3, 5, 6, 7, 10, 14, 15, 16**.

**How to read it.** Each slide has a one-line *idea*, what is *on screen*, and the words to *say*. A line starting with **▶** is a click (`next_slide`), and what follows it is what you say once that click has happened. Say the *Say* text in your own words; it is written to be spoken, not read.

**The story in one breath.** Attention keeps every token, so it remembers exactly but gets expensive as the context grows. Mamba keeps a small notebook instead. Mamba-2 reorganises that notebook so GPUs can train it fast, and the maths shows it is attention with a decaying memory. Feed-forward layers and mixtures of experts add knowledge; the residual stream connects everything. Real models mix all of these.

**The cast.** Theo keeps one card per token (attention). Manu keeps one small notebook (Mamba). Max keeps the same kind of notebook, organised into tiles that GPUs love (Mamba-2). Colour code throughout: blue = attention, coral = Mamba, mint = Mamba-2, gold = MoE, cyan/purple/yellow = query/key/value.

---

## 1 · Three brothers
**Idea:** Same question, three ways to remember.
**On screen:** Theo, Manu and Max. A sentence runs across the top. Question: *Where is Leo's key?*

**Say:**
"Leo hid the green key under the bridge. Noah hid the red key in the cupboard. Then someone asks: where is Leo's key? Three brothers have to answer, and each remembers the story differently."

▶ "Theo pins every word on a wall, one card per token. Manu has a single notebook of fixed size, and every new word is written into it, so old words slowly fade. Max has the same kind of notebook as Manu, but organised in tiles so a GPU can fill it in bulk."

▶ "Watch the question. Theo walks to the card that says 'bridge' and reads it. Manu and Max read their notebooks. Theo is exact but his wall keeps growing. The other two have a fixed size but must compress. That is the whole talk: what does each memory cost, and what does it lose?"

**Bridge:** "Let's start with Theo and remind ourselves how his wall actually works."

---

## 2 · Attention in one picture
**Idea:** Query asks, key advertises, value delivers. Heads let you ask several questions at once.
**On screen:** a token passes through three lenses; a row of index cards; bars that become percentages; three small attention maps.

**Say:**
"Every token is turned into three vectors by three learned lenses. The **query** says what I am looking for. The **key** says how I can be found, like the label on a card's spine. The **value** is what I hand over if you pick me."

▶ "Now a query goes along the wall. For each card we take the dot product of the query with its key. That is just a score for how well they match."

▶ "Softmax turns the scores into shares that add up to a hundred percent. Here the card 'the key' gets 46%, 'the bridge' 25%."

▶ "The output is a blend of the values, weighted by those shares. Nothing is looked up exactly; it is a soft weighted average."

▶ "Why three separate roles? Because what I look for, how I am found and what I give are three different jobs. And why many heads? Each head has its own lenses, so one can follow who-did-what, another the nearby words, another something far back. We concatenate the heads and mix them with an output matrix. That is the whole attention layer."

**Bridge:** "So far so good, but how do we run this when we generate text one token at a time?"

---

## 3 · Prefill → KV cache → decode
**Idea:** Cache the past keys and values; each new token then costs O(T), not O(T²).
**On screen:** a triangle of scores, a row of cards (the cache), a memory bar, two pills.

**Say:**
"First the prompt goes in all at once. This is **prefill**. Every token attends to all earlier tokens, so there are about T-squared scores. It is heavy, but it is parallel, so the GPU is happy."

▶ "Here is the key observation. With a causal mask, the keys and values of past tokens never change when new tokens arrive. So we keep them. That store of keys and values is the **KV cache**. Now each new token only computes its own query, key and value, appends the key and value to the cache, and sweeps one row across the cache. That is the **decode** step."

▶ "Without the cache we would redo the whole T-squared triangle for every new token. With it, each token costs work proportional to T, and the memory grows with T. Notice that it still grows: the cost per token keeps rising as the context gets longer. That is the problem the next slides try to solve."

**Bridge:** "Before we fix the cost, let me show what actually flows through these layers."

---

## 4 · Tokens evolve in the residual stream
**Idea:** A token starts as a lookup-table vector and is rewritten by every layer; context makes the same word end up in different places.
**On screen:** two sentences, one with a river bank, one with a money bank. Both start from the same arrow. Layers add arrows; the ends point in different directions.

**Say:**
"A token begins as an embedding, just a row of a lookup table. The word 'bank' starts as the *same* vector in both sentences."

▶ "Then every layer reads the current vector and adds a small update to it. Blue arrows are what attention writes, gold arrows what the feed-forward block writes. After a few layers, the river bank has drifted towards 'river' and the money bank towards 'money'. The vector that carries all of this is the **residual stream**: the running sum of everything the layers wrote. The formula is on the slide: the next stream is the current stream plus what the block computes from it."

**Bridge:** "That is Theo's world. Now let's see why we would want something else."

---

## 5 · From the growing wall to Mamba-2
**Idea:** Long context makes the wall too big. RNNs fix that but have problems; Mamba fixes those, has a new one; Mamba-2 fixes that.
**On screen:** Theo's wall and a cost curve; then three columns, RNN → Mamba → Mamba-2.

**Say:**
"Theo's wall keeps growing, and attention cost grows with the square of the context. What if the memory had a *fixed* size, no matter how long the text?"

▶ "That is the old idea of recurrent networks: read one token at a time and carry a state."

▶ "**RNNs** have two problems. Training is sequential: token t waits for token t minus one, like a line of dominoes, so you cannot parallelise over time. And the learning signal fades over long distances, so they forget early tokens."

▶ "**Mamba** fixes both. First, the model chooses what to write and what to forget at each token. The step size Δ depends on the input, like a pen whose pressure changes with the word. Second, a parallel scan lets us train all tokens at once. Its remaining problem: that scan is not a matrix multiplication, and GPUs have special hardware that does matrix multiplication much faster. So the state had to stay small."

▶ "**Mamba-2** fixes that. It uses a single decay number per head, which makes the whole computation into matrix multiplications. The state can be bigger, and the paper measures the core layer as two to eight times faster."

**Bridge:** "Let's open Max's notebook and look at the maths. It is shorter than you think."

---

## 6 · Mamba-2 in one picture
**Idea:** The memory is one N×P sheet. Each token fades it, writes a note, reads it.
**On screen:** a grid (the notebook), the equation, three labels: fade / write / read.

**Say:**
"Mamba-2's memory is a sheet of N by P numbers, and it never grows. Each token does three things. The equation is on screen."

▶ "**Fade.** The whole sheet is multiplied by a number a-t between zero and one. That is how old information is forgotten."

▶ "**Write.** The token writes a new note: its key as a column times its value as a row. That is a rank-one update, added on top of the faded sheet."

▶ "**Read.** The token's query reads the sheet and produces the output. That is the entire layer. Since the sheet has a fixed size, decoding costs the same at token ten and at token one million."

**Bridge:** "Now let's unroll this recurrence and see what it looks like as an attention matrix."

---

## 7 · Unroll it: attention with a decay mask
**Idea:** Unrolled, Mamba-2 is Y = (L ⊙ QKᵀ)V: attention where the mask L is a decay, not a softmax.
**On screen:** four tokens with decay weights, then the matrix L next to QKᵀ, then the formula.

**Say:**
"Unroll the sheet. The note written by token *s* has been faded by every decay since then. For the last token the old notes count 0.3, 0.375, 0.75 and 1. The 0.3 is simply 0.8 times 0.5 times 0.75."

▶ "Do that for every pair of tokens and you get a matrix L, with ones on the diagonal and decays below. Multiply it element by element with the usual query-key score matrix. That is it."

▶ "So Mamba-2 computes Y equals L element-wise times Q K-transpose, times V. Compare with attention: softmax of Q K-transpose, times V. Same shape, different mask. Attention normalises with softmax and keeps every token; Mamba-2 uses a decay and keeps a fixed-size summary. It is attention with a fading memory, computed as a recurrence."

**Bridge:** "That is the maths. The next slide shows why it is fast to train."

---

## 8 · Fast to train: chunks and low rank
**Idea:** Chunk the sequence; far-away blocks have low rank, so they collapse into a small state that is passed along.
**On screen:** a triangular matrix cut into blocks, one block factorised into two skinny rectangles, four numbered steps, the cost.

**Say:**
"We cut the sequence into chunks. Inside a chunk, the part of the matrix is small and dense, so we just do a matrix multiplication."

▶ "The blocks far from the diagonal look big, but they have low rank, at most N. So each one is a product of two skinny matrices, which means everything the later chunk needs from the earlier one fits in the small state."

▶ "That gives four steps: do each chunk, summarise each chunk into a state, pass the states along, and add the past back into each chunk's output. Training cost is linear in the sequence length. Decoding costs O(N·P) per token, which does not depend on the context length."

**Bridge:** "We've covered how tokens mix. Now what do the other layers do?"

---

## 9 · Why a feed-forward network?
**Idea:** Attention moves information between tokens; the FFN thinks about each token alone, and holds most of the parameters.
**On screen:** tokens with arcs between them; the same tokens, each with its own hourglass; a bar split 1/3 attention, 2/3 FFN.

**Say:**
"Attention lets tokens talk to each other. But after the talk, each token still has to *do something* with what it heard."

▶ "That is the feed-forward network. It works on each token alone: it expands the vector to four times the width, applies a non-linearity, and squeezes it back."

▶ "Count the parameters. Attention has about four d-squared, the FFN about eight d-squared. So the FFN is roughly two thirds of each layer, and that is why it is the expensive part. One useful way to think about it: attention is a memory of this *context*; the FFN is a memory of the *training data*."

**Bridge:** "If the FFN is the expensive part, can we make it much bigger without paying for it?"

---

## 10 · MoE: a hospital with specialist rooms
**Idea:** Build many experts, but send each token to only a few.
**On screen:** a hospital with eight rooms, a router at the door, tokens routed to two rooms each.

**Say:**
"Think of a hospital with eight specialist rooms, and a triage nurse at the door. That nurse is the **router**."

▶ "A patient, which is a token, arrives. The nurse picks the two best rooms. They light up and the rest stay dark. The two answers are blended using the nurse's weights. The next patient goes to different rooms."

▶ "Here is the point. All eight rooms exist, so you store all their parameters. But only two work for each token, so you only pay for two. More knowledge, same compute per token. The rooms are not neat human specialities; whatever specialisation they have emerges during training."

**Bridge:** "That is the idea. Here is how today's big models make it work."

---

## 11 · How modern LLMs run MoE
**Idea:** Many small experts plus a shared one, balanced load, experts spread over GPUs.
**On screen:** three panels: a grid of small experts with a shared one; bars equalising; four GPUs exchanging tokens.

**Say:**
"First, the recipe. Instead of a few big experts, use many small ones, plus one shared expert that is always on. DeepSeek-V3 has 256 routed experts and one shared expert, and each token uses eight. Mixtral was simpler: eight experts, two per token."

▶ "Second, keep the rooms equally busy. If the router sends everyone to the same room, the others sit idle. DeepSeek-V3 nudges a bias in the router instead of adding an extra loss term."

▶ "Third, the experts live on different GPUs, so tokens have to travel to wherever their experts are, and then travel back. That all-to-all communication is the real cost of MoE in practice."

**Bridge:** "Now let's see what connects all these blocks."

---

## 12 · The residual stream: one shared bus
**Idea:** All blocks talk through one running sum; Mamba's state runs along time, the residual runs along depth.
**On screen:** a belt with four blocks reading from it and writing back; a stack of coloured writes; a small grid with horizontal and vertical arrows.

**Say:**
"All the blocks, attention, MoE and Mamba-2, never talk directly. Each reads the residual stream, computes something, and adds its result back. The stream is the sum of everything that was ever written."

▶ "There are two directions here. Within a layer, Mamba's state carries information along *time*, from token to token. The residual stream carries information along *depth*, from layer to layer. Real hybrid models interleave a few attention layers among many Mamba-2 layers, and it all works because they share this one stream."

**Bridge:** "And Transformers themselves are cleverer than the plain version we started with."

---

## 13 · Real Transformers are cleverer than that
**Idea:** A toolbox of tricks shrinks the cache, the attention cost or the memory traffic; each came from a particular paper and ships in particular models.
**On screen:** ten tiles, each with an icon, a name, who introduced it and where it is used.

**Say:**
"Production Transformers are not the plain version. **MQA** (Shazeer, 2019, used in PaLM) and **GQA** (Ainslie et al., 2023, in Llama 2 70B and Llama 3) share keys and values between heads, which shrinks the KV cache. **Sliding-window attention** (Longformer 2020, Mistral 7B) lets each token see only recent tokens, and Gemma 2 and 3 mix local layers with global ones. **Attention sinks** (StreamingLLM) keep the first tokens so long streams stay stable. **MLA** (DeepSeek-V2) compresses keys and values into a small latent."

▶ "On the systems side: **FlashAttention** (Dao et al.) computes exactly the same attention but moves much less data through memory. **PagedAttention** (vLLM) stores the cache in pages like an operating system. **Sparse attention** (BigBird, and DeepSeek's recent work) skips most pairs. And **RoPE** with **YaRN** handles position and stretches the context window. Many of these already cut the cost we worried about on slide 3."

**Bridge:** "So where does that leave Mamba, and what is the trade-off?"

---

## 14 · The trade-off, and what 'universal' means
**Idea:** Keeping more memory means better recall but more cost; hybrids sit in a sweet spot; MoE is a separate axis. Universality can mean two things.
**On screen:** a recall-versus-memory plot with Mamba, Mamba-2, hybrid and Transformer, and a gold thermometer for MoE; then two cards, one about computing anything and one about shared circuits.

**Say:**
"This is the efficiency-versus-expressivity plot. Horizontally, how much memory the layer keeps. Vertically, how well it can recall exact details. Mamba keeps little and recalls less. Transformers keep everything and recall everything, at a cost. Mamba-2 sits in between, and a hybrid, mostly Mamba-2 with a few attention layers, gets most of the recall for a fraction of the memory. Studies back this up: the Zoology work shows the recall gap, 'Repeat After Me' shows that copying needs memory that grows with the input, and NVIDIA's empirical study found hybrids beat both pure designs. MoE is on its own axis: more capacity without more compute."

▶ "'Universality' can mean two things. First: can the architecture in principle compute any sequence function? For Transformers this is the Yun et al. result from 2020, and similar results exist for state-space models. Second, and more interesting for us: do different architectures learn the *same* things? Recent work by Wang et al. at ICLR 2025 found that Transformers and Mamba learn many of the same features and circuits, like the same two sets of points rotated onto each other."

**Bridge:** "Back to the maths: let me show where the Mamba-2 attention formula comes from."

---

## 15 · Deriving Mamba-2's attention form
**Idea:** Two roads meet at the same formula: Y = (L ⊙ QKᵀ)V.
**On screen:** two columns of equations that converge, then two lists of numbers that match.

**Say:**
"Two routes, one destination. On the left, start from the notebook. The recurrence says fade then write. Unroll it and the state is a sum of old notes, each scaled by the product of the decays since. Read it with the query and you get the output, with those decay products as the weights, which we call L."

▶ "On the right, start from attention. Softmax attention is softmax of QKᵀ times V. Drop the softmax and you get linear attention. Add a decay mask and you get the same expression as on the left."

▶ "Both roads arrive at Y equals L element-wise times QKᵀ, times V. This is not just a claim: our test code runs the recurrence and the matrix form on the same random numbers, and the outputs match digit for digit."

**Bridge:** "Last question: how do we check what these models do inside?"

---

## 16 · Looking inside, and the closing
**Idea:** We have tools for testing whether architectures compute similar things; the three brothers end up in the same stream.
**On screen:** five tiles (logit lens, patching, induction heads, sparse features, Mamba's hidden attention), then three coloured lines merging into one arrow.

**Say:**
"A few of the tools. The **logit lens** reads the residual stream early and asks what the model would predict now. **Activation patching** swaps a part between a clean and a corrupted run and sees what breaks. **Induction heads** copy what came after an earlier occurrence of the same token. **Sparse autoencoders** split the stream into clean features. And for Mamba, we can write its computation as a hidden attention matrix, using the formula from slide 15, and read it like an attention map. These are the tools with which we can test whether the brothers really compute the same thing."

▶ "To close: Theo, Manu and Max remember in very different ways. They all write into the same stream. Different memories; often similar computations."

---

## Cheat sheet (for the slide behind your back)

| Idea | Formula |
|---|---|
| Attention | O = softmax(QKᵀ / √d) V |
| KV cache | per-token decode work ∝ T, cache memory ∝ T |
| Residual stream | r⁽ℓ⁺¹⁾ = r⁽ℓ⁾ + block(norm r⁽ℓ⁾) |
| Mamba-2 state | S_t = a_t S_{t−1} + k_t v_tᵀ, y_t = S_tᵀ q_t |
| Decay mask | L_ts = a_{s+1} ⋯ a_t |
| Matrix form | Y = (L ⊙ QKᵀ) V |
| Mamba-2 cost | training linear in T, decode O(N·P) per token |
| FFN | ≈ 8d² parameters vs ≈ 4d² for attention |
| MoE | y = Σ over the top-k experts of g_i · E_i(x) |

## Questions you may get

- **Is Mamba-2 just attention?** The algebra is the same shape, but the mask is a decay instead of a softmax and the memory is a fixed-size state.
- **Does the cache make decoding free?** No. Each token still costs work proportional to T and the cache grows with T. It saves the quadratic recompute.
- **Is FlashAttention subquadratic?** No, it computes exactly the same attention with much less memory traffic.
- **Do experts specialise by topic?** Often partly, but not reliably. Treat the rooms in the hospital as a metaphor, not a map.
- **Why can't Mamba copy long texts perfectly?** The state has a fixed size, so it cannot store an arbitrarily long input verbatim. That is why hybrids add a few attention layers.
- **Are the numbers on the model-card slides exact?** They come from the model papers (DeepSeek-V3: 256 routed + 1 shared expert, top-8; Mixtral: 8 experts, top-2). Check the current model card before quoting them in a paper.

## Sources

Attention: Vaswani et al. 2017. MQA: Shazeer 2019. GQA: Ainslie et al. 2023. Longformer: Beltagy et al. 2020. StreamingLLM: Xiao et al. 2023. FlashAttention: Dao et al. 2022. PagedAttention (vLLM): Kwon et al. 2023. RoPE: Su et al. 2021. YaRN: Peng et al. 2023. DeepSeek-V2 (MLA) 2024 and DeepSeek-V3 2024.
RNN/SSM line: S4 (Gu et al. 2022), Mamba (Gu & Dao 2023), Mamba-2 / SSD "Transformers are SSMs" (Dao & Gu 2024), linear attention (Katharopoulos et al. 2020).
FFN as memory: Geva et al. 2021. MoE: Shazeer et al. 2017, GShard 2020, Switch Transformer 2021, Mixtral 2024, DeepSeekMoE 2024.
Trade-off and theory: Zoology (Arora et al. 2024), Repeat After Me (Jelassi et al. 2024), An Empirical Study of Mamba-based Language Models (Waleffe et al. 2024), Yun et al. 2020, Wang et al. ICLR 2025.
Interpretability: logit lens (nostalgebraist 2020), induction heads (Olsson et al. 2022), sparse autoencoders (Cunningham et al. 2023, Bricken et al. 2023), Hidden Attention of Mamba (Ali et al. 2024).
