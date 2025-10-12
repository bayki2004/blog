## What does an LLM actually compute?

Everyone uses ChatGPT or some LLM for almost everything. I can hardly imagine working without it—I'm using it as I write this. When GPT‑2 came out, I mostly thought of AI as a magical box that helps with homework. People would say “it just predicts the next token,” but what does that really mean? Even after courses on neural networks and transformers, the mechanics can feel opaque. This is my attempt to explain what actually happens—both intuitively and mathematically.

I'll focus on a decoder‑only GPT‑2–style transformer. Encoder–decoder models are similar in many ways, but decoder‑only is the workhorse for LLMs today.

Let's say you ask GPT:

“The best football player of all time is …”

When I refer to the Input below, think of that example sentence—a human‑readable sequence of words. We'll walk through what the model does before and during computation.

### Pre‑computation

1. **Tokenize**: Each LLM has a tokenizer—a mapping from text to integers. For example, the GPT tokenizer might map the word “The” → 4. The tokenized input is a sequence of integers, e.g., x = [4, 5, 8, …].
2. **Special tokens**: We prepend a special beginning‑of‑sequence token: ``<|bos|>``. So the model actually sees ``<|bos|>`` followed by your tokenized input.

Assumption for simplicity: the input fits within the context window (so no truncation or loss masking), and we ignore attention masking details here.

## Computation

Assume a GPT‑2–like architecture with 12 transformer blocks, 12 attention heads per block, and embedding size \(e = 768\). Let the context length be \(L\) (the number of tokens in your prompt).


Notation: $C=\text{Context Length}$, $E=\text{Embedding Size}$, $H=\text{Number of Heads}$, $d_H=\text{Head Size}$\\
$$X_{in} \in \R^{C \times E}, \quad W^{(q)}, W^{(k)}, W^{(v)} \in \R^{E \times E}$$

1. **Embeddings and positional encodings**
   - The token sequence is mapped to embeddings: $X \in \mathbb{R}^{C \times E}$. Each row is a learned embedding vector for one token.
   - Positional encodings (often sinusoidal or learned) are added: $P_{enc} \in \mathbb{R}^{C \times E}$. The input to the first block is $H = X + P_{enc}$.

2. **Multi-Head Self Attention**

* We start of by computing three projections of the input to obtain the Query, Key, Value Matrices: 
$$Q= XW^{(Q)} + b, \quad K = XW^{(k)} + b, \quad V = XW^{(v)} + b$$
$b=0$ as we assume no bias vector. We also assume no dropout which will be omitted in the future. So for the future we will not write $+b$. 

* Now we reshape our Query, Key, Value Matrices to use Heads, which can be described by following function: $$f_R: \R^{C\times E} \rightarrow \R^{H \times C \times d_H}, \quad x_{i,j} \in \R^{C\times E} \rightarrow x_{\lfloor \frac{j}{d_H} \rfloor, i, j \bmod d_H} \in \R^{H\times C \times d_H}$$

$$ Q = f(Q), \quad K = f(K), \quad V=f(V)$$

* Now comes our Key, Query approximation. This is where we encode our self information: $$T^{(0)} = \frac{QK^T}{\sqrt{d_H}}$$

* Apply Self-Attention Mask: 
$$T^{(1)} = T^{(0)} \times L, \quad L \in \R^{C \times C} \quad \text{(Lower Triangular Matrix)}$$

* Take the Softmax across the head dimension: $$T^{(1)} = softmax(T^{(1)})$$

* Multiply by Value Matrix: (Applying learned information about previous tokens)
$$T^{(2)} = T^{(1)} \times V$$

* Reshaping our current multi-head matrix into single head: Applying $f^{-1}_R$:
$$T^{(3)} = f^{-1}_R(C)$$

* Have One Linear Layer Again as a projection, $W^{(p)} \in \R^{E \times E}$ and assuming no bias again:
$$T^{(4)} = T^{(3)}W^{(p)}$$

* Finally the Residual Connection: 
$$ X_{out} = T^{(4)} + X_{in}$$



Here’s a cleaned-up, blog-style walkthrough that keeps all your math, fixes the formatting, and clarifies a couple of details (notably: softmax is over the **key**/sequence dimension, and masks are applied to the **logits** before softmax).

---

# What does an LLM actually compute?

Everyone uses ChatGPT or some LLM for almost everything. I can hardly imagine working without it—I'm using it as I write this. When GPT-2 came out, I mostly thought of AI as a magical box that helps with homework. People would say “it just predicts the next token,” but what does that really mean? Even after courses on neural networks and transformers, the mechanics can feel opaque. This is my attempt to explain what actually happens—both intuitively and mathematically.

We’ll focus on a **decoder-only (GPT-2–style)** transformer. Encoder–decoder models are similar, but decoder-only is the workhorse for LLMs today.

Suppose you ask GPT:

> “The best football player of all time is …”

When I say **Input** below, think of that sentence—a human-readable sequence of words. We’ll walk through what the model does **before** and **during** computation.

---

## Pre-computation

1. **Tokenize**
   Each LLM has a tokenizer mapping text → integers. For example, the GPT tokenizer might map “The” → 4. The tokenized input is a sequence of integers, e.g.
   `x = $[4, 5, 8, …]$`.

2. **Special tokens**
   We prepend a beginning-of-sequence token `<|bos|>`. So the model actually sees `<|bos|>` followed by your tokens.

*Assumption:* The input fits within the context window; we ignore padding and loss masking here.

---

## Computation

Assume a GPT-2–like architecture with **12 transformer blocks**, **12 attention heads** per block, and **embedding size** (E=768). Let the **context length** be (C) (number of tokens).

**Notation.**
$(C=)$ Context length, $(E=)$ Embedding size, $(H=)$ Number of heads, $(d_H=)$ Head size with $E = H\cdot d_H$.
$$
X_{\text{in}} \in \mathbb{R}^{C \times E},
\quad W^{(q)}, W^{(k)}, W^{(v)} \in \mathbb{R}^{E \times E}.$$

---

## 1) Embeddings + Positional Encodings

**Idea.** Convert discrete tokens to vectors, then inject position information.

* Token embeddings: $(X \in \mathbb{R}^{C \times E})$ (one row per token).
* Positional encodings: $(P_{\text{enc}} \in \mathbb{R}^{C \times E})$.

**Input to the first block:**

$$H^{(0)} ;=; X + P_{\text{enc}}.$$




**Elementwise computation** — how each $x^{\text{out}}_{i,j}$ is computed:

$$
\begin{aligned}
x^{\text{out}}\_{i,j} &= x_{i,j} + t^4_{i,j} \cr
&= x_{i,j} + \sum_{k=1}^E t^3_{i,k} w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E t^2_{\lfloor k/d_H \rfloor,\, i,\, k \bmod d_H} w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C t^1_{\lfloor k/d_H \rfloor,\, i,\, k'} v_{\lfloor k/d_H \rfloor,\, k',\, k \bmod d_H}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\!\left(t^{(0)}\_{\lfloor k/d_H \rfloor,\, i,\, k'}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\!\left(t^{(0)}\_{\lfloor k/d_H \rfloor,\, i,\, k''}\right) \mathbf{1}\_{\{i \geq k''\}}} v_{\lfloor k/d_H \rfloor,\, k',\, k \bmod d_H}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\!\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{\lfloor k/d_H \rfloor,\, i,\, k'''} k^{\star}\_{\lfloor k/d_H \rfloor,\, k',\, k'''}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\!\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{\lfloor k/d_H \rfloor,\, i,\, k'''} k^{\star}\_{\lfloor k/d_H \rfloor,\, k'',\, k'''}\right) \mathbf{1}\_{\{i \geq k''\}}} v_{\lfloor k/d_H \rfloor,\, k',\, k \bmod d_H}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\!\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{i,\, \lfloor k/d_H \rfloor d_H + k'''} k^{\star}\_{k',\, \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\!\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{i,\, \lfloor k/d_H \rfloor d_H + k'''} k^{\star}\_{k'',\, \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k''\}}} v_{k',\, k}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\!\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E x_{i,k''''} w^q_{k'''',\, \lfloor k/d_H \rfloor d_H + k'''} \sum_{k''''=1}^E x_{k',k''''} w^k_{k'''',\, \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\!\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E x_{i,k''''} w^q_{k'''',\, \lfloor k/d_H \rfloor d_H + k'''} \sum_{k''''=1}^E x_{k'',k''''} w^k_{k'''',\, \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k''\}}} \sum_{k''=1}^E x_{k',k''} w^v_{k'',\, k}\right) w_{k,j}
\end{aligned}
$$


In a full transformer block, this attention sublayer is wrapped with residual connections and layer normalization, followed by a position‑wise feed‑forward network (also with residuals and layer norm). Repeating this block 12 times yields the final hidden states, which are then projected to the vocabulary logits to produce next‑token probabilities.
