
# What Does an LLM Actually Compute?

Everyone uses ChatGPT or some other LLM for almost everything. I can hardly imagine working without it—I'm using it as I write this. When GPT-2 came out, I mostly thought of AI as a magical box that helps with homework. People would say *“it just predicts the next token”*, but what does that really mean? Even after studying neural networks and transformers, the mechanics felt abstract.
This is my attempt to explain what *actually happens* a bit more mathematically instead of just matrix multiplications and also give small intuitive explanations. Below I'll try to derive the exact formula for each output number of a transformer architecture, as we essentially have one very long formula (which is supposed to describe "language").
I’ll focus on a decoder-only GPT-2–style transformer, since that’s the core of most modern LLMs. Encoder–decoder models (like T5 or BERT+decoder hybrids) work similarly, but the decoder-only model is the workhorse behind GPT models. 

---
## Example Prompt

> “The best football player of all time is …”

Whenever I say **Input** below, think of that sentence—a sequence of words (tokens). We’ll walk through what the model does **before** and **during** computation.

We’ll assume a GPT-2–like architecture:

* 12 transformer blocks
* 12 attention heads per block
* Embedding size (E = 768)
* Context length (C = L) (number of tokens)

## 1. Computation

### High-Level Pipeline


$$
\begin{aligned}
&\quad X_1 \leftarrow \text{Tokenize}(X_{input}) \cr
&\quad \text{for } i = 1 \rightarrow 12: \cr
&\quad \quad X_{i+1} \leftarrow Block_i(X_{i}) \cr
&\quad X_{14} \leftarrow \text{LayerNorm}(X_{13}) \cr
&\quad X_{out} \leftarrow \text{FinalProjection}(X_{14})
\end{aligned}
$$

As we can see the main computation happens in each Block: 
Each Block has the same architecture and consists of an Attention Layer and an MLP. If we unroll what happens at each block we obtain following pipeline:

$$
\begin{aligned}
1.&\quad X_{1} \leftarrow \text{Tokenize}(X) \cr
2.&\quad \text{for } i = 1 \rightarrow 12: \cr
&\quad 3.\ X_{i_1} \leftarrow \text{LayerNorm}(X_i) \cr
&\quad 4.\ X_{i_2} \leftarrow \text{Self-AttentionBlock}(X_{i_1}) \cr
&\quad 5.\ X_{i_3} \leftarrow X_{i} + X_{i_2} \quad \text{(residual)}\cr
&\quad 6.\ X_{i_4} \leftarrow \text{LayerNorm}(X_{i_3}) \cr
&\quad 7.\ X_{i_5} \leftarrow \text{Projection}(X_{i_4}) \cr
&\quad 8.\ X_{i_6} \leftarrow \text{GeLU}(X_{i_5}) \cr
&\quad 9.\ X_{i_7} \leftarrow \text{Projection}(X_{i_6}) \cr
&\quad 10.\ X_{i+1} \leftarrow X_{i_3} + X_{i_7} \quad \text{(residual)}\cr
11.&\quad X_{14} \leftarrow \text{LayerNorm}(X_{13}) \cr
12.&\quad X_{out} \leftarrow \text{FinalProjection}(X_{14})
\end{aligned}
$$

---

## Step-by-Step Walkthrough
### **Notations**

**Notation**
$$
X_{\text{i}} \in \mathbb{R}^{C \times E}, \quad
W^{(Q), i}, W^{(K),i}, W^{(V),i} \in \mathbb{R}^{E \times E}, \quad
H = \text{\# heads}, \quad d_H = E / H
$$
$W^{i, j} ;=$ denotes the $j-th$ learnable parameter matrix at block $i$ and $W^{(Q), i}, W^{(K),i}, W^{(V),i}$ denote the Key, Query, Value Matrices in block $i$. $X_{i} ;=$ is the **input** to Block $i$ and we denote $x_{j,k}^{(i)} \in X_{i}$ to show which element of the input $X_{i}$, when its clear to which matrix and element belongs to I will omit writing it and just write: $x_{i,j}$.

### **1. Tokenization and Embedding**

Each LLM has a **tokenizer**—a mapping from text to integers.
The first step is to turn words into numbers the model can work with. Tokenization breaks the text into subword units, each mapped to an integer ID.
These IDs are then turned into continuous vectors (embeddings), which let the model represent relationships between tokens as geometric patterns in a high-dimensional space.
Example:
$$
\text{"The"} \rightarrow 4
$$
Thus a sentence becomes a sequence of token IDs:
$$
x = [4, 5, 8, \dots ]
$$
Assume the input fits within the context window (no truncation), and we ignore attention masking details for simplicity.

After tokenization, each integer is mapped to a learnable embedding vector and summed with a positional encoding:
$$
X_1 \in \mathbb{R}^{C \times E}
$$

---

### **2. Transformer Block (repeated 12×)**

Each transformer block refines the token representations through two main components:
(1) Self-attention, which lets each token gather information from relevant previous tokens, and
(2) Feedforward MLP, which applies nonlinear transformations to enrich the representation.

---
#### **2.1. Attention Layer**

This is the first part of each block, which consists of steps 3,4,5 in above pipeline:

**2.1.1 LayerNorm** Before computing attention, we normalize each token’s vector so that the following computations are numerically stable and independent of scaling differences between tokens. We take each element $x_{j,k}^{i} \in X_i$ and normalize accordingly:
$$
\mu_i = \frac{1}{E} \sum_{j=1}^E x_{i,j}, \quad
\sigma_i^2 = \frac{1}{E} \sum_{j=1}^E (x_{i,j} - \mu_i)^2
$$
Then normalize and scale:
$$
y_{i,j} = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j
$$
where $(\gamma, \beta \in \mathbb{R}^E)$ are learnable parameters.
We can write this as a function of $ln(x_{i,j}) = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j$

---

**2.1.2 Multi-Head Self-Attention**
Now the model computes how much each token should attend to previous tokens. This is done by projecting the input into three different spaces: Queries (what I want), Keys (what I offer), and Values (what I carry)

**(a) Linear projections:**
$$
Q^{(i)} = XW^{(Q),i}+b, \quad K^{(i)} = XW^{(K),i}+b, \quad V^{(i)} = XW^{(V),i}+b
$$

**(b) Reshape into heads:**
$$
\begin{align}
f_R: \mathbb{R}^{C\times E} &\rightarrow \mathbb{R}^{H\times C\times d_H} \cr
x_{i,j} \in \mathbb{R}^{C \times E} &\rightarrow x_{\lfloor \frac{j}{d_H} \rfloor, i, j \bmod d_H} \in \mathbb{R}^{C \times C \times d_H}
\end{align}
$$

$$
Q, K, V = f_R(Q), f_R(K), f_R(V)
$$

**(c) Scaled dot-product attention:**
The attention score between token i and token j is the scaled dot product of their query and key vectors. The higher this score, the more i will attend to j.
$$
T^{(0)} = \frac{QK^\top}{\sqrt{d_H}}
$$

**(d) Apply causal (lower-triangular) mask (L \in \mathbb{R}^{C \times C}):**
$$
T^{(1)} = T^{(0)} \odot L
$$

**(e) Softmax along sequence dimension:**
$$
T^{(1)} = \text{softmax}(T^{(1)})
$$

**(f) Multiply by values:**
$$
T^{(2)} = T^{(1)} V
$$

**(g) Merge heads back:**
$$
T^{(3)} = f_R^{-1}(T^{(2)})
$$

**(h) Output projection:**
$$
T^{(4)} = T^{(3)} W^{(P)}, \quad W^{(P)} \in \mathbb{R}^{E \times E}
$$

**(i) Residual connection:**
$$
X_{\text{out}} = X_{\text{in}} + T^{(4)}
$$


**Elementwise Expression**

This is the full unrolled computation of attention + projection per output dimension. This way we express the output of any attention layer at any block.
The output is a new contextualized embedding for each token: its original representation plus a weighted mix of information from other relevant tokens. In the last line we explicitely show the indexing for the output of the attention layer in block $l$

$$
\begin{aligned}
x_{i,j}^{\text{attn}} &= x_{i,j} + t^4_{i,j} \cr
&= x_{i,j} + \sum_{k=1}^E t^3_{i,k} w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E t^2_{\lfloor k/d_H \rfloor,  i,  k \bmod d_H} w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C t^1_{\lfloor k/d_H \rfloor,  i,  k'} v_{\lfloor k/d_H \rfloor,  k',  k \bmod d_H}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\left(t^{(0)}\_{\lfloor k/d_H \rfloor,  i,  k'}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\left(t^{(0)}\_{\lfloor k/d_H \rfloor,  i,  k''}\right) \mathbf{1}\_{\{i \geq k''\}}} v_{\lfloor k/d_H \rfloor,  k',  k \bmod d_H}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{\lfloor k/d_H \rfloor,  i,  k'''} k^{\star}\_{\lfloor k/d_H \rfloor,  k',  k'''}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{\lfloor k/d_H \rfloor,  i,  k'''} k^{\star}\_{\lfloor k/d_H \rfloor,  k'',  k'''}\right) \mathbf{1}\_{\{i \geq k''\}}} v_{\lfloor k/d_H \rfloor,  k',  k \bmod d_H}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{i,  \lfloor k/d_H \rfloor d_H + k'''} k^{\star}\_{k',  \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} q_{i,  \lfloor k/d_H \rfloor d_H + k'''} k^{\star}\_{k'',  \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k''\}}} v_{k',  k}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E x_{i,k''''} w^q_{k'''',  \lfloor k/d_H \rfloor d_H + k'''} \sum_{k''''=1}^E x_{k',k''''} w^k_{k'''',  \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E x_{i,k''''} w^q_{k'''',  \lfloor k/d_H \rfloor d_H + k'''} \sum_{k''''=1}^E x_{k'',k''''} w^k_{k'''', \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k''\}}} \sum_{k''=1}^E x_{k',k''} w^v_{k'', k}\right) w_{k,j} \cr
&= x_{i,j} + \sum_{k=1}^E \left(\sum_{k'=1}^C \frac{\exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E ln(x_{i,k''''}) w^q_{k'''',  \lfloor k/d_H \rfloor d_H + k'''} \sum_{k''''=1}^E ln(x_{k',k''''}) w^k_{k'''',  \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k'\}}}{\sum_{k''=1}^C \exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E ln(x_{i,k''''}) w^q_{k'''',  \lfloor k/d_H \rfloor d_H + k'''} \sum_{k''''=1}^E ln(x_{k'',k''''}) w^k_{k'''', \lfloor k/d_H \rfloor d_H + k'''}\right) \mathbf{1}\_{\{i \geq k''\}}} \sum_{k''=1}^E ln(x_{k',k''}) w^v_{k'', k}\right) w_{k,j} \cr
x_{i,j}^{attn, l+1} &= x_{i,j}^{(l)} + \sum_{k=1}^E \left((\sum_{k'=1}^C \frac{\exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E ln(x_{i,k''''}^{(l)})) w_{k'''',  \lfloor k/d_H \rfloor d_H + k'''}^{(q),l} \sum_{k''''=1}^E ln(x_{k',k''''}^{(l)}) w_{k'''',  \lfloor k/d_H \rfloor d_H + k'''}^{(k), l} \right) \mathbf{1}\_{\{i \geq k'\}}}
{\sum_{k''=1}^C \exp\left(\frac{1}{\sqrt{d_H}} \sum_{k'''=1}^{d_H} \sum_{k''''=1}^E ln(x_{i,k''''}^{(l)}) w_{k'''',  \lfloor k/d_H \rfloor d_H + k'''}^{(q), l} \sum_{k''''=1}^E ln(x_{k'',k''''}^{(l)}) w_{k'''', \lfloor k/d_H \rfloor d_H + k'''}^{(k), l}\right) \mathbf{1}\_{\{i \geq k''\}}} 
\sum_{k''=1}^E ln(x_{k',k''}^{(l)}) w_{k'', k}^{(v), l} \right) w_{k,j}^{l,1}
\end{aligned}
$$

---

#### **2.2 Feedforward MLP**

Once attention has gathered context, the MLP processes each token independently. It expands the embedding dimension, applies a nonlinear transformation (GeLU), and compresses it back. This helps the model capture complex patterns that attention alone cannot.

**(a) LayerNorm:**
$$
T^{1} \leftarrow \text{LayerNorm}(Attn(X^{l}))
$$

**(b) Expand to hidden dimension:**
$$
T^{2} \leftarrow T^{1} W^{l,1} + b
$$
$$
t_{i,j}^{2} = \sum_{k=1}^E x_{i,k} w_{k,j}^{l,1} + b_j
$$

**(c) Apply GeLU nonlinearity:**
$$
t_{i,j}^{3} = \text{GeLU}(t_{i,j}^{2})
$$

**(d) Project back to embedding size:**
$$
T^{4} \leftarrow T^{3} W^{l,2} + b, \quad W^{l,2} \in \mathbb{R}^{4E \times E}
$$
$$
t_{i,j}^{4} = \sum_{k=1}^{4E} t_{i,k}^{3} w_{k,j}^{l,2} + b_j
$$

**(e) Residual connection:**
$$
X^{l+1} = Attn(X^{l}) + T^{4}
$$

### **3. Output of Block $i$**
When we put the above together we get the full output of each block $l$:
$$
\begin{align}
X^{l+1} &= Attn(X^{l}) + T^{4} \cr
&= Attn(X^{l}) + T^{3} W^{l,2} + b \cr
&= Attn(X^{l}) + GeLu(T^{2}) W^{l,2} + b \cr
&= Attn(X^{l}) + GeLu(LN(X^{l}) W^{l,1} + b) W^{l,2} + b \cr
\end{align}
$$

**Elementwise Computation**
Let's continue to work with $x_{i,j}^{attn, l}$ as input to this part of block $l$. 

$$\begin{align}
x_{i,j}^{l+1} &= x_{i,j}^{attn, l} + \sum_{k=1}^{E} t_{i,k}^{3} w_{k,j}^{l,2} + b_j \cr
&= x_{i,j}^{attn, l} + \sum_{k=1}^{E} GeLU(t{i,k}^2) w_{k,j}^{l,2} + b_j \cr
&= x_{i,j}^{attn, l} + \sum_{k=1}^{E} GeLU(\sum_{k'=1}^{E} t_{i,k}^{1} w_{k'',k}^{l,1}+ b_k) w_{k,j}^{l,2} + b_j \cr
&= x_{i,j}^{attn, l} + \sum_{k=1}^{E} GeLU(\sum_{k'=1}^{E} ln(x_{i,k''}^{attn, l})w_{k'',k}^{l,1}+ b_k) w_{k,j}^{l,2} + b_j 
\end{align}
$$



### **4. Output Projection (Logits)**

After passing through all layers, we obtain a contextual embedding for each token — a compact representation of everything the model “knows” so far about it and its context.
The final linear layer maps this embedding into a vector the size of the vocabulary, producing one logit per possible next token.
$$
X \in \mathbb{R}^{C \times E}
$$
$$
\text{Logits} = X W^{(\text{vocab})} + b, \quad W^{(\text{vocab})} \in \mathbb{R}^{E \times |\mathcal{V}|}
$$

Each element:
$$
x_{i,j} = \sum_{k=1}^E x_{i,k} w_{k,j} + b_j
$$


### **5. Putting it all togethter**

Now if we assume 12 layers we can express one output $x_{i,j}^{out}$ following way using the notation and expressions derived above:

$$
\begin{align}
x_{i,j}^{out} &= \sum_{k=1}^{E} x_{i,k} w_{k,j} + b_j \cr
&= \sum_{k=1}^{E} ln(x_{i,k}^{13}) w_{k,j} + b_j \cr
&= \sum_{k=1}^{E} ln(x_{i,k}^{13}) w_{k,j} + b_j ,\quad x_{i,k}^{l} = x_{i,j}^{attn, l} + \sum_{k=1}^{E} GeLU(\sum_{k'=1}^{E}ln(x_{i,k''}^{attn, l})w_{k'',k}^{l,1}+ b_k) w_{k,j}^{l,2} + b_j , \: 1 \leq l \leq 13 \cr
\end{align}
$$

So in total we do above computation for all $x_{i,j}^{out}$ which are $C \times VocabSize$ elements (Output Dimension).

---

### **Interpretation**

$$
x_{i,j} = \text{logit representing } P(\text{token } j \mid \text{previous } i-1 \text{ tokens})
$$

The softmax of each row gives the probability distribution over the next token.

---

✅ **In short:**
An LLM computes, for every position (i),
$$
P(x_i \mid x_{<i}) = \text{softmax}(X_i W^{(\text{vocab})})
$$
and learns parameters $(W), (b), (\gamma), (\beta)$ that maximize the likelihood of the training text.
