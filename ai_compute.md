
# What Does an LLM Actually Compute?

Everyone uses ChatGPT or some other LLM for almost everything. I can hardly imagine working without it—I'm using it as I write this. When GPT-2 came out, I mostly thought of AI as a magical box that helps with homework. People would say *“it just predicts the next token”*, but what does that really mean? Even after studying neural networks and transformers, the mechanics felt abstract.
This is my attempt to explain what *actually happens*—both intuitively and mathematically.

We’ll focus on a **decoder-only GPT-2–style transformer**, since that’s the core of most modern LLMs. Encoder–decoder models (like T5 or BERT+decoder hybrids) work similarly, but the decoder-only model is the workhorse behind GPT models.

---

## Example Prompt

> “The best football player of all time is …”

Whenever I say **Input** below, think of that sentence—a sequence of words (tokens). We’ll walk through what the model does **before** and **during** computation.

---

## 1. Pre-computation

### Tokenization

Each LLM has a **tokenizer**—a mapping from text to integers.
Example:
$$
\text{"The"} \rightarrow 4
$$
Thus a sentence becomes a sequence of token IDs:
$$
x = [4, 5, 8, \dots ]
$$
Assume the input fits within the context window (no truncation), and we ignore attention masking details for simplicity.

---

## 2. Computation

We’ll assume a GPT-2–like architecture:

* 12 transformer blocks
* 12 attention heads per block
* Embedding size (E = 768)
* Context length (C = L) (number of tokens)

---

### High-Level Pipeline

$$
\begin{aligned}
1.&\quad X \leftarrow \text{Tokenize}(X) \\
2.&\quad \text{for } i = 1 \rightarrow 12: \\
&\quad 3.\ X_{\text{in}} \leftarrow X \\
&\quad 4.\ X \leftarrow \text{LayerNorm}(X) \\
&\quad 5.\ X \leftarrow \text{AttentionBlock}(X) \\
&\quad 6.\ X \leftarrow X_{\text{in}} + X \quad \text{(residual)}\\
&\quad 7.\ X_{\text{in}} \leftarrow X \\
&\quad 8.\ X \leftarrow \text{LayerNorm}(X) \\
&\quad 9.\ X \leftarrow \text{Projection}(X) \\
&\quad 10.\ X \leftarrow \text{GeLU}(X) \\
&\quad 11.\ X \leftarrow \text{Projection}(X) \\
&\quad 12.\ X \leftarrow X_{\text{in}} + X \quad \text{(residual)}\\
13.&\quad X \leftarrow \text{LayerNorm}(X) \\
14.&\quad X \leftarrow \text{FinalProjection}(X)
\end{aligned}
$$

---

## Step-by-Step Walkthrough

### **1. Tokenization and Embedding**

After tokenization, each integer is mapped to a learnable embedding vector and summed with a positional encoding:
$$
X \in \mathbb{R}^{C \times E}
$$

---

### **2. Transformer Block (repeated 12×)**

Each block performs:

1. LayerNorm → Self-Attention → Residual
2. LayerNorm → MLP → Residual

---

### **3. Layer Normalization**

For each token embedding (x_i \in \mathbb{R}^E):
$$
\mu_i = \frac{1}{E} \sum_{j=1}^E x_{i,j}, \quad
\sigma_i^2 = \frac{1}{E} \sum_{j=1}^E (x_{i,j} - \mu_i)^2
$$
Then normalize and scale:
$$
y_{i,j} = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j
$$
where $(\gamma, \beta \in \mathbb{R}^E)$ are learnable parameters.
Output: $(Y \in \mathbb{R}^{C \times E})$.

---

### **4. Multi-Head Self-Attention**

**Notation**
$$
X_{\text{in}} \in \mathbb{R}^{C \times E}, \quad
W^{(Q)}, W^{(K)}, W^{(V)} \in \mathbb{R}^{E \times E}, \quad
H = \text{\# heads}, \quad d_H = E / H
$$

**(a) Linear projections:**
$$
Q = XW^{(Q)}, \quad K = XW^{(K)}, \quad V = XW^{(V)}
$$

**(b) Reshape into heads:**
$$
\begin{align}
f_R: \mathbb{R}^{C\times E} &\rightarrow \mathbb{R}^{H\times C\times d_H} \\
x_{i,j} \in \R^{C \times E} &\rightarrow x_{\lfloor \frac{j}{d_H} \rfloor, i, j \bmod d_H} \in \R^{C \times C \times d_H}
\end{align}
$$

$$
Q, K, V = f_R(Q), f_R(K), f_R(V)
$$

**(c) Scaled dot-product attention:**
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

---

### **5. Elementwise Expression**

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

This is the full unrolled computation of attention + projection per output dimension.

---

### **6. Feedforward MLP**

After attention and residual addition:

**(a) LayerNorm:**
$$
X \leftarrow \text{LayerNorm}(X)
$$

**(b) Expand to hidden dimension:**
$$
X \leftarrow X W_1 + b_1, \quad W_1 \in \mathbb{R}^{E \times 4E}
$$
$$
x_{i,j} = \sum_{k=1}^E x_{i,k} w^{(1)}_{k,j} + b^{(1)}_j
$$

**(c) Apply GeLU nonlinearity:**
$$
x_{i,j} = \text{GeLU}(x_{i,j})
$$

**(d) Project back to embedding size:**
$$
X \leftarrow X W_2 + b_2, \quad W_2 \in \mathbb{R}^{4E \times E}
$$
$$
x_{i,j} = \sum_{k=1}^{4E} x_{i,k} w^{(2)}_{k,j} + b^{(2)}_j
$$

**(e) Residual connection:**
$$
X_{\text{out}} = X_{\text{in}} + X
$$

---

### **7. Output Projection (Logits)**

After all blocks:
$$
X \in \mathbb{R}^{C \times E}
$$

Apply final LayerNorm and a linear projection to vocabulary dimension:
$$
\text{Logits} = X W^{(\text{vocab})} + b, \quad W^{(\text{vocab})} \in \mathbb{R}^{E \times |\mathcal{V}|}
$$

Each element:
$$
x_{i,j} = \sum_{k=1}^E x_{i,k} w_{k,j} + b_j
$$

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
and learns parameters (W), (b), (\gamma), (\beta) that maximize the likelihood of the training text.
