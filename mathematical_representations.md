# Mathematical Representations of Models

## 1. Gemini Model

The Gemini model is a multimodal transformer-based architecture that processes both text and images. Here's its mathematical representation:

### Text Processing
For text input sequence \(X = (x_1, x_2, ..., x_n)\):

1. **Token Embedding**:
\[E_{text} = \text{Embedding}(X) \in \mathbb{R}^{n \times d_{model}}\]

2. **Positional Encoding**:
\[PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})\]
\[PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})\]

3. **Multi-Head Attention**:
\[Q = XW^Q, K = XW^K, V = XW^V\]
\[\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V\]

4. **Feed-Forward Network**:
\[FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2\]

### Image Processing
For image input \(I \in \mathbb{R}^{H \times W \times C}\):

1. **Image Embedding**:
\[E_{image} = \text{VisionTransformer}(I) \in \mathbb{R}^{m \times d_{model}}\]

2. **Cross-Modal Attention**:
\[A_{text-image} = \text{Attention}(E_{text}, E_{image}, E_{image})\]
\[A_{image-text} = \text{Attention}(E_{image}, E_{text}, E_{text})\]

### Final Output
\[P(y|X,I) = \text{softmax}(W_o[A_{text-image}; A_{image-text}] + b_o)\]

## 2. BERT Model

BERT (Bidirectional Encoder Representations from Transformers) uses a bidirectional transformer architecture:

### Input Processing
For input sequence \(X = (x_1, x_2, ..., x_n)\):

1. **Token Embedding**:
\[E_{token} = \text{Embedding}(X) \in \mathbb{R}^{n \times d_{model}}\]

2. **Position Embedding**:
\[E_{pos} = \text{PositionEmbedding}(X) \in \mathbb{R}^{n \times d_{model}}\]

3. **Segment Embedding**:
\[E_{seg} = \text{SegmentEmbedding}(X) \in \mathbb{R}^{n \times d_{model}}\]

4. **Combined Embedding**:
\[E = E_{token} + E_{pos} + E_{seg}\]

### Transformer Layers
For each layer \(l\):

1. **Multi-Head Self-Attention**:
\[Q_l = E_lW^Q_l, K_l = E_lW^K_l, V_l = E_lW^V_l\]
\[A_l = \text{softmax}(\frac{Q_lK_l^T}{\sqrt{d_k}})V_l\]

2. **Layer Normalization**:
\[LN(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\]

3. **Feed-Forward Network**:
\[FFN_l(x) = \max(0, xW_{1,l} + b_{1,l})W_{2,l} + b_{2,l}\]

### Final Representation
\[H = \text{LayerNorm}(E + \sum_{l=1}^L A_l)\]

## 3. Detoxify Model

Detoxify is based on a transformer architecture with specific modifications for toxicity detection:

### Input Processing
For input sequence \(X = (x_1, x_2, ..., x_n)\):

1. **Token Embedding**:
\[E = \text{Embedding}(X) \in \mathbb{R}^{n \times d_{model}}\]

### Transformer Encoder
For each layer \(l\):

1. **Multi-Head Attention**:
\[Q_l = E_lW^Q_l, K_l = E_lW^K_l, V_l = E_lW^V_l\]
\[A_l = \text{softmax}(\frac{Q_lK_l^T}{\sqrt{d_k}})V_l\]

2. **Residual Connection and Layer Normalization**:
\[E_{l+1} = \text{LayerNorm}(E_l + A_l)\]

3. **Feed-Forward Network**:
\[FFN_l(x) = \max(0, xW_{1,l} + b_{1,l})W_{2,l} + b_{2,l}\]

### Toxicity Classification
\[P(toxic|X) = \sigma(W_o \cdot \text{pool}(H) + b_o)\]

Where:
- \(\sigma\) is the sigmoid function
- \(\text{pool}\) is the pooling operation (usually mean or max pooling)
- \(W_o\) and \(b_o\) are the output layer parameters

### Loss Function
The model uses Binary Cross-Entropy loss:
\[L = -\frac{1}{N}\sum_{i=1}^N [y_i \log(p_i) + (1-y_i)\log(1-p_i)]\]

Where:
- \(N\) is the batch size
- \(y_i\) is the true label
- \(p_i\) is the predicted probability

## Model Parameters

### Gemini Model
- \(d_{model}\): Model dimension (typically 2048)
- \(d_k\): Key dimension (typically 256)
- \(n_{heads}\): Number of attention heads (typically 16)
- \(n_{layers}\): Number of transformer layers (typically 32)

### BERT Model
- \(d_{model}\): Model dimension (typically 768 for BERT-base)
- \(d_k\): Key dimension (typically 64)
- \(n_{heads}\): Number of attention heads (typically 12)
- \(n_{layers}\): Number of transformer layers (typically 12)

### Detoxify Model
- \(d_{model}\): Model dimension (typically 768)
- \(d_k\): Key dimension (typically 64)
- \(n_{heads}\): Number of attention heads (typically 12)
- \(n_{layers}\): Number of transformer layers (typically 6) 