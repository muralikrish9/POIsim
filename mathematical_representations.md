# Mathematical Representations of Models

## 1. Gemini Model

The Gemini model is a multimodal transformer-based architecture that processes both text and images. Here's its mathematical representation:

### Text Processing
For text input sequence \(\mathbf{X} = (x_1, x_2, \ldots, x_n)\):

1. **Token Embedding**:
\[\mathbf{E}_{\text{text}} = \text{Embedding}(\mathbf{X}) \in \mathbb{R}^{n \times d_{\text{model}}}\]

2. **Positional Encoding**:
\[PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)\]
\[PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)\]

3. **Multi-Head Attention**:
\[\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V\]
\[\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}\]

4. **Feed-Forward Network**:
\[\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2\]

### Image Processing
For image input \(\mathbf{I} \in \mathbb{R}^{H \times W \times C}\):

1. **Image Embedding**:
\[\mathbf{E}_{\text{image}} = \text{VisionTransformer}(\mathbf{I}) \in \mathbb{R}^{m \times d_{\text{model}}}\]

2. **Cross-Modal Attention**:
\[\mathbf{A}_{\text{text-image}} = \text{Attention}(\mathbf{E}_{\text{text}}, \mathbf{E}_{\text{image}}, \mathbf{E}_{\text{image}})\]
\[\mathbf{A}_{\text{image-text}} = \text{Attention}(\mathbf{E}_{\text{image}}, \mathbf{E}_{\text{text}}, \mathbf{E}_{\text{text}})\]

### Final Output
\[P(y|\mathbf{X},\mathbf{I}) = \text{softmax}(\mathbf{W}_o[\mathbf{A}_{\text{text-image}}; \mathbf{A}_{\text{image-text}}] + \mathbf{b}_o)\]

## 2. BERT Model

BERT (Bidirectional Encoder Representations from Transformers) uses a bidirectional transformer architecture:

### Input Processing
For input sequence \(\mathbf{X} = (x_1, x_2, \ldots, x_n)\):

1. **Token Embedding**:
\[\mathbf{E}_{\text{token}} = \text{Embedding}(\mathbf{X}) \in \mathbb{R}^{n \times d_{\text{model}}}\]

2. **Position Embedding**:
\[\mathbf{E}_{\text{pos}} = \text{PositionEmbedding}(\mathbf{X}) \in \mathbb{R}^{n \times d_{\text{model}}}\]

3. **Segment Embedding**:
\[\mathbf{E}_{\text{seg}} = \text{SegmentEmbedding}(\mathbf{X}) \in \mathbb{R}^{n \times d_{\text{model}}}\]

4. **Combined Embedding**:
\[\mathbf{E} = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{pos}} + \mathbf{E}_{\text{seg}}\]

### Transformer Layers
For each layer \(l\):

1. **Multi-Head Self-Attention**:
\[\mathbf{Q}_l = \mathbf{E}_l\mathbf{W}^Q_l, \quad \mathbf{K}_l = \mathbf{E}_l\mathbf{W}^K_l, \quad \mathbf{V}_l = \mathbf{E}_l\mathbf{W}^V_l\]
\[\mathbf{A}_l = \text{softmax}\left(\frac{\mathbf{Q}_l\mathbf{K}_l^T}{\sqrt{d_k}}\right)\mathbf{V}_l\]

2. **Layer Normalization**:
\[\text{LN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} + \boldsymbol{\beta}\]

3. **Feed-Forward Network**:
\[\text{FFN}_l(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_{1,l} + \mathbf{b}_{1,l})\mathbf{W}_{2,l} + \mathbf{b}_{2,l}\]

### Final Representation
\[\mathbf{H} = \text{LayerNorm}\left(\mathbf{E} + \sum_{l=1}^L \mathbf{A}_l\right)\]

## 3. Detoxify Model

Detoxify is based on a transformer architecture with specific modifications for toxicity detection:

### Input Processing
For input sequence \(\mathbf{X} = (x_1, x_2, \ldots, x_n)\):

1. **Token Embedding**:
\[\mathbf{E} = \text{Embedding}(\mathbf{X}) \in \mathbb{R}^{n \times d_{\text{model}}}\]

### Transformer Encoder
For each layer \(l\):

1. **Multi-Head Attention**:
\[\mathbf{Q}_l = \mathbf{E}_l\mathbf{W}^Q_l, \quad \mathbf{K}_l = \mathbf{E}_l\mathbf{W}^K_l, \quad \mathbf{V}_l = \mathbf{E}_l\mathbf{W}^V_l\]
\[\mathbf{A}_l = \text{softmax}\left(\frac{\mathbf{Q}_l\mathbf{K}_l^T}{\sqrt{d_k}}\right)\mathbf{V}_l\]

2. **Residual Connection and Layer Normalization**:
\[\mathbf{E}_{l+1} = \text{LayerNorm}(\mathbf{E}_l + \mathbf{A}_l)\]

3. **Feed-Forward Network**:
\[\text{FFN}_l(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_{1,l} + \mathbf{b}_{1,l})\mathbf{W}_{2,l} + \mathbf{b}_{2,l}\]

### Toxicity Classification
\[P(\text{toxic}|\mathbf{X}) = \sigma(\mathbf{W}_o \cdot \text{pool}(\mathbf{H}) + \mathbf{b}_o)\]

Where:
- \(\sigma\) is the sigmoid function: \(\sigma(x) = \frac{1}{1 + e^{-x}}\)
- \(\text{pool}\) is the pooling operation (usually mean or max pooling)
- \(\mathbf{W}_o\) and \(\mathbf{b}_o\) are the output layer parameters

### Loss Function
The model uses Binary Cross-Entropy loss:
\[\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N [y_i \log(p_i) + (1-y_i)\log(1-p_i)]\]

Where:
- \(N\) is the batch size
- \(y_i\) is the true label
- \(p_i\) is the predicted probability

## Model Parameters

### Gemini Model
- \(d_{\text{model}}\): Model dimension (typically 2048)
- \(d_k\): Key dimension (typically 256)
- \(n_{\text{heads}}\): Number of attention heads (typically 16)
- \(n_{\text{layers}}\): Number of transformer layers (typically 32)

### BERT Model
- \(d_{\text{model}}\): Model dimension (typically 768 for BERT-base)
- \(d_k\): Key dimension (typically 64)
- \(n_{\text{heads}}\): Number of attention heads (typically 12)
- \(n_{\text{layers}}\): Number of transformer layers (typically 12)

### Detoxify Model
- \(d_{\text{model}}\): Model dimension (typically 768)
- \(d_k\): Key dimension (typically 64)
- \(n_{\text{heads}}\): Number of attention heads (typically 12)
- \(n_{\text{layers}}\): Number of transformer layers (typically 6) 