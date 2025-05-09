# Jailbreak Detection System Architecture

## Workflow Diagram

```mermaid
sequenceDiagram
    participant User
    participant StreamlitApp
    participant JailbreakDetector
    participant GeminiModel
    participant DetoxifyModel
    participant HistoryManager

    User->>StreamlitApp: Enter Prompt
    StreamlitApp->>JailbreakDetector: Analyze Prompt
    JailbreakDetector->>DetoxifyModel: Check Toxicity
    DetoxifyModel-->>JailbreakDetector: Toxicity Scores
    JailbreakDetector->>GeminiModel: Analyze Intent
    GeminiModel-->>JailbreakDetector: Analysis Results
    JailbreakDetector->>JailbreakDetector: Calculate Risk Score
    JailbreakDetector-->>StreamlitApp: Analysis Results
    StreamlitApp->>HistoryManager: Save to History
    StreamlitApp-->>User: Display Results

    Note over User,HistoryManager: Real-time Analysis Flow
```

## Architecture Diagram

```mermaid
graph TB
    subgraph Frontend
        StreamlitApp[Streamlit Dashboard]
        UI[User Interface]
        Charts[Visualization Charts]
    end

    subgraph Core
        JailbreakDetector[Jailbreak Detector]
        RiskCalculator[Risk Calculator]
        SentimentAnalyzer[Sentiment Analyzer]
    end

    subgraph Models
        GeminiModel[Gemini Model]
        DetoxifyModel[Detoxify Model]
        BERTModel[BERT Model]
        DistilBERTModel[DistilBERT Model]
    end

    subgraph Data
        HistoryManager[History Manager]
        ConversationDB[(Conversation History)]
    end

    %% Frontend Connections
    StreamlitApp --> UI
    StreamlitApp --> Charts
    UI --> Core
    Charts --> Data

    %% Core Connections
    JailbreakDetector --> RiskCalculator
    JailbreakDetector --> SentimentAnalyzer
    RiskCalculator --> Models
    SentimentAnalyzer --> Models

    %% Model Connections
    GeminiModel --> JailbreakDetector
    DetoxifyModel --> JailbreakDetector
    BERTModel --> JailbreakDetector
    DistilBERTModel --> SentimentAnalyzer

    %% Data Connections
    HistoryManager --> ConversationDB
    JailbreakDetector --> HistoryManager

    classDef frontend fill:#f9f,stroke:#333,stroke-width:2px
    classDef core fill:#bbf,stroke:#333,stroke-width:2px
    classDef models fill:#bfb,stroke:#333,stroke-width:2px
    classDef data fill:#fbb,stroke:#333,stroke-width:2px

    class StreamlitApp,UI,Charts frontend
    class JailbreakDetector,RiskCalculator,SentimentAnalyzer core
    class GeminiModel,DetoxifyModel,BERTModel,DistilBERTModel models
    class HistoryManager,ConversationDB data
```

## Component Description

### Frontend Layer
- **Streamlit Dashboard**: Main application interface
- **User Interface**: Input forms, buttons, and interactive elements
- **Visualization Charts**: Data visualization components

### Core Layer
- **Jailbreak Detector**: Main analysis engine
- **Risk Calculator**: Calculates risk scores and thresholds
- **Sentiment Analyzer**: Analyzes text sentiment

### Models Layer
- **Gemini Model**: Handles intent analysis and response generation
- **Detoxify Model**: Analyzes toxicity and harmful content
- **BERT Model**: General text analysis and jailbreak detection
- **DistilBERT Model**: Specialized sentiment analysis

### Data Layer
- **History Manager**: Manages conversation history
- **Conversation Database**: Stores analysis results and history

## Key Features and Flows

### 1. Prompt Analysis Flow
- User inputs prompt
- System performs multi-model analysis
- Results are displayed in real-time
- History is updated

### 2. History Management
- Automatic saving of analysis results
- History viewing and filtering
- Data visualization
- Memory clearing capability

### 3. Risk Assessment
- Multi-factor analysis
- Real-time scoring
- Threshold-based classification
- Detailed explanation generation

### 4. Data Visualization
- Interactive charts
- Real-time updates
- Filtering capabilities
- Export functionality

## System Requirements

- Python 3.8+
- Streamlit
- Google Gemini API
- Detoxify
- BERT (bert-base-uncased)
- DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- Pandas
- Plotly

## Setup and Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

3. Run the application:
```bash
streamlit run streamlit_app.py
``` 