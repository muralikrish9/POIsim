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

    StreamlitApp --> UI
    StreamlitApp --> Charts
    UI --> Core
    Charts --> Data
    JailbreakDetector --> RiskCalculator
    JailbreakDetector --> SentimentAnalyzer
    RiskCalculator --> Models
    SentimentAnalyzer --> Models
    GeminiModel --> JailbreakDetector
    DetoxifyModel --> JailbreakDetector
    BERTModel --> JailbreakDetector
    DistilBERTModel --> SentimentAnalyzer
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
