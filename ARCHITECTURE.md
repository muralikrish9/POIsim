# System Architecture

```mermaid
flowchart TB
    %% Frontend Layer
    subgraph Frontend["Frontend Layer"]
        direction TB
        StreamlitApp["Streamlit Dashboard"]
        UI["User Interface"]
        Charts["Visualization Charts"]
        
        StreamlitApp --> UI
        StreamlitApp --> Charts
    end
    
    %% Core Layer
    subgraph Core["Core Layer"]
        direction TB
        JailbreakDetector["Jailbreak Detector"]
        RiskCalculator["Risk Calculator"]
        SentimentAnalyzer["Sentiment Analyzer"]
        
        JailbreakDetector --> RiskCalculator
        JailbreakDetector --> SentimentAnalyzer
    end
    
    %% Models Layer
    subgraph Models["Models Layer"]
        direction TB
        GeminiModel["Gemini Model"]
        DetoxifyModel["Detoxify Model"]
        BERTModel["BERT Model"]
        DistilBERTModel["DistilBERT Model"]
    end
    
    %% Data Layer
    subgraph Data["Data Layer"]
        direction TB
        HistoryManager["History Manager"]
        ConversationDB["Conversation History"]
        
        HistoryManager --> ConversationDB
    end
    
    %% Cross-Layer Connections
    UI --> JailbreakDetector
    Charts --> HistoryManager
    RiskCalculator --> Models
    SentimentAnalyzer --> Models
    JailbreakDetector --> HistoryManager
    
    %% Styling
    classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef core fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#2e7d32
    classDef models fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#c2185b
    
    class Frontend,StreamlitApp,UI,Charts frontend
    class Core,JailbreakDetector,RiskCalculator,SentimentAnalyzer core
    class Models,GeminiModel,DetoxifyModel,BERTModel,DistilBERTModel models
    class Data,HistoryManager,ConversationDB data
