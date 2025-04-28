from classifier.jailbreak_detector import JailbreakDetector
import json
import os
from datetime import datetime
import gc

def clear_gemini_memory():
    # Create the detector
    detector = JailbreakDetector()
    
    # Only reset Gemini's chat sessions
    if detector.analysis_chat:
        detector.analysis_chat = detector.gemini_model.start_chat(history=[])
    if detector.response_chat:
        detector.response_chat = detector.response_model.start_chat(history=[])
    
    # Force garbage collection
    gc.collect()
    
    print("✅ Successfully cleared Gemini's memory!")
    print("Note: Conversation history has been preserved in conversation_history.json")

if __name__ == "__main__":
    clear_gemini_memory() 