from classifier.jailbreak_detector import JailbreakDetector
import json
import os
from datetime import datetime
import gc

def clear_conversation_history():
    """Clear the conversation history file"""
    try:
        # Clear the conversation history file
        with open('conversation_history.json', 'w') as f:
            json.dump([], f)
        print("✅ Successfully cleared conversation history!")
    except Exception as e:
        print(f"❌ Error clearing conversation history: {str(e)}")

def clear_gemini_memory():
    """Clear Gemini model's memory"""
    try:
        # Create the detector
        detector = JailbreakDetector()
        
        # Reset Gemini's chat sessions
        if detector.analysis_chat:
            detector.analysis_chat = detector.gemini_model.start_chat(history=[])
        if detector.response_chat:
            detector.response_chat = detector.response_model.start_chat(history=[])
        
        # Force garbage collection
        gc.collect()
        
        print("✅ Successfully cleared Gemini's memory!")
    except Exception as e:
        print(f"❌ Error clearing Gemini memory: {str(e)}")

def clear_all():
    """Clear both conversation history and Gemini memory"""
    clear_conversation_history()
    clear_gemini_memory()
    print("✅ All memory cleared successfully!")

if __name__ == "__main__":
    clear_all() 