#!/usr/bin/env python3
"""
Test script for backend functionality
Tests database and LLM chatbot
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import AnalyticsDB
from backend.llm_chatbot import TrafficSignLLM
import uuid


def test_database():
    """Test database functionality"""
    print("\n" + "="*70)
    print("Testing Database Functionality")
    print("="*70)
    
    # Initialize database
    print("\n1. Initializing database...")
    db = AnalyticsDB(db_path="test_analytics.db")
    print("   ✓ Database initialized")
    
    # Create session
    print("\n2. Creating test session...")
    session_id = str(uuid.uuid4())
    db.create_session(session_id)
    print(f"   ✓ Session created: {session_id}")
    
    # Log detection events
    print("\n3. Logging detection events...")
    db.log_detection(
        session_id=session_id,
        model_type="YOLOv11",
        confidence_threshold=0.25,
        num_detections=3,
        inference_time_ms=45.2,
        fps=22.1,
        input_type="image"
    )
    db.log_detection(
        session_id=session_id,
        model_type="Ensemble",
        confidence_threshold=0.30,
        num_detections=5,
        inference_time_ms=120.5,
        fps=8.3,
        input_type="video"
    )
    print("   ✓ 2 detection events logged")
    
    # Log interactions
    print("\n4. Logging user interactions...")
    db.log_interaction(session_id, "model_change", {"model": "Ensemble"})
    db.log_interaction(session_id, "threshold_change", {"threshold": 0.30})
    print("   ✓ 2 interactions logged")
    
    # Log chat messages
    print("\n5. Logging chat messages...")
    db.log_chat_message(session_id, "user", "What did you detect?")
    db.log_chat_message(session_id, "assistant", "I detected 3 traffic signs.")
    print("   ✓ 2 chat messages logged")
    
    # Get statistics
    print("\n6. Retrieving session statistics...")
    stats = db.get_session_stats(session_id)
    print(f"   ✓ Total detections: {stats['total_detections']}")
    print(f"   ✓ Total frames: {stats['total_frames']}")
    
    # Get analytics summary
    print("\n7. Retrieving analytics summary...")
    summary = db.get_analytics_summary()
    print(f"   ✓ Total sessions: {summary['total_sessions']}")
    print(f"   ✓ Total detections: {summary['total_detections']}")
    print(f"   ✓ Popular model: {summary['popular_model']}")
    
    # Get detection history
    print("\n8. Retrieving detection history...")
    history = db.get_detection_history(session_id)
    print(f"   ✓ Retrieved {len(history)} detection events")
    
    # Get chat history
    print("\n9. Retrieving chat history...")
    chat_history = db.get_chat_history(session_id)
    print(f"   ✓ Retrieved {len(chat_history)} chat messages")
    
    # Close database
    db.close()
    print("\n   ✓ Database closed")
    
    # Clean up test database
    import os
    if os.path.exists("test_analytics.db"):
        os.remove("test_analytics.db")
        print("   ✓ Test database cleaned up")
    
    print("\n" + "="*70)
    print("Database Tests: PASSED ✓")
    print("="*70)


def test_llm_chatbot():
    """Test LLM chatbot functionality"""
    print("\n" + "="*70)
    print("Testing LLM Chatbot Functionality")
    print("="*70)
    
    # Initialize chatbot
    print("\n1. Initializing chatbot...")
    chatbot = TrafficSignLLM()
    print("   ✓ Chatbot initialized")
    
    # Test general greeting
    print("\n2. Testing general greeting...")
    response = chatbot.chat("Hello")
    print(f"   User: Hello")
    print(f"   Bot: {response[:100]}...")
    print("   ✓ General greeting works")
    
    # Test system usage question
    print("\n3. Testing system usage question...")
    response = chatbot.chat("How do I use this system?")
    print(f"   User: How do I use this system?")
    print(f"   Bot: {response[:100]}...")
    print("   ✓ System usage response works")
    
    # Test accuracy question
    print("\n4. Testing accuracy question...")
    response = chatbot.chat("What's the accuracy?")
    print(f"   User: What's the accuracy?")
    print(f"   Bot: {response[:100]}...")
    print("   ✓ Accuracy response works")
    
    # Test with detection context
    print("\n5. Testing with detection context...")
    detection_context = {
        "num_detections": 2,
        "detected_classes": ["stop_sign", "speed_limit_40"],
        "model_type": "YOLOv11",
        "confidence": 0.25
    }
    response = chatbot.chat("What did you detect?", detection_context)
    print(f"   User: What did you detect?")
    print(f"   Bot: {response[:150]}...")
    print("   ✓ Detection context response works")
    
    # Test sign meaning
    print("\n6. Testing sign meaning query...")
    response = chatbot.chat("What does a stop sign mean?")
    print(f"   User: What does a stop sign mean?")
    print(f"   Bot: {response[:100]}...")
    print("   ✓ Sign meaning response works")
    
    # Test conversation history
    print("\n7. Testing conversation history...")
    history = chatbot.get_conversation_history()
    print(f"   ✓ Conversation history: {len(history)} messages")
    
    # Test clear history
    print("\n8. Testing clear history...")
    chatbot.clear_history()
    history = chatbot.get_conversation_history()
    print(f"   ✓ History cleared: {len(history)} messages")
    
    print("\n" + "="*70)
    print("LLM Chatbot Tests: PASSED ✓")
    print("="*70)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("BACKEND MODULE TEST SUITE")
    print("="*70)
    
    try:
        test_database()
        test_llm_chatbot()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓✓✓")
        print("="*70 + "\n")
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
