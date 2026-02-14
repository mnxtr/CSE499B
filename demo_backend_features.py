#!/usr/bin/env python3
"""
Demo script for LLM chatbot and analytics features
Can be run without the actual detection model
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import AnalyticsDB
from backend.llm_chatbot import TrafficSignLLM
import uuid
from datetime import datetime


def demo_chatbot():
    """Demonstrate chatbot functionality"""
    print("\n" + "="*70)
    print("🤖 LLM CHATBOT DEMO")
    print("="*70 + "\n")
    
    chatbot = TrafficSignLLM()
    
    print("Welcome to the Bangladesh Traffic Sign Detection AI Assistant!\n")
    print("Type 'quit' or 'exit' to end the demo.\n")
    print("-" * 70)
    
    # Sample detection context
    detection_context = {
        "num_detections": 3,
        "detected_classes": ["stop_sign", "speed_limit_40", "no_parking"],
        "model_type": "YOLOv11",
        "confidence": 0.25
    }
    
    # Set initial context
    chatbot.set_detection_context(
        detection_context["num_detections"],
        detection_context["detected_classes"],
        detection_context["model_type"],
        detection_context["confidence"]
    )
    
    print("\n[Simulated Detection: 3 signs detected - stop_sign, speed_limit_40, no_parking]\n")
    
    # Sample questions
    sample_questions = [
        "What did you detect?",
        "What does a stop sign mean?",
        "How accurate is this system?",
        "How do I use this system?"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n💬 Question {i}: {question}")
        print("-" * 70)
        response = chatbot.chat(question)
        print(f"\n🤖 Assistant:\n{response}")
        print("-" * 70)
        
        if i < len(sample_questions):
            input("\nPress Enter to continue to next question...")
    
    print("\n" + "="*70)
    print("✓ Chatbot demo completed!")
    print("="*70 + "\n")


def demo_analytics():
    """Demonstrate analytics functionality"""
    print("\n" + "="*70)
    print("📊 ANALYTICS DEMO")
    print("="*70 + "\n")
    
    # Initialize database
    db = AnalyticsDB(db_path="demo_analytics.db")
    
    # Create multiple sessions
    print("Creating sample sessions and logging events...\n")
    
    for i in range(3):
        session_id = str(uuid.uuid4())
        db.create_session(session_id)
        print(f"Session {i+1}: {session_id}")
        
        # Log some detection events
        for j in range(5):
            db.log_detection(
                session_id=session_id,
                model_type=["YOLOv11", "Ensemble", "Multi-Scale"][i % 3],
                confidence_threshold=0.25,
                num_detections=j + 1,
                inference_time_ms=40 + j * 5,
                fps=22.0 - j,
                input_type=["image", "webcam", "video"][j % 3]
            )
        
        # Log some interactions
        db.log_interaction(session_id, "model_change", {"model": "YOLOv11"})
        db.log_interaction(session_id, "confidence_change", {"threshold": 0.30})
        
        # Log some chat messages
        db.log_chat_message(session_id, "user", "What did you detect?")
        db.log_chat_message(session_id, "assistant", "I detected 3 traffic signs.")
    
    print("\n" + "-" * 70)
    print("\n📈 Analytics Summary:")
    print("-" * 70)
    
    summary = db.get_analytics_summary()
    print(f"\n  Total Sessions: {summary['total_sessions']}")
    print(f"  Total Detections: {summary['total_detections']}")
    print(f"  Total Frames: {summary['total_frames']}")
    print(f"  Average FPS: {summary['avg_fps']}")
    print(f"  Most Popular Model: {summary['popular_model']}")
    print(f"  Most Popular Input Type: {summary['popular_input_type']}")
    
    print("\n" + "-" * 70)
    print("\n📋 Recent Sessions:")
    print("-" * 70)
    
    sessions = db.get_all_sessions_stats()
    for idx, session in enumerate(sessions[:5], 1):
        print(f"\n  Session {idx}:")
        print(f"    ID: {session['session_id'][:8]}...")
        print(f"    Start: {session['start_time']}")
        print(f"    Detections: {session['total_detections']}")
        print(f"    Frames: {session['total_frames']}")
    
    # Clean up
    db.close()
    
    print("\n" + "="*70)
    print("✓ Analytics demo completed!")
    print("="*70 + "\n")
    
    # Clean up demo database
    import os
    if os.path.exists("demo_analytics.db"):
        os.remove("demo_analytics.db")
        print("Demo database cleaned up.\n")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🚦 BANGLADESH TRAFFIC SIGN DETECTION - BACKEND DEMO")
    print("="*70)
    print("\nThis demo showcases the new features:")
    print("  1. LLM Chatbot Assistant")
    print("  2. User Behavior Analytics")
    print("\nNote: This demo runs without the actual detection model.")
    print("="*70)
    
    input("\nPress Enter to start the chatbot demo...")
    demo_chatbot()
    
    input("\nPress Enter to start the analytics demo...")
    demo_analytics()
    
    print("\n" + "="*70)
    print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nTo use these features with real detection:")
    print("  1. Ensure you have the trained model (best.pt)")
    print("  2. Run: python web_app_llm.py")
    print("  3. Open http://localhost:7860 in your browser")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
