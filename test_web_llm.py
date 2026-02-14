#!/usr/bin/env python3
"""
Test script for Web-LLM Integration

Demonstrates behavior tracking and LLM handler functionality.
"""

import sys
import time
from utils.behavior_tracker import get_tracker
from utils.web_llm_handler import get_llm_handler

print("="*70)
print("🧪 Testing Web-LLM Integration Components")
print("="*70)

# Test 1: Behavior Tracker
print("\n📊 Test 1: Behavior Tracker")
print("-" * 70)

tracker = get_tracker()
session_id = tracker.start_session()
print(f"✅ Session started: {session_id}")

# Simulate some detection events
print("\n🔄 Simulating detection events...")

detections = [
    {
        "detection_count": 3,
        "classes_detected": ["stop_sign", "speed_limit_40", "no_entry"],
        "avg_confidence": 0.92,
        "inference_time_ms": 45.2,
        "model_used": "YOLOv11",
        "image_source": "upload"
    },
    {
        "detection_count": 2,
        "classes_detected": ["stop_sign", "pedestrian_crossing"],
        "avg_confidence": 0.88,
        "inference_time_ms": 48.5,
        "model_used": "YOLOv11",
        "image_source": "webcam"
    },
    {
        "detection_count": 5,
        "classes_detected": ["stop_sign", "speed_limit_60", "one_way", "no_parking", "keep_left"],
        "avg_confidence": 0.94,
        "inference_time_ms": 52.1,
        "model_used": "Ensemble",
        "image_source": "upload"
    }
]

for i, det in enumerate(detections, 1):
    tracker.track_detection(**det)
    print(f"   ✓ Detection event {i} tracked")
    time.sleep(0.1)

# Simulate some interactions
print("\n🔄 Simulating user interactions...")
interactions = [
    {"action": "change_model", "details": {"from": "YOLOv11", "to": "Ensemble"}},
    {"action": "adjust_confidence", "details": {"threshold": 0.3}},
    {"action": "view_analytics", "details": {}}
]

for interaction in interactions:
    tracker.track_interaction(**interaction)
    print(f"   ✓ Interaction tracked: {interaction['action']}")
    time.sleep(0.1)

# Get summary
print("\n📈 Session Summary:")
print("-" * 70)
summary = tracker.get_session_summary()

print(f"Session ID: {summary['session_id']}")
print(f"Duration: {summary['duration_seconds']:.1f} seconds")
print(f"Total Detections: {summary['total_detections']}")
print(f"Total Frames: {summary['total_frames']}")
print(f"Avg Confidence: {summary['avg_confidence']:.2%}")
print(f"Avg Inference Time: {summary['avg_inference_time_ms']:.1f}ms")
print(f"Most Used Model: {summary['most_used_model']}")

print("\nTop Detected Signs:")
for item in summary['top_signs']:
    print(f"  - {item['sign'].replace('_', ' ').title()}: {item['count']}x")

# Test 2: Web-LLM Handler
print("\n\n🤖 Test 2: Web-LLM Handler")
print("-" * 70)

handler = get_llm_handler()
print("✅ Handler initialized")

# Test traffic sign database
print("\n🚦 Traffic Sign Database:")
db = handler._get_traffic_sign_database()
print(f"Categories: {list(db.keys())}")
print(f"Total signs in DB: {sum(len(signs) for signs in db.values())}")

# Test sign explanation
print("\n📖 Sign Explanation Example:")
print("-" * 70)
explanation = handler.get_sign_explanation("stop_sign")
print(explanation)

# Test session prompt generation
print("\n💬 Session Prompt Generation:")
print("-" * 70)
prompt = handler.generate_session_prompt()
print(prompt)

# Test insight generation
print("\n\n🔍 Insight Prompt Generation:")
print("-" * 70)
insight_prompt = handler.generate_insight_prompt(
    "What patterns do you see in my detection behavior?"
)
print(insight_prompt)

# Test detection result formatting
print("\n\n📊 Detection Result Formatting:")
print("-" * 70)
result = handler.format_detection_result(
    detection_count=3,
    classes=['stop_sign', 'speed_limit_40', 'stop_sign'],
    confidence=0.91,
    model='YOLOv11'
)
print(result)

# Test behavior insights
print("\n\n📈 Behavior Insights:")
print("-" * 70)
insights = tracker.get_behavior_insights()
print(f"Summary available: {'Yes' if 'summary' in insights else 'No'}")
print(f"Source distribution: {insights.get('source_distribution', {})}")
print(f"Action distribution: {insights.get('action_distribution', {})}")
print(f"Recent detections: {len(insights.get('detection_timeline', []))} events")

# Test chat context
print("\n\n🗨️ Chat Context:")
print("-" * 70)
context = handler.get_chat_context()
print(f"System prompt length: {len(context['system_prompt'])} chars")
print(f"Session data keys: {list(context['session_data'].keys())}")
print(f"Traffic sign DB categories: {list(context['traffic_sign_info'].keys())}")

# End session
print("\n\n🏁 Ending Session")
print("-" * 70)
tracker.end_session()
print("✅ Session ended and saved")

# Verify saved data
import json
from pathlib import Path

storage_path = Path("data/analytics/sessions.json")
if storage_path.exists():
    with open(storage_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Sessions saved: {len(data)} session(s) in storage")
else:
    print("⚠️ No session data file found")

print("\n" + "="*70)
print("✅ All tests completed successfully!")
print("="*70)

print("\n💡 Next Steps:")
print("   1. Start the web app: python web_app_llm.py")
print("   2. Open browser to http://localhost:7860")
print("   3. Navigate to '🤖 AI Chat' tab")
print("   4. Click 'Load AI Model' to start Web-LLM")
print("   5. Try detecting signs and chat with the AI!\n")
