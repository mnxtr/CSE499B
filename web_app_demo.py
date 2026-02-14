#!/usr/bin/env python3
"""
Mock Web Interface Demo - LLM Chatbot and Analytics (No Model Required)

This is a demonstration version that works without the actual detection model.
It simulates detections to showcase the LLM chatbot and analytics features.
"""

import gradio as gr
import numpy as np
import time
import uuid
from datetime import datetime
from typing import Tuple, List, Dict
import random

from backend.database import AnalyticsDB
from backend.llm_chatbot import TrafficSignLLM


print("\n" + "="*70)
print("🚦  MOCK DEMO: LLM & ANALYTICS (No Model Required)")
print("="*70)
print("🔄 Initializing components...\n")

# Initialize database and LLM
print(f"   🗄️  Initializing analytics database...")
analytics_db = AnalyticsDB(db_path="demo_analytics.db")

print(f"   🤖 Initializing LLM chatbot...")
llm_chatbot = TrafficSignLLM()

print(f"   ✅ System ready!")
print("\n" + "="*70)
print("🎉  Launching demo interface...")
print("="*70 + "\n")

# Session management
current_session_id = str(uuid.uuid4())
analytics_db.create_session(current_session_id)

# Mock class names
class_names = [
    "stop_sign", "speed_limit_40", "speed_limit_60", "no_entry", 
    "no_parking", "one_way", "pedestrian_crossing", "school_zone",
    "roundabout", "keep_left", "keep_right"
]

# Detection statistics
stats = {
    'total_detections': 0,
    'total_frames': 0,
    'avg_fps': 0,
    'avg_confidence': 0,
    'session_start': datetime.now(),
    'last_detection_context': None
}


def simulate_detection(image: np.ndarray, model_choice: str, conf_threshold: float) -> Dict:
    """Simulate a detection result"""
    if image is None:
        return None
    
    # Randomly decide how many signs to "detect"
    num_detections = random.randint(0, 4)
    
    # Randomly select detected classes
    detected_classes = random.sample(class_names, min(num_detections, len(class_names)))
    
    # Simulate inference time based on model
    if model_choice == "YOLOv11":
        inference_time = random.uniform(0.040, 0.050)  # 40-50ms
    elif model_choice == "Multi-Scale YOLOv11":
        inference_time = random.uniform(0.080, 0.100)  # 80-100ms
    else:  # Ensemble
        inference_time = random.uniform(0.120, 0.150)  # 120-150ms
    
    fps = 1.0 / inference_time
    
    # Simulate confidence scores
    confidences = [random.uniform(conf_threshold, 0.95) for _ in range(num_detections)]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        "num_detections": num_detections,
        "detected_classes": detected_classes,
        "inference_time": inference_time,
        "fps": fps,
        "avg_confidence": avg_confidence
    }


def draw_mock_detection(image: np.ndarray, detection_result: Dict) -> np.ndarray:
    """Draw mock detection boxes on image"""
    if image is None or detection_result is None:
        return image
    
    # Just return the original image (in real version, would draw boxes)
    # Add a text overlay to indicate mock mode
    annotated = image.copy()
    
    return annotated


def process_image(
    image: np.ndarray, 
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25,
    show_stats: bool = True
) -> Tuple[np.ndarray, str]:
    """Process a single image (mock version)"""
    if image is None:
        return None, "No image provided"
    
    # Simulate detection
    detection = simulate_detection(image, model_choice, conf_threshold)
    
    # Draw detections
    annotated = draw_mock_detection(image, detection)
    
    # Update stats
    stats['total_detections'] += detection['num_detections']
    stats['total_frames'] += 1
    stats['avg_fps'] = (stats['avg_fps'] * (stats['total_frames'] - 1) + detection['fps']) / stats['total_frames']
    stats['avg_confidence'] = detection['avg_confidence']
    
    # Store detection context for chatbot
    stats['last_detection_context'] = {
        "num_detections": detection['num_detections'],
        "detected_classes": detection['detected_classes'],
        "model_type": model_choice,
        "confidence": conf_threshold
    }
    
    # Log to analytics database
    analytics_db.log_detection(
        session_id=current_session_id,
        model_type=model_choice,
        confidence_threshold=conf_threshold,
        num_detections=detection['num_detections'],
        inference_time_ms=detection['inference_time'] * 1000,
        fps=detection['fps'],
        input_type="image"
    )
    
    # Log interaction
    analytics_db.log_interaction(
        session_id=current_session_id,
        interaction_type="image_detection",
        details={"model": model_choice, "detections": detection['num_detections']}
    )
    
    # Generate statistics text
    stats_text = f"**📊 Detection Results (MOCK)**\n\n"
    stats_text += f"**Mode:** {model_choice}\n"
    stats_text += f"**Detections:** {detection['num_detections']}\n"
    stats_text += f"**Inference Time:** {detection['inference_time']*1000:.1f}ms\n"
    stats_text += f"**FPS:** {detection['fps']:.1f}\n"
    
    if detection['num_detections'] > 0:
        stats_text += f"**Avg Confidence:** {detection['avg_confidence']:.2%}\n\n"
        stats_text += "**Classes Detected:**\n"
        for class_name in detection['detected_classes']:
            stats_text += f"- {class_name.replace('_', ' ').title()}\n"
    else:
        stats_text += "\n*No signs detected in this image.*"
    
    return annotated, stats_text


def chat_with_llm(message: str, history: List) -> Tuple[str, List]:
    """Handle chat interaction with LLM chatbot"""
    if not message or message.strip() == "":
        return "", history
    
    # Get response from LLM with detection context
    response = llm_chatbot.chat(message, stats['last_detection_context'])
    
    # Log chat message to database
    analytics_db.log_chat_message(current_session_id, "user", message)
    analytics_db.log_chat_message(current_session_id, "assistant", response)
    
    # Log interaction
    analytics_db.log_interaction(
        session_id=current_session_id,
        interaction_type="chat_message",
        details={"message_length": len(message)}
    )
    
    # Update history
    history.append((message, response))
    
    return "", history


def get_session_stats() -> str:
    """Get current session statistics"""
    runtime = datetime.now() - stats['session_start']
    
    return f"""## 📈 Session Statistics

**Session Duration:** {str(runtime).split('.')[0]}
**Total Frames Processed:** {stats['total_frames']}
**Total Detections:** {stats['total_detections']}
**Average FPS:** {stats['avg_fps']:.1f}
**Last Avg Confidence:** {stats['avg_confidence']:.2%}
"""


def get_analytics_dashboard() -> str:
    """Get overall analytics dashboard"""
    summary = analytics_db.get_analytics_summary()
    
    dashboard = f"""## 📊 Analytics Dashboard

### Overall Statistics
- **Total Sessions:** {summary['total_sessions']}
- **Total Detections:** {summary['total_detections']}
- **Total Frames Processed:** {summary['total_frames']}
- **Average FPS:** {summary['avg_fps']}

### Popular Choices
- **Most Used Model:** {summary['popular_model']}
- **Most Used Input Type:** {summary['popular_input_type']}

*Note: This is a demo with simulated data*
"""
    
    return dashboard


# Custom CSS
CUSTOM_CSS = """
.header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 24px 32px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.info-section {
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid #667eea;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
}

.demo-warning {
    background: rgba(255, 193, 7, 0.1);
    border-left: 4px solid #ffc107;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
    color: #ffc107;
}
"""


# Build the Gradio interface
with gr.Blocks(css=CUSTOM_CSS, title="BD Traffic Sign Detection - Demo") as demo:
    
    # Header
    gr.HTML("""
    <div class="header-banner">
        <h1>🚦 Bangladesh Traffic Sign Detection - LLM & Analytics Demo</h1>
        <p>AI Chatbot Assistant • User Behavior Analytics</p>
    </div>
    """)
    
    # Demo warning
    gr.HTML("""
    <div class="demo-warning">
        ⚠️ <strong>DEMO MODE:</strong> This version simulates detections to showcase the LLM chatbot and analytics features without requiring the actual detection model.
    </div>
    """)
    
    # Main tabs
    with gr.Tabs():
        
        # Tab 1: Image Detection with Chatbot
        with gr.Tab("🖼️ Image Detection", id="image"):
            gr.HTML('<div class="info-section">📌 <strong>Upload an image</strong> and ask the AI assistant about the detected signs!</div>')
            
            with gr.Row():
                # Left column: Controls and Image
                with gr.Column(scale=2):
                    with gr.Row():
                        image_model = gr.Radio(
                            choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                            value="YOLOv11",
                            label="Detection Model (Simulated)"
                        )
                        image_conf = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                    
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_btn = gr.Button("🔍 Detect Signs (Simulated)", variant="primary", size="lg")
                    
                    image_output = gr.Image(label="Detection Result", type="numpy")
                    image_stats = gr.Markdown(label="Statistics")
                
                # Right column: AI Chatbot
                with gr.Column(scale=1):
                    gr.HTML('<div class="info-section">🤖 <strong>AI Assistant</strong> - Ask about detected signs!</div>')
                    
                    chatbot = gr.Chatbot(
                        label="Traffic Sign AI Assistant",
                        height=500,
                        type="messages"
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask me about the detected signs...",
                            label="Your Question",
                            scale=4
                        )
                        chat_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    gr.HTML("""
                    <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                        <strong>Try asking:</strong><br>
                        • "What signs did you detect?"<br>
                        • "What does a stop sign mean?"<br>
                        • "How do I use this system?"<br>
                        • "What's the accuracy?"
                    </div>
                    """)
            
            # Connect interactions
            image_btn.click(
                process_image,
                inputs=[image_input, image_model, image_conf],
                outputs=[image_output, image_stats]
            )
            
            chat_btn.click(
                chat_with_llm,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )
            
            chat_input.submit(
                chat_with_llm,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )
        
        # Tab 2: Analytics Dashboard
        with gr.Tab("📊 Analytics", id="analytics"):
            gr.HTML('<div class="info-section">📌 <strong>User behavior analytics</strong> and session statistics.</div>')
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Current Session")
                    session_stats_display = gr.Markdown(value=get_session_stats())
                    refresh_session_btn = gr.Button("🔄 Refresh Session Stats")
                    
                with gr.Column():
                    gr.Markdown("### Overall Analytics")
                    analytics_display = gr.Markdown(value=get_analytics_dashboard())
                    refresh_analytics_btn = gr.Button("🔄 Refresh Analytics")
            
            refresh_session_btn.click(
                get_session_stats,
                outputs=session_stats_display
            )
            
            refresh_analytics_btn.click(
                get_analytics_dashboard,
                outputs=analytics_display
            )
        
        # Tab 3: About
        with gr.Tab("ℹ️ About", id="about"):
            gr.Markdown("""
## 🚦 Demo Information

This is a **demonstration version** that showcases:

### 🤖 LLM Chatbot Features
- Knowledge-based AI assistant
- Context-aware responses
- Traffic sign information database
- No external API required

### 📊 Analytics Features
- Real-time event tracking
- Session management
- User interaction logging
- SQLite database storage

### 📋 How It Works

This demo simulates detection results to demonstrate the chatbot and analytics features.
In the full version with the actual model:
1. Real YOLOv11 detection
2. Accurate bounding boxes
3. Real inference timing
4. Production-ready performance

### 🚀 To Use the Full Version

```bash
# Ensure you have the trained model
python web_app_llm.py
```

Then open http://localhost:7860 in your browser.
            """)
    
    # Footer
    gr.HTML("""
    <footer style="text-align: center; padding: 24px; opacity: 0.7; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 32px;">
        <p>🚀 Demo Version - LLM & Analytics Features | 🇧🇩 Bangladesh Traffic Sign Detection</p>
    </footer>
    """)


if __name__ == "__main__":
    try:
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    finally:
        # Close database connection on shutdown
        analytics_db.close()
        print("\n✓ Demo completed. Database closed.")
