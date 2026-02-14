#!/usr/bin/env python3
"""
Enhanced Web Interface with LLM Chatbot and User Behavior Analytics

New Features:
- AI Chatbot for user assistance and traffic sign information
- User behavior tracking and analytics
- Session management
- Real-time analytics dashboard
"""

import gradio as gr
import cv2
import numpy as np
import yaml
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime

from ensemble_detector import EnsembleDetector, YOLOv11Detector
from backend.database import AnalyticsDB
from backend.llm_chatbot import TrafficSignLLM


print("\n" + "="*70)
print("🚦  BANGLADESH TRAFFIC SIGN DETECTION SYSTEM - LLM ENHANCED")
print("="*70)
print("🔄 Initializing components...\n")

# Configuration
MODEL_PATH = "results/bd_signs_v1/weights/best.pt"
CONFIG_PATH = "data/processed/data.yaml"

# Initialize detectors
print(f"   📦 Loading YOLOv11 model...")
yolo_detector = YOLOv11Detector(model_path=MODEL_PATH)

print(f"   📦 Initializing Ensemble detector...")
ensemble_detector = EnsembleDetector(
    yolo_path=MODEL_PATH,
    use_multi_scale=True,
    scales=[480, 640, 800]
)

# Initialize database and LLM
print(f"   🗄️  Initializing analytics database...")
analytics_db = AnalyticsDB(db_path="analytics.db")

print(f"   🤖 Initializing LLM chatbot...")
llm_chatbot = TrafficSignLLM()

# Load class names
try:
    with open(CONFIG_PATH, "r") as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get("names", [])
except Exception as e:
    class_names = list(ensemble_detector.class_names.values())
    print(f"   ⚠️ Could not load config, using model class names")

model_size = os.path.getsize(MODEL_PATH) / (1024*1024)  # MB

print(f"   ✅ Loaded {len(class_names)} traffic sign classes")
print(f"   📊 Model size: {model_size:.2f} MB")
print("\n" + "="*70)
print("🎉  System ready! Launching enhanced web interface with LLM...")
print("="*70 + "\n")

# Session management
current_session_id = str(uuid.uuid4())
analytics_db.create_session(current_session_id)

# Detection statistics
stats = {
    'total_detections': 0,
    'total_frames': 0,
    'avg_fps': 0,
    'avg_confidence': 0,
    'session_start': datetime.now(),
    'last_detection_context': None
}


def process_image(
    image: np.ndarray, 
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25,
    show_stats: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Process a single image with the selected detection model.
    Logs analytics and updates detection context.
    """
    if image is None:
        return None, "No image provided"
    
    start_time = time.time()
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Get mode
    if model_choice == "YOLOv11":
        mode = "yolo"
    elif model_choice == "Multi-Scale YOLOv11":
        mode = "multi_scale"
    elif model_choice == "Ensemble":
        mode = "ensemble"
    else:
        mode = "yolo"
    
    # Run detection
    ensemble_detector.conf_threshold = conf_threshold
    ensemble_detector.yolo.conf_threshold = conf_threshold
    
    boxes, scores, labels, _ = ensemble_detector.predict(image_bgr, mode=mode)
    
    # Draw detections
    annotated = ensemble_detector.draw_detections(image_bgr, boxes, scores, labels)
    
    # Convert back to RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    
    # Get detected class names
    detected_classes = [ensemble_detector.class_names.get(int(label), f"class_{label}") 
                       for label in labels]
    
    # Update stats
    stats['total_detections'] += len(boxes)
    stats['total_frames'] += 1
    stats['avg_fps'] = (stats['avg_fps'] * (stats['total_frames'] - 1) + fps) / stats['total_frames']
    if len(scores) > 0:
        stats['avg_confidence'] = float(scores.mean())
    
    # Store detection context for chatbot
    stats['last_detection_context'] = {
        "num_detections": len(boxes),
        "detected_classes": detected_classes,
        "model_type": model_choice,
        "confidence": conf_threshold
    }
    
    # Log to analytics database
    analytics_db.log_detection(
        session_id=current_session_id,
        model_type=model_choice,
        confidence_threshold=conf_threshold,
        num_detections=len(boxes),
        inference_time_ms=inference_time * 1000,
        fps=fps,
        input_type="image"
    )
    
    # Log interaction
    analytics_db.log_interaction(
        session_id=current_session_id,
        interaction_type="image_detection",
        details={"model": model_choice, "detections": len(boxes)}
    )
    
    # Generate statistics text
    if show_stats and len(boxes) > 0:
        summary = ensemble_detector.get_detection_summary(boxes, scores, labels)
        
        stats_text = f"**📊 Detection Results**\n\n"
        stats_text += f"**Mode:** {model_choice}\n"
        stats_text += f"**Detections:** {len(boxes)}\n"
        stats_text += f"**Inference Time:** {inference_time*1000:.1f}ms\n"
        stats_text += f"**FPS:** {fps:.1f}\n"
        stats_text += f"**Avg Confidence:** {summary['avg_confidence']:.2%}\n\n"
        
        stats_text += "**Classes Detected:**\n"
        for item in summary['classes_detected']:
            stats_text += f"- {item['class_name']}: {item['count']}\n"
    else:
        stats_text = f"**📊 Detection Results**\n\n"
        stats_text += f"**Mode:** {model_choice}\n"
        stats_text += f"**Detections:** {len(boxes)}\n"
        stats_text += f"**Inference Time:** {inference_time*1000:.1f}ms\n"
        stats_text += f"**FPS:** {fps:.1f}\n"
        if len(boxes) == 0:
            stats_text += "\n*No signs detected in this image.*"
    
    return annotated_rgb, stats_text


def process_webcam(
    frame: np.ndarray,
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25
) -> np.ndarray:
    """Process webcam frame in real-time."""
    if frame is None:
        return None
    
    annotated, _ = process_image(frame, model_choice, conf_threshold, show_stats=False)
    return annotated


def chat_with_llm(message: str, history: List) -> Tuple[str, List]:
    """
    Handle chat interaction with LLM chatbot.
    Uses detection context to provide relevant responses.
    """
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
    """Get current session statistics."""
    runtime = datetime.now() - stats['session_start']
    
    return f"""## 📈 Session Statistics

**Session Duration:** {str(runtime).split('.')[0]}
**Total Frames Processed:** {stats['total_frames']}
**Total Detections:** {stats['total_detections']}
**Average FPS:** {stats['avg_fps']:.1f}
**Last Avg Confidence:** {stats['avg_confidence']:.2%}
"""


def get_analytics_dashboard() -> str:
    """Get overall analytics dashboard."""
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
"""
    
    return dashboard


# Custom CSS for enhanced dark theme
CUSTOM_CSS = """
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --dark-bg: #0f0f1a;
    --card-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
}

.dark {
    --background-fill-primary: #0f0f1a !important;
}

.header-banner {
    background: var(--primary-gradient);
    padding: 24px 32px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.header-banner h1 {
    margin: 0;
    font-size: 2.2em;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.header-banner p {
    margin: 8px 0 0 0;
    font-size: 1.1em;
    opacity: 0.95;
}

.stat-card {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
}

.info-section {
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid #667eea;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
}

.chatbot-container {
    height: 500px;
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    overflow-y: auto;
}

footer {
    text-align: center;
    padding: 24px;
    opacity: 0.7;
    border-top: 1px solid var(--glass-border);
    margin-top: 32px;
}

.gr-button-primary {
    background: var(--primary-gradient) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}
"""


# Build the Gradio interface
with gr.Blocks(css=CUSTOM_CSS, title="BD Traffic Sign Detection - LLM Enhanced") as demo:
    
    # Header
    gr.HTML("""
    <div class="header-banner">
        <h1>🚦 Bangladesh Traffic Sign Detection with AI Assistant</h1>
        <p>Real-time detection powered by YOLOv11 • AI Chatbot • User Analytics</p>
    </div>
    """)
    
    # Quick stats row
    with gr.Row():
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">22.2</div>
            <div class="stat-label">FPS (CPU)</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">99.45%</div>
            <div class="stat-label">mAP@50</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">📦</div>
            <div class="stat-value">{model_size:.1f}</div>
            <div class="stat-label">MB Model</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div class="stat-icon">🤖</div>
            <div class="stat-value">AI</div>
            <div class="stat-label">LLM Assistant</div>
        </div>
        ''')
    
    gr.HTML("<br>")
    
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
                            label="Detection Model"
                        )
                        image_conf = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                    
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_btn = gr.Button("🔍 Detect Signs", variant="primary", size="lg")
                    
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
        
        # Tab 2: Real-time Webcam
        with gr.Tab("📹 Live Webcam", id="webcam"):
            gr.HTML('<div class="info-section">📌 <strong>Real-time detection</strong> from your webcam.</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_model = gr.Radio(
                        choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                        value="YOLOv11",
                        label="Detection Model"
                    )
                    webcam_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                
                with gr.Column(scale=3):
                    with gr.Row():
                        webcam_input = gr.Image(
                            sources=["webcam"], 
                            streaming=True, 
                            type="numpy",
                            label="Camera Input"
                        )
                        webcam_output = gr.Image(
                            label="Detected Signs",
                            type="numpy"
                        )
            
            webcam_input.stream(
                process_webcam,
                inputs=[webcam_input, webcam_model, webcam_conf],
                outputs=webcam_output
            )
        
        # Tab 3: Analytics Dashboard
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
        
        # Tab 4: About & Help
        with gr.Tab("ℹ️ About", id="about"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
## 🚦 System Information

**Model Path:** `{MODEL_PATH}`  
**Architecture:** YOLOv11 Nano  
**Accuracy:** 99.45% mAP@50  
**Size:** {model_size:.2f} MB

---

## 🤖 AI Assistant Features

The integrated LLM chatbot can:
- Explain detection results in detail
- Answer questions about traffic signs
- Provide usage guidance
- Share system performance information

---

## 📊 Analytics Features

User behavior tracking includes:
- Detection events per session
- Model selection preferences
- Chat interactions
- Session duration and statistics

---

## 🔧 Detection Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| **YOLOv11** | Standard detection at 640px | ⚡ Fast | High |
| **Multi-Scale** | Detection at 480, 640, 800px | 🔄 Medium | Higher |
| **Ensemble** | Combines all scales with WBF | 🎯 Slower | Highest |
                    """)
                
                with gr.Column():
                    gr.Markdown(f"""
## 📋 Detected Traffic Sign Classes

Total Classes: **{len(class_names)}**

### Categories:
- **Regulatory Signs:** Stop, Speed Limits, No Entry, No Parking, One Way
- **Warning Signs:** Pedestrian Crossing, School Zone, Curves, Intersections
- **Mandatory Signs:** Roundabout, Keep Left/Right, Bicycle Path

---

## 💬 Chat Examples

Try asking the AI assistant:

1. **About Detections:**
   - "What signs did you detect?"
   - "Tell me about the stop sign"

2. **About Usage:**
   - "How do I use this system?"
   - "Which model should I choose?"

3. **About Traffic Signs:**
   - "What does a speed limit sign mean?"
   - "What are regulatory signs?"

4. **About Performance:**
   - "How accurate is this system?"
   - "What's the FPS?"
                    """)
    
    # Footer
    gr.HTML("""
    <footer>
        <p>🚀 Powered by YOLOv11 & LLM Technology | 🇧🇩 Bangladesh Traffic Sign Detection</p>
        <p style="font-size: 0.9em; opacity: 0.7;">Enhanced with AI Assistant and User Behavior Analytics</p>
    </footer>
    """)


if __name__ == "__main__":
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    finally:
        # Close database connection on shutdown
        analytics_db.close()
