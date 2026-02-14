#!/usr/bin/env python3
"""
Enhanced Web Application with Web-LLM Integration for User Behavior Analysis

Features:
- User behavior tracking
- Web-LLM powered chat assistant
- Session analytics and insights
- Real-time detection with behavior analysis
"""

import gradio as gr
import cv2
import numpy as np
import yaml
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

from ensemble_detector import EnsembleDetector, YOLOv11Detector
from utils.behavior_tracker import get_tracker
from utils.web_llm_handler import get_llm_handler


print("\n" + "="*70)
print("🚦  BANGLADESH TRAFFIC SIGN DETECTION - WEB-LLM ENHANCED")
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

# Initialize behavior tracking
print(f"   📊 Initializing behavior tracking...")
tracker = get_tracker()
llm_handler = get_llm_handler()

# Start session
session_id = tracker.start_session()
print(f"   ✅ Session started: {session_id}")

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
print("🎉  System ready with Web-LLM integration!")
print("="*70 + "\n")


def process_image_with_tracking(
    image: np.ndarray, 
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25,
    image_source: str = "upload"
) -> Tuple[np.ndarray, str]:
    """
    Process image with detection and behavior tracking.
    """
    if image is None:
        return None, "No image provided"
    
    start_time = time.time()
    
    # Track interaction
    tracker.track_interaction("detect", {
        "model": model_choice,
        "confidence": conf_threshold,
        "source": image_source
    })
    
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
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Get class names for tracking
    detected_classes = [
        ensemble_detector.class_names.get(label, f"class_{label}") 
        for label in labels
    ]
    
    # Track detection event
    tracker.track_detection(
        detection_count=len(boxes),
        classes_detected=detected_classes,
        avg_confidence=float(scores.mean()) if len(scores) > 0 else 0.0,
        inference_time_ms=inference_time,
        model_used=model_choice,
        image_source=image_source
    )
    
    # Generate statistics text
    if len(boxes) > 0:
        summary = ensemble_detector.get_detection_summary(boxes, scores, labels)
        
        stats_text = f"**🎯 Detection Results**\n\n"
        stats_text += f"**Model:** {model_choice}\n"
        stats_text += f"**Detections:** {len(boxes)}\n"
        stats_text += f"**Inference Time:** {inference_time:.1f}ms\n"
        stats_text += f"**Avg Confidence:** {summary['avg_confidence']:.2%}\n\n"
        
        stats_text += "**Classes Detected:**\n"
        for item in summary['classes_detected']:
            stats_text += f"- {item['class_name']}: {item['count']}\n"
        
        # Add LLM-generated insight
        stats_text += f"\n---\n\n"
        stats_text += llm_handler.format_detection_result(
            len(boxes), detected_classes, 
            summary['avg_confidence'], model_choice
        )
    else:
        stats_text = f"**📊 Detection Results**\n\n"
        stats_text += f"**Model:** {model_choice}\n"
        stats_text += f"**Detections:** 0\n"
        stats_text += f"**Inference Time:** {inference_time:.1f}ms\n"
        stats_text += "\n*No signs detected. Try adjusting confidence threshold or use a different image.*"
    
    return annotated_rgb, stats_text


def process_webcam(
    frame: np.ndarray,
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25
) -> np.ndarray:
    """Process webcam frame with tracking."""
    if frame is None:
        return None
    
    annotated, _ = process_image_with_tracking(
        frame, model_choice, conf_threshold, image_source="webcam"
    )
    return annotated


def process_video(
    video_path: str,
    model_choice: str = "YOLOv11",
    conf_threshold: float = 0.25,
    progress=gr.Progress()
) -> str:
    """Process video with tracking."""
    if video_path is None:
        return None
    
    tracker.track_interaction("process_video", {
        "model": model_choice,
        "confidence": conf_threshold
    })
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = f"output_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_rgb, _ = process_image_with_tracking(
            frame_rgb, model_choice, conf_threshold, image_source="video"
        )
        
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
        
        frame_count += 1
        progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path


def get_session_summary() -> str:
    """Get formatted session summary."""
    summary = tracker.get_session_summary()
    
    if "error" in summary:
        return "No session data available yet. Start detecting to generate insights!"
    
    output = "## 📊 Session Statistics\n\n"
    output += f"**Session ID:** {summary['session_id']}\n"
    output += f"**Duration:** {summary['duration_seconds']:.0f} seconds\n"
    output += f"**Total Detections:** {summary['total_detections']}\n"
    output += f"**Frames Processed:** {summary['total_frames']}\n"
    output += f"**Avg Confidence:** {summary['avg_confidence']:.1%}\n"
    output += f"**Avg Inference Time:** {summary['avg_inference_time_ms']:.1f}ms\n\n"
    
    output += f"**Most Used Model:** {summary['most_used_model']}\n\n"
    
    if summary['top_signs']:
        output += "**Top Detected Signs:**\n"
        for item in summary['top_signs']:
            output += f"- {item['sign'].replace('_', ' ').title()}: {item['count']}x\n"
    
    return output


def get_llm_context() -> str:
    """Get LLM context for chat."""
    context = llm_handler.get_chat_context()
    return f"""**🤖 AI Assistant Context**

The Web-LLM chatbot has access to your session data and can provide intelligent insights.

**What you can ask:**
- Explain specific traffic signs
- Analyze your detection patterns
- Get recommendations for better accuracy
- Learn about traffic regulations

**Session Data Available:**
- Total detections: {context['session_data'].get('total_detections', 0)}
- Frames processed: {context['session_data'].get('total_frames', 0)}
- Models used: {', '.join(context['session_data'].get('models_usage', {}).keys())}

Open the **"🤖 AI Chat"** tab to interact with the assistant!
"""


# Custom CSS
CUSTOM_CSS = """
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.header-banner {
    background: var(--primary-gradient);
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.stat-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    transition: transform 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
}

.info-section {
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid #667eea;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
}
"""


# Build Gradio interface
with gr.Blocks(title="BD Traffic Sign Detection with AI", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
    <div class="header-banner">
        <h1>🚦 Bangladesh Traffic Sign Detection</h1>
        <p>AI-Powered Detection with Behavior Analysis & Web-LLM Assistant</p>
    </div>
    """)
    
    # Stats
    with gr.Row():
        gr.HTML(f'''
        <div class="stat-card">
            <div style="font-size: 2em;">⚡</div>
            <div style="font-size: 1.8em; font-weight: 700;">22.2</div>
            <div style="opacity: 0.8;">FPS (CPU)</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div style="font-size: 2em;">🎯</div>
            <div style="font-size: 1.8em; font-weight: 700;">99.45%</div>
            <div style="opacity: 0.8;">mAP@50</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div style="font-size: 2em;">📦</div>
            <div style="font-size: 1.8em; font-weight: 700;">{model_size:.1f}</div>
            <div style="opacity: 0.8;">MB Model</div>
        </div>
        ''')
        gr.HTML(f'''
        <div class="stat-card">
            <div style="font-size: 2em;">🏷️</div>
            <div style="font-size: 1.8em; font-weight: 700;">{len(class_names)}</div>
            <div style="opacity: 0.8;">Classes</div>
        </div>
        ''')
    
    gr.HTML("<br>")
    
    # Main tabs
    with gr.Tabs():
        
        # Image Detection
        with gr.Tab("🖼️ Image Detection"):
            gr.HTML('<div class="info-section">📌 Upload an image to detect traffic signs with AI analysis</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_model = gr.Radio(
                        choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                        value="YOLOv11",
                        label="Detection Model"
                    )
                    image_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    image_btn = gr.Button("🔍 Detect Signs", variant="primary")
                
                with gr.Column(scale=2):
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    
                with gr.Column(scale=2):
                    image_output = gr.Image(label="Detection Result", type="numpy")
                    image_stats = gr.Markdown(label="Statistics")
            
            image_btn.click(
                process_image_with_tracking,
                inputs=[image_input, image_model, image_conf],
                outputs=[image_output, image_stats]
            )
        
        # Webcam
        with gr.Tab("📹 Live Webcam"):
            gr.HTML('<div class="info-section">📌 Real-time detection from webcam</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_model = gr.Radio(
                        choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                        value="YOLOv11",
                        label="Model"
                    )
                    webcam_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Confidence"
                    )
                
                with gr.Column(scale=3):
                    with gr.Row():
                        webcam_input = gr.Image(
                            sources=["webcam"], 
                            streaming=True, 
                            type="numpy",
                            label="Camera"
                        )
                        webcam_output = gr.Image(
                            label="Detected",
                            type="numpy"
                        )
            
            webcam_input.stream(
                process_webcam,
                inputs=[webcam_input, webcam_model, webcam_conf],
                outputs=webcam_output
            )
        
        # Video Processing
        with gr.Tab("🎥 Video Processing"):
            gr.HTML('<div class="info-section">📌 Process video files with detection</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_model = gr.Radio(
                        choices=["YOLOv11", "Multi-Scale YOLOv11", "Ensemble"],
                        value="YOLOv11",
                        label="Model"
                    )
                    video_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Confidence"
                    )
                    video_btn = gr.Button("▶️ Process Video", variant="primary")
                
                with gr.Column(scale=2):
                    video_input = gr.Video(label="Upload Video")
                    
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Processed Video")
            
            video_btn.click(
                process_video,
                inputs=[video_input, video_model, video_conf],
                outputs=video_output
            )
        
        # Session Analytics
        with gr.Tab("📊 Session Analytics"):
            gr.HTML('<div class="info-section">📌 View your detection patterns and behavior analytics</div>')
            
            session_display = gr.Markdown(value=get_session_summary())
            refresh_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
            refresh_btn.click(get_session_summary, outputs=session_display)
            
            gr.HTML("<br>")
            llm_context_display = gr.Markdown(value=get_llm_context())
        
        # AI Chat Assistant
        with gr.Tab("🤖 AI Chat"):
            gr.HTML('<div class="info-section">📌 Chat with AI assistant powered by Web-LLM</div>')
            
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <p style="font-size: 1.1em; margin-bottom: 15px;">
                    The AI assistant runs entirely in your browser using Web-LLM technology.
                </p>
                <p style="opacity: 0.8;">
                    Click the button below to open the chat interface in a new window.
                </p>
            </div>
            """)
            
            # Read the HTML file and embed it
            chat_html_path = Path("assets/web_llm_chat.html")
            if chat_html_path.exists():
                with open(chat_html_path, 'r') as f:
                    chat_html = f.read()
                gr.HTML(chat_html)
            else:
                gr.HTML("""
                <div style="text-align: center; padding: 40px; background: rgba(245, 87, 108, 0.1); border-radius: 12px;">
                    <p style="font-size: 1.2em;">⚠️ Chat interface file not found</p>
                    <p>Please ensure assets/web_llm_chat.html exists</p>
                </div>
                """)
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 24px; opacity: 0.7; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 32px;">
        <p>🚀 Powered by YOLOv11 & Web-LLM | 🇧🇩 Bangladesh Traffic Sign Detection</p>
        <p style="font-size: 0.9em;">Behavior tracking enabled • Session ID: """ + session_id + """</p>
    </div>
    """)


# API endpoint for session data (for Web-LLM)
def get_session_data_api():
    """API endpoint for session data."""
    insights = tracker.get_behavior_insights()
    return insights


# Mount API
demo.load(fn=lambda: None, api_name="session-data")


if __name__ == "__main__":
    print(f"\n🌐 Launching enhanced web application...")
    print(f"📊 Behavior tracking: ENABLED")
    print(f"🤖 Web-LLM integration: ENABLED")
    print(f"🔑 Session ID: {session_id}\n")
    
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    finally:
        # End session on shutdown
        print("\n🔄 Shutting down...")
        tracker.end_session()
        print("✅ Session data saved")
