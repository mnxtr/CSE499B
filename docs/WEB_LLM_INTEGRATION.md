# Web-LLM Integration for User Behavior Analysis

## Overview

This feature integrates Web-LLM technology into the Bangladesh Traffic Sign Detection system to provide intelligent user behavior analysis and an AI-powered chat assistant that runs entirely in the browser.

## Features

### 🤖 Browser-Based AI Assistant
- **Web-LLM Powered**: Uses Llama-3.2-1B-Instruct model running completely in your browser
- **No Server Required**: Model inference happens client-side using WebGPU
- **Privacy-First**: Your data never leaves your device

### 📊 User Behavior Tracking
- **Session Analytics**: Track detections, models used, and interaction patterns
- **Real-time Insights**: Get instant analysis of your detection behavior
- **Pattern Recognition**: Identify most commonly detected signs and usage patterns

### 💬 Intelligent Chat Interface
- **Contextual Responses**: AI has access to your session data
- **Traffic Sign Information**: Ask about any detected sign
- **Behavior Analysis**: Get personalized insights and recommendations
- **Natural Language**: Chat naturally about your detection results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Gradio)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Image   │  │  Webcam  │  │  Video   │  │  Analytics  │ │
│  │ Detection│  │  Stream  │  │ Process  │  │  Dashboard  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Behavior Tracking & Analysis Layer              │
│  ┌─────────────────────┐      ┌─────────────────────────┐  │
│  │  BehaviorTracker    │◄────►│   WebLLMHandler         │  │
│  │  - Session data     │      │   - Context builder     │  │
│  │  - Event logging    │      │   - Prompt generator    │  │
│  │  - Analytics        │      │   - Sign database       │  │
│  └─────────────────────┘      └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Server (Flask)                      │
│  Endpoints: /api/session-data, /api/chat-context, etc.     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Browser-Based Web-LLM (JavaScript)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Web-LLM Engine (Llama-3.2-1B-Instruct)              │  │
│  │  - Runs in browser using WebGPU                      │  │
│  │  - Model cached locally                              │  │
│  │  - Inference on client device                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

1. **Python Requirements**:
```bash
pip install -r requirements.txt
```

The key additions are:
- `flask>=3.0.0` - API server
- `flask-cors>=4.0.0` - CORS support for browser access

2. **Browser Requirements**:
- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- Minimum 4GB RAM available
- ~2GB free disk space for model cache

### Verify Installation

```bash
# Test behavior tracking
python -c "from utils.behavior_tracker import get_tracker; print('✅ Behavior tracking OK')"

# Test LLM handler
python -c "from utils.web_llm_handler import get_llm_handler; print('✅ LLM handler OK')"
```

## Usage

### Quick Start

1. **Launch the Enhanced Web App**:
```bash
python web_app_llm.py
```

2. **Access the Interface**:
   - Open browser to `http://localhost:7860`
   - Navigate to the "🤖 AI Chat" tab
   - Click "Load AI Model" (first load takes 1-2 minutes)

3. **Start Detecting**:
   - Upload images or use webcam
   - Detection events are automatically tracked
   - View analytics in "📊 Session Analytics" tab

4. **Chat with AI**:
   - Ask questions about detected signs
   - Request behavior analysis
   - Get recommendations

### Running with API Server

For separate API server (optional):

```bash
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Start web app
python web_app_llm.py
```

## Components

### 1. BehaviorTracker (`utils/behavior_tracker.py`)

Tracks user interactions and detection events:

```python
from utils.behavior_tracker import get_tracker

tracker = get_tracker()
tracker.start_session()

# Track a detection
tracker.track_detection(
    detection_count=5,
    classes_detected=['stop_sign', 'speed_limit_40'],
    avg_confidence=0.92,
    inference_time_ms=45.2,
    model_used='YOLOv11',
    image_source='upload'
)

# Get session summary
summary = tracker.get_session_summary()
```

**Key Features**:
- Thread-safe implementation
- Persistent storage (JSON)
- Real-time analytics
- Session management

### 2. WebLLMHandler (`utils/web_llm_handler.py`)

Prepares data and prompts for the LLM:

```python
from utils.web_llm_handler import get_llm_handler

handler = get_llm_handler()

# Get chat context
context = handler.get_chat_context()

# Generate insight prompt
prompt = handler.generate_insight_prompt("What are the most common signs I detect?")

# Get sign information
info = handler.get_sign_explanation('stop_sign')
```

**Key Features**:
- Traffic sign database
- Context-aware prompts
- Session data formatting
- Sign explanations

### 3. Web-LLM Chat Interface (`assets/web_llm_chat.html`)

Browser-based chat interface using Web-LLM:

**Features**:
- Llama-3.2-1B-Instruct model
- Client-side inference
- Session data integration
- Real-time chat

### 4. API Server (`api_server.py`)

Flask-based REST API for session data access:

**Endpoints**:
- `GET /api/session-data` - Full session insights
- `GET /api/session-summary` - Summary statistics
- `GET /api/chat-context` - LLM context data
- `GET /api/sign-info/<name>` - Traffic sign information
- `GET /api/health` - Health check

## Session Data Structure

### DetectionEvent
```json
{
  "timestamp": "2026-02-14T13:30:45.123456",
  "detection_count": 3,
  "classes_detected": ["stop_sign", "speed_limit_40"],
  "avg_confidence": 0.94,
  "inference_time_ms": 42.5,
  "model_used": "YOLOv11",
  "image_source": "upload"
}
```

### SessionData
```json
{
  "session_id": "session_20260214_133045",
  "start_time": "2026-02-14T13:30:45",
  "total_detections": 25,
  "total_frames": 10,
  "avg_confidence": 0.89,
  "avg_inference_time": 45.2,
  "models_used": {
    "YOLOv11": 7,
    "Ensemble": 3
  },
  "common_signs": {
    "stop_sign": 12,
    "speed_limit_40": 8,
    "no_entry": 5
  }
}
```

## Example Use Cases

### 1. Behavior Analysis

```python
# User uploads 10 images
# System tracks: 
# - 8 used YOLOv11, 2 used Ensemble
# - Most detected: stop_sign (15x)
# - Avg confidence: 92%

# LLM Analysis:
"""
Based on your session:
🎯 You prefer YOLOv11 (80% usage) - great choice for speed!
🚦 Stop signs are your most common detection (60% of signs)
✨ High confidence (92%) indicates good image quality
💡 Recommendation: Try Ensemble mode for challenging images
"""
```

### 2. Traffic Sign Education

```
User: "What does the stop sign mean?"

AI: "**Stop Sign** 🛑
- **Complete stop required** at intersection
- **Shape:** Octagon (8-sided)
- **Color:** Red with white text
- **Importance:** Critical - failure to stop can cause accidents
- In Bangladesh: Must stop completely, check all directions before proceeding"
```

### 3. Performance Optimization

```
User: "How can I improve detection accuracy?"

AI: "Based on your session data:
📊 Current avg confidence: 78%
🔍 Suggestions:
1. Increase lighting - 40% of low-conf detections in dim images
2. Try Multi-Scale YOLOv11 for distant signs
3. Adjust confidence threshold to 0.3 for better recall
4. Ensure signs are centered in frame"
```

## Configuration

### Tracker Storage Path
```python
tracker = BehaviorTracker(storage_path="data/analytics/sessions.json")
```

### LLM Model Selection

Edit `assets/web_llm_chat.html`:
```javascript
engine = await webllm.CreateMLCEngine(
    "Llama-3.2-1B-Instruct-q4f16_1-MLC",  // Change model here
    { initProgressCallback: ... }
);
```

**Available Models**:
- `Llama-3.2-1B-Instruct-q4f16_1-MLC` (Fast, 1B params)
- `Llama-3.2-3B-Instruct-q4f16_1-MLC` (Better, 3B params)
- `Phi-3.5-mini-instruct-q4f16_1-MLC` (Compact, 3.8B params)

### API Server Configuration

```python
# In api_server.py
if __name__ == "__main__":
    run_api_server(host='0.0.0.0', port=5000)  # Change port here
```

## Browser Compatibility

### Supported Browsers
✅ Chrome 113+ (Windows, macOS, Linux)
✅ Edge 113+ (Windows, macOS)
✅ Opera 99+

### Not Supported
❌ Firefox (WebGPU experimental)
❌ Safari (WebGPU in development)
❌ Mobile browsers (insufficient memory)

### Check WebGPU Support
Visit: `chrome://gpu` and look for "WebGPU: Hardware accelerated"

## Performance

### Model Loading
- **First load**: 1-2 minutes (downloads ~1GB model)
- **Subsequent loads**: 5-10 seconds (cached)

### Inference Speed
- **1B model**: 10-30 tokens/sec (depending on hardware)
- **3B model**: 5-15 tokens/sec

### Memory Usage
- **Browser**: 2-4GB RAM
- **Python app**: 500MB-1GB RAM

## Troubleshooting

### Web-LLM Not Loading
1. Check WebGPU support: `chrome://gpu`
2. Clear browser cache
3. Try incognito mode
4. Update to latest Chrome/Edge

### API Connection Errors
1. Verify API server is running: `curl http://localhost:5000/api/health`
2. Check CORS settings in browser console
3. Ensure ports 5000 and 7860 are not blocked

### Session Data Not Saving
1. Check write permissions: `data/analytics/`
2. Verify disk space available
3. Check logs for JSON serialization errors

### Low Inference Speed
1. Close other browser tabs
2. Use smaller model (1B instead of 3B)
3. Ensure GPU drivers are updated
4. Check system resources in Task Manager

## Security Considerations

### Privacy
- ✅ All LLM inference happens in browser
- ✅ No data sent to external servers
- ✅ Model cached locally
- ⚠️ Session data stored in `data/analytics/` - ensure proper permissions

### API Security
- For production, add authentication:
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@app.route('/api/session-data')
@auth.login_required
def get_session_data():
    ...
```

### CORS Configuration
```python
# Restrict origins in production
CORS(app, origins=["http://localhost:7860"])
```

## Future Enhancements

### Planned Features
- [ ] Export session reports (PDF/CSV)
- [ ] Comparison across multiple sessions
- [ ] Advanced behavior patterns (time-series analysis)
- [ ] Voice input for chat
- [ ] Offline mode with IndexedDB storage
- [ ] Multi-language support

### Community Contributions
We welcome contributions! Areas for improvement:
- Additional LLM models
- Enhanced analytics visualizations
- Mobile browser support
- Alternative storage backends

## Citation

If you use this Web-LLM integration in your research, please cite:

```bibtex
@software{bdtrafficsigns_webllm2024,
  title={Web-LLM Integration for Traffic Sign Detection Behavior Analysis},
  author={BD Traffic Signs Research Team},
  year={2024},
  url={https://github.com/your-username/bd-traffic-signs}
}
```

## License

MIT License - See LICENSE file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/bd-traffic-signs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/bd-traffic-signs/discussions)
- **Email**: research@bdtrafficsigns.org

---

**Made with ❤️ for the Bangladeshi AI Community**

🚦 Traffic Sign Detection | 🤖 Web-LLM | 📊 Behavior Analysis
