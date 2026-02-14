# Quick Start Guide: Web-LLM for User Behavior Analysis

## 🚀 Getting Started in 5 Minutes

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Modern browser with WebGPU support
# - Chrome 113+ or Edge 113+
# - Check: chrome://gpu (look for "WebGPU: Hardware accelerated")
```

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /path/to/CSE499B

# Install Python packages
pip install -r requirements.txt
```

### Step 2: Test Installation

```bash
# Run the test script
python test_web_llm.py

# Expected output:
# ✅ All tests completed successfully!
```

### Step 3: Launch Web Application

```bash
# Option A: Launch enhanced app with Web-LLM
python web_app_llm.py

# Option B: Launch with separate API server (recommended for production)
# Terminal 1:
python api_server.py

# Terminal 2:
python web_app_llm.py
```

### Step 4: Use the Application

1. **Open Browser**: Navigate to `http://localhost:7860`

2. **Detect Signs**:
   - Go to "🖼️ Image Detection" tab
   - Upload a traffic sign image
   - Click "🔍 Detect Signs"
   - View results and statistics

3. **Enable AI Assistant**:
   - Go to "🤖 AI Chat" tab
   - Click "🚀 Load AI Model" (first time: 1-2 minutes)
   - Wait for: "✅ Web-LLM ready!"

4. **Chat with AI**:
   ```
   You: What does the stop sign mean?
   AI: **Stop Sign** 🛑
       Complete stop required at intersection...
   
   You: What patterns do you see in my detections?
   AI: Based on your session...
   ```

5. **View Analytics**:
   - Go to "📊 Session Analytics" tab
   - See detection patterns and behavior insights
   - Click "🔄 Refresh Statistics" for updates

## 📝 Example Session

### Scenario: Detecting Traffic Signs

```python
# Session automatically starts when app launches
# Session ID: session_20260214_133000

# User Actions:
1. Upload image → Detect 3 signs (stop_sign, speed_limit_40, no_entry)
2. Switch to Ensemble model
3. Upload video → Process 100 frames
4. View analytics

# System Tracks:
- Total detections: 85
- Frames processed: 101
- Avg confidence: 89.5%
- Most used model: YOLOv11
- Top sign: stop_sign (32x)

# AI Insights:
"You show a preference for YOLOv11 (92% usage) - great for speed!
 Stop signs are your most common detection (38%).
 High avg confidence suggests good image quality.
 Recommendation: Try Multi-Scale for distant signs."
```

## 🤖 Chat Examples

### Ask About Traffic Signs
```
Q: What is a pedestrian crossing sign?
A: **Pedestrian Crossing** 🚶
   Category: Warning
   Description: Watch for pedestrians crossing
   Shape: Triangle
   Color: Red border, white background
   Importance: High
```

### Request Analysis
```
Q: How can I improve detection accuracy?
A: Based on your 78% avg confidence:
   1. ✅ Improve lighting (40% of low-conf in dim images)
   2. ✅ Try Multi-Scale for distant signs
   3. ✅ Ensure signs are centered in frame
```

### Get Session Summary
```
Click "📊 Get Session Insights" button

AI: Your detection session analysis:
    - Duration: 15 minutes
    - 127 signs detected across 45 frames
    - Most detected: stop_sign (48 times)
    - Preferred model: YOLOv11 (87% usage)
    - Recommendation: Explore Ensemble for complex scenes
```

## 🔧 Configuration

### Change LLM Model

Edit `assets/web_llm_chat.html`:

```javascript
// Line 20: Change model
engine = await webllm.CreateMLCEngine(
    "Llama-3.2-1B-Instruct-q4f16_1-MLC",  // Fast (1B params)
    // OR
    "Llama-3.2-3B-Instruct-q4f16_1-MLC",  // Better (3B params)
    // OR  
    "Phi-3.5-mini-instruct-q4f16_1-MLC",  // Compact (3.8B params)
    { initProgressCallback: ... }
);
```

### Change Storage Path

Edit `utils/behavior_tracker.py`:

```python
# Line 56
def __init__(self, storage_path: str = "data/analytics/sessions.json"):
    # Change to your preferred path
```

### API Server Port

Edit `api_server.py`:

```python
# Last line
if __name__ == "__main__":
    run_api_server(host='0.0.0.0', port=5000)  # Change port
```

## 📊 Understanding Session Data

### Session Summary Format
```json
{
  "session_id": "session_20260214_133000",
  "duration_seconds": 900,
  "total_detections": 127,
  "total_frames": 45,
  "avg_confidence": 0.89,
  "avg_inference_time_ms": 47.3,
  "most_used_model": "YOLOv11",
  "top_signs": [
    {"sign": "stop_sign", "count": 48},
    {"sign": "speed_limit_40", "count": 32}
  ]
}
```

### Viewing Raw Data

```bash
# Check saved sessions
cat data/analytics/sessions.json | python -m json.tool

# Or in Python:
python -c "
import json
with open('data/analytics/sessions.json') as f:
    data = json.load(f)
    print(f'Total sessions: {len(data)}')
    print(f'Latest: {data[-1][\"session_id\"]}')
"
```

## 🐛 Troubleshooting

### Issue: "Model failed to load"

**Solution:**
```bash
# 1. Check WebGPU support
chrome://gpu

# 2. Try incognito mode
# 3. Clear browser cache
# 4. Update to latest Chrome/Edge
```

### Issue: "API connection failed"

**Solution:**
```bash
# Check API server is running
curl http://localhost:5000/api/health

# Should return: {"status": "healthy", ...}

# If not running:
python api_server.py
```

### Issue: "No session data"

**Solution:**
```bash
# 1. Ensure you've made at least one detection
# 2. Check directory exists
mkdir -p data/analytics

# 3. Check permissions
ls -la data/analytics/

# 4. Test manually
python -c "
from utils.behavior_tracker import get_tracker
tracker = get_tracker()
tracker.start_session()
tracker.track_detection(1, ['test'], 0.9, 50, 'YOLOv11', 'upload')
print('Test detection tracked')
tracker.end_session()
"
```

### Issue: Slow inference

**Solution:**
```bash
# 1. Use smaller model (1B instead of 3B)
# 2. Close other browser tabs
# 3. Check system resources (Task Manager)
# 4. Ensure WebGPU is hardware accelerated
```

## 📚 Additional Resources

- **Full Documentation**: [docs/WEB_LLM_INTEGRATION.md](docs/WEB_LLM_INTEGRATION.md)
- **Project README**: [README.md](README.md)
- **API Reference**: See `api_server.py` for endpoint details
- **Web-LLM Docs**: https://mlc.ai/web-llm/

## 🎯 Next Steps

1. **Try Different Models**: Experiment with 1B vs 3B models
2. **Explore Analytics**: View detection patterns over multiple sessions
3. **Customize Prompts**: Edit `WebLLMHandler` to add domain-specific knowledge
4. **Export Data**: Write scripts to export session data for analysis
5. **Build Features**: Add new chat commands or analytics visualizations

## 💡 Tips for Best Results

1. **Image Quality**: Use clear, well-lit images for best detection
2. **Model Selection**: 
   - YOLOv11: Fast, good for real-time
   - Multi-Scale: Better for varied sign sizes
   - Ensemble: Best accuracy, slower
3. **Confidence Threshold**: Lower (0.25) for recall, higher (0.5) for precision
4. **Chat Prompts**: Be specific - reference your session data
5. **Browser**: Use hardware acceleration for faster LLM inference

## 🚀 Ready to Go!

You're all set! Start detecting traffic signs and chatting with your AI assistant.

**Questions?** Check the [full documentation](docs/WEB_LLM_INTEGRATION.md) or [open an issue](https://github.com/mnxtr/CSE499B/issues).

---

**Happy Detecting! 🚦🤖**
