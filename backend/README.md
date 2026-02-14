# Backend Module

This module provides backend functionality for the Bangladesh Traffic Sign Detection System, including:

## Features

### 1. User Behavior Analytics (`database.py`)
- **SQLite Database**: Stores user sessions, detection events, interactions, and chat messages
- **Session Management**: Tracks user sessions with unique IDs
- **Event Logging**: Records all detection events with timestamps, model types, and performance metrics
- **Analytics Dashboard**: Provides summary statistics and insights

#### Database Schema:
- `sessions`: User session information and aggregated statistics
- `detection_events`: Individual detection events with performance metrics
- `user_interactions`: User interactions like button clicks, model changes
- `chat_messages`: Chat history between users and AI assistant

### 2. LLM Chatbot (`llm_chatbot.py`)
- **Knowledge-Based AI Assistant**: Answers questions about traffic signs and system usage
- **Context-Aware Responses**: Uses detection results to provide relevant information
- **Traffic Sign Information**: Provides meanings, categories, and required actions
- **No External API**: Runs entirely locally without external dependencies

#### Chatbot Capabilities:
- Explain detection results in detail
- Provide information about specific traffic signs
- Answer questions about model performance
- Guide users on how to use the system
- Classify traffic signs by category (Regulatory, Warning, Mandatory)

## Usage

### Analytics Database

```python
from backend.database import AnalyticsDB

# Initialize database
db = AnalyticsDB(db_path="analytics.db")

# Create a new session
session_id = "unique-session-id"
db.create_session(session_id)

# Log a detection event
db.log_detection(
    session_id=session_id,
    model_type="YOLOv11",
    confidence_threshold=0.25,
    num_detections=5,
    inference_time_ms=45.2,
    fps=22.1,
    input_type="image"
)

# Log user interaction
db.log_interaction(
    session_id=session_id,
    interaction_type="model_change",
    details={"model": "Ensemble"}
)

# Get analytics summary
summary = db.get_analytics_summary()
print(f"Total sessions: {summary['total_sessions']}")
print(f"Total detections: {summary['total_detections']}")

# Close connection
db.close()
```

### LLM Chatbot

```python
from backend.llm_chatbot import TrafficSignLLM

# Initialize chatbot
chatbot = TrafficSignLLM()

# Chat without context
response = chatbot.chat("How do I use this system?")
print(response)

# Chat with detection context
detection_context = {
    "num_detections": 2,
    "detected_classes": ["stop_sign", "speed_limit_40"],
    "model_type": "YOLOv11",
    "confidence": 0.25
}
response = chatbot.chat("What signs did you detect?", detection_context)
print(response)

# Get conversation history
history = chatbot.get_conversation_history()
```

## Integration

The backend module is integrated into `web_app_llm.py`, which provides:
- Real-time analytics tracking during detection
- AI chatbot for user assistance
- Analytics dashboard showing usage statistics
- Session management

## Data Privacy

All analytics data is stored locally in SQLite database:
- No external API calls
- No personal information collected
- Session IDs are randomly generated UUIDs
- Data never leaves your system

## Dependencies

- `sqlite3` (built-in with Python)
- No external dependencies required

## Files

- `database.py` - Analytics database implementation
- `llm_chatbot.py` - LLM chatbot implementation
- `__init__.py` - Module initialization
- `README.md` - This documentation
