#!/usr/bin/env python3
"""
API Server for Web-LLM Integration

Provides RESTful API endpoints for session data access by the Web-LLM client.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from utils.behavior_tracker import get_tracker
from utils.web_llm_handler import get_llm_handler

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for browser access

# Get global instances
tracker = get_tracker()
llm_handler = get_llm_handler()


@app.route('/api/session-data', methods=['GET'])
def get_session_data():
    """Get current session data and insights."""
    try:
        insights = tracker.get_behavior_insights()
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/session-summary', methods=['GET'])
def get_session_summary():
    """Get session summary."""
    try:
        summary = tracker.get_session_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat-context', methods=['GET'])
def get_chat_context():
    """Get context for LLM chat."""
    try:
        context = llm_handler.get_chat_context()
        # Convert to JSON-serializable format
        return jsonify({
            "system_prompt": context["system_prompt"],
            "session_data": context["session_data"],
            "behavior_insights": context["behavior_insights"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sign-info/<sign_name>', methods=['GET'])
def get_sign_info(sign_name):
    """Get information about a specific traffic sign."""
    try:
        info = llm_handler.get_sign_explanation(sign_name)
        return jsonify({"sign_name": sign_name, "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "web-llm-api",
        "version": "1.0.0"
    })


def run_api_server(host='0.0.0.0', port=5000):
    """Run the API server."""
    print(f"\n🚀 Starting Web-LLM API server on {host}:{port}")
    print(f"📡 Endpoints available:")
    print(f"   - GET  /api/session-data    - Get session insights")
    print(f"   - GET  /api/session-summary - Get session summary")
    print(f"   - GET  /api/chat-context    - Get chat context")
    print(f"   - GET  /api/sign-info/<name> - Get sign information")
    print(f"   - GET  /api/health          - Health check\n")
    
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_api_server()
