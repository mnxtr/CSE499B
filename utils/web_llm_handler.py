#!/usr/bin/env python3
"""
Web-LLM Handler for User Behavior Analysis

Provides backend support for Web-LLM integration, preparing
data and insights for the browser-based LLM to analyze.
"""

import json
from typing import Dict, List, Any, Optional
from .behavior_tracker import get_tracker


class WebLLMHandler:
    """
    Handler for Web-LLM integration with behavior analysis.
    Prepares prompts and data for the browser-based LLM.
    """
    
    def __init__(self):
        self.tracker = get_tracker()
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are an AI assistant integrated into a Bangladeshi Traffic Sign Detection System. 
Your role is to help users understand their detection results, analyze their behavior patterns, 
and provide insights about traffic signs detected in images and videos.

Key capabilities:
1. Explain detected traffic signs and their meanings
2. Analyze user behavior patterns (which signs they detect most, preferred models, etc.)
3. Provide recommendations for improving detection accuracy
4. Answer questions about the detection system and traffic regulations
5. Generate summaries of detection sessions

When answering:
- Be concise and helpful
- Use emojis to make responses engaging (🚦, 🎯, ⚡, etc.)
- Focus on actionable insights
- Reference specific data from the session when available
"""
    
    def get_chat_context(self) -> Dict[str, Any]:
        """Get context for chat including session data."""
        insights = self.tracker.get_behavior_insights()
        summary = self.tracker.get_session_summary()
        
        return {
            "system_prompt": self.system_prompt,
            "session_data": summary,
            "behavior_insights": insights,
            "traffic_sign_info": self._get_traffic_sign_database()
        }
    
    def _get_traffic_sign_database(self) -> Dict[str, Any]:
        """Get traffic sign information database for reference."""
        return {
            "regulatory": {
                "stop_sign": {
                    "name": "Stop Sign",
                    "description": "Complete stop required at intersection",
                    "shape": "Octagon",
                    "color": "Red",
                    "importance": "Critical"
                },
                "no_entry": {
                    "name": "No Entry",
                    "description": "Entry prohibited for all vehicles",
                    "shape": "Circle",
                    "color": "Red with white bar",
                    "importance": "Critical"
                },
                "no_parking": {
                    "name": "No Parking",
                    "description": "Parking prohibited in this area",
                    "shape": "Circle",
                    "color": "Red and blue",
                    "importance": "Important"
                },
                "one_way": {
                    "name": "One Way",
                    "description": "Traffic flows in one direction only",
                    "shape": "Rectangle",
                    "color": "Blue with white arrow",
                    "importance": "Important"
                },
                "speed_limit_40": {
                    "name": "Speed Limit 40",
                    "description": "Maximum speed 40 km/h",
                    "shape": "Circle",
                    "color": "Red circle with white background",
                    "importance": "Important"
                },
                "speed_limit_60": {
                    "name": "Speed Limit 60",
                    "description": "Maximum speed 60 km/h",
                    "shape": "Circle",
                    "color": "Red circle with white background",
                    "importance": "Important"
                }
            },
            "warning": {
                "pedestrian_crossing": {
                    "name": "Pedestrian Crossing",
                    "description": "Watch for pedestrians crossing",
                    "shape": "Triangle",
                    "color": "Red border, white background",
                    "importance": "High"
                },
                "school_zone": {
                    "name": "School Zone",
                    "description": "Children may be crossing",
                    "shape": "Triangle",
                    "color": "Red border, white background",
                    "importance": "High"
                },
                "curve_ahead": {
                    "name": "Curve Ahead",
                    "description": "Road curves ahead, reduce speed",
                    "shape": "Triangle",
                    "color": "Red border, white background",
                    "importance": "Important"
                }
            },
            "mandatory": {
                "roundabout": {
                    "name": "Roundabout",
                    "description": "Enter roundabout, give way to traffic",
                    "shape": "Circle",
                    "color": "Blue",
                    "importance": "Important"
                },
                "keep_left": {
                    "name": "Keep Left",
                    "description": "Keep to the left side",
                    "shape": "Circle",
                    "color": "Blue with white arrow",
                    "importance": "Important"
                },
                "keep_right": {
                    "name": "Keep Right",
                    "description": "Keep to the right side",
                    "shape": "Circle",
                    "color": "Blue with white arrow",
                    "importance": "Important"
                }
            }
        }
    
    def generate_session_prompt(self) -> str:
        """Generate a prompt about the current session for LLM."""
        summary = self.tracker.get_session_summary()
        
        if "error" in summary:
            return "No active session data available yet. Start detecting traffic signs to begin tracking!"
        
        prompt = f"""Current Session Analysis:

**Session Overview:**
- Duration: {summary['duration_seconds']:.0f} seconds
- Total Detections: {summary['total_detections']}
- Frames Processed: {summary['total_frames']}
- Average Confidence: {summary['avg_confidence']:.1%}
- Average Inference Time: {summary['avg_inference_time_ms']:.1f}ms

**Most Used Model:** {summary['most_used_model']}

**Top Detected Signs:**
"""
        
        for item in summary['top_signs']:
            prompt += f"\n- {item['sign']}: {item['count']} times"
        
        return prompt
    
    def generate_insight_prompt(self, user_question: str = None) -> str:
        """Generate a prompt for insight generation."""
        insights = self.tracker.get_behavior_insights()
        
        if "error" in insights:
            return "No behavior data available yet."
        
        summary = insights["summary"]
        
        base_prompt = f"""Please analyze this user's traffic sign detection session:

Session Stats:
- Total detections: {summary['total_detections']}
- Frames processed: {summary['total_frames']}
- Average confidence: {summary['avg_confidence']:.1%}
- Most used model: {summary['most_used_model']}

Detection Sources:
{json.dumps(insights['source_distribution'], indent=2)}

Top Signs Detected:
{json.dumps([{'sign': s['sign'], 'count': s['count']} for s in summary['top_signs']], indent=2)}

"""
        
        if user_question:
            base_prompt += f"\nUser Question: {user_question}\n"
        else:
            base_prompt += "\nProvide insights and recommendations based on this data.\n"
        
        return base_prompt
    
    def get_sign_explanation(self, sign_name: str) -> str:
        """Get explanation for a specific traffic sign."""
        db = self._get_traffic_sign_database()
        
        # Search through all categories
        for category, signs in db.items():
            if sign_name in signs:
                info = signs[sign_name]
                return f"""**{info['name']}** 🚦

**Category:** {category.title()}
**Description:** {info['description']}
**Shape:** {info['shape']}
**Color:** {info['color']}
**Importance:** {info['importance']}
"""
        
        return f"Information not available for sign: {sign_name}"
    
    def format_detection_result(
        self,
        detection_count: int,
        classes: List[str],
        confidence: float,
        model: str
    ) -> str:
        """Format detection result for display."""
        if detection_count == 0:
            return "No traffic signs detected in this image. Try with a different image or adjust the confidence threshold."
        
        result = f"**🎯 Detected {detection_count} traffic sign(s)** using {model}:\n\n"
        
        for sign in set(classes):
            count = classes.count(sign)
            result += f"- {sign.replace('_', ' ').title()} ({count}x)\n"
        
        result += f"\n**Average Confidence:** {confidence:.1%}\n"
        
        return result


# Global handler instance
_global_handler = None


def get_llm_handler() -> WebLLMHandler:
    """Get or create global LLM handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = WebLLMHandler()
    return _global_handler
