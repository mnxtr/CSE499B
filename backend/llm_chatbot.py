"""
Web-LLM Chatbot for Traffic Sign Detection System

This module provides an AI assistant that can answer questions about:
- Traffic sign detection results
- How to use the system
- Bangladesh traffic signs and their meanings
- Model performance and accuracy
"""
import json
from typing import List, Dict, Tuple, Optional


class TrafficSignLLM:
    """
    Local LLM-style chatbot for traffic sign detection assistance.
    Uses a knowledge base approach instead of external API calls.
    """
    
    def __init__(self):
        """Initialize the chatbot with traffic sign knowledge base"""
        self.conversation_history = []
        self.knowledge_base = self._build_knowledge_base()
        self.current_detection_context = None
    
    def _build_knowledge_base(self) -> Dict:
        """Build knowledge base about Bangladesh traffic signs and the system"""
        return {
            # System capabilities
            "system": {
                "model": "YOLOv11 Nano",
                "accuracy": "99.45% mAP@50",
                "speed": "22.2 FPS on CPU",
                "model_size": "5.2 MB",
                "classes": 29,
                "features": [
                    "Real-time webcam detection",
                    "Video file processing",
                    "Image upload detection",
                    "Multi-scale detection",
                    "Ensemble model inference"
                ]
            },
            
            # Traffic sign categories and meanings
            "traffic_signs": {
                "stop_sign": {
                    "category": "Regulatory",
                    "meaning": "Complete stop required at intersection",
                    "action": "Stop completely before proceeding"
                },
                "speed_limit_40": {
                    "category": "Regulatory",
                    "meaning": "Maximum speed limit is 40 km/h",
                    "action": "Reduce speed to 40 km/h or below"
                },
                "speed_limit_60": {
                    "category": "Regulatory",
                    "meaning": "Maximum speed limit is 60 km/h",
                    "action": "Reduce speed to 60 km/h or below"
                },
                "no_entry": {
                    "category": "Regulatory",
                    "meaning": "Entry prohibited for all vehicles",
                    "action": "Do not enter this road or area"
                },
                "no_parking": {
                    "category": "Regulatory",
                    "meaning": "Parking is not allowed in this area",
                    "action": "Do not park your vehicle here"
                },
                "one_way": {
                    "category": "Regulatory",
                    "meaning": "Traffic flows in one direction only",
                    "action": "Follow the indicated direction"
                },
                "pedestrian_crossing": {
                    "category": "Warning",
                    "meaning": "Pedestrian crossing ahead",
                    "action": "Slow down and yield to pedestrians"
                },
                "school_zone": {
                    "category": "Warning",
                    "meaning": "School zone ahead - children may be present",
                    "action": "Reduce speed and drive cautiously"
                },
                "roundabout": {
                    "category": "Mandatory",
                    "meaning": "Circular intersection ahead",
                    "action": "Yield to traffic already in roundabout, proceed counterclockwise"
                },
                "keep_left": {
                    "category": "Mandatory",
                    "meaning": "Keep to the left side of the road",
                    "action": "Stay in the left lane"
                },
                "keep_right": {
                    "category": "Mandatory",
                    "meaning": "Keep to the right side of obstruction",
                    "action": "Pass on the right side"
                }
            },
            
            # Common questions and answers
            "faq": {
                "how_to_use": "You can use this system in three ways:\n1. Live Webcam: Click 'Start streaming' for real-time detection\n2. Image Upload: Upload a traffic sign photo\n3. Video Processing: Upload a video file for frame-by-frame detection\n\nAdjust the confidence threshold to control detection sensitivity.",
                
                "model_selection": "Three detection modes are available:\n- YOLOv11: Fast, standard detection (recommended)\n- Multi-Scale: Better for small signs (slower)\n- Ensemble: Highest accuracy by combining multiple scales (slowest)",
                
                "accuracy": "Our YOLOv11 model achieves 99.45% mAP@50 accuracy on Bangladesh traffic signs. It was trained on 8,953 images with 29 different sign classes.",
                
                "speed": "The system runs at 22.2 FPS on CPU and can achieve 200+ FPS on GPU. Detection takes approximately 45ms per frame.",
                
                "confidence_threshold": "The confidence threshold (0.1-0.9) filters detections. Lower values show more signs but may include false positives. Higher values show only very confident detections. We recommend 0.25 for balanced results.",
                
                "supported_signs": "The system detects 29 traffic sign classes including:\n- Regulatory signs: stop, speed limits, no entry, no parking, one way\n- Warning signs: pedestrian crossing, school zone, curves, intersections\n- Mandatory signs: roundabout, keep left/right, bicycle path"
            }
        }
    
    def set_detection_context(self, num_detections: int, detected_classes: List[str], 
                             model_type: str, confidence: float):
        """Set context from latest detection for context-aware responses"""
        self.current_detection_context = {
            "num_detections": num_detections,
            "detected_classes": detected_classes,
            "model_type": model_type,
            "confidence": confidence
        }
    
    def _match_intent(self, user_message: str) -> str:
        """Simple intent matching based on keywords"""
        message_lower = user_message.lower()
        
        # Detection result queries
        if self.current_detection_context and any(word in message_lower for word in 
            ["what", "detect", "found", "see", "result", "this sign", "mean"]):
            return "detection_result"
        
        # System usage
        if any(word in message_lower for word in ["how", "use", "start", "upload", "stream"]):
            return "how_to_use"
        
        # Model questions
        if any(word in message_lower for word in ["model", "which model", "mode", "difference"]):
            return "model_selection"
        
        # Accuracy/performance
        if any(word in message_lower for word in ["accuracy", "accurate", "performance", "fps", "speed"]):
            return "performance"
        
        # Confidence threshold
        if any(word in message_lower for word in ["confidence", "threshold", "sensitivity"]):
            return "confidence_threshold"
        
        # Supported signs
        if any(word in message_lower for word in ["support", "classes", "types", "signs detect"]):
            return "supported_signs"
        
        # Specific sign meaning
        for sign_name in self.knowledge_base["traffic_signs"].keys():
            if sign_name.replace("_", " ") in message_lower:
                return f"sign_meaning:{sign_name}"
        
        # General traffic sign categories
        if any(word in message_lower for word in ["regulatory", "warning", "mandatory", "category"]):
            return "sign_categories"
        
        return "general"
    
    def _generate_response(self, intent: str, user_message: str) -> str:
        """Generate response based on intent"""
        
        # Handle detection result queries
        if intent == "detection_result" and self.current_detection_context:
            ctx = self.current_detection_context
            if ctx["num_detections"] == 0:
                return "No traffic signs were detected in your image. Try adjusting the confidence threshold or ensuring the image clearly shows traffic signs."
            
            response = f"I detected {ctx['num_detections']} traffic sign(s) using the {ctx['model_type']} model:\n\n"
            for sign_class in ctx['detected_classes']:
                sign_info = self.knowledge_base["traffic_signs"].get(sign_class, {})
                if sign_info:
                    response += f"• **{sign_class.replace('_', ' ').title()}**\n"
                    response += f"  - Category: {sign_info.get('category', 'Unknown')}\n"
                    response += f"  - Meaning: {sign_info.get('meaning', 'N/A')}\n"
                    response += f"  - Action: {sign_info.get('action', 'N/A')}\n\n"
                else:
                    response += f"• {sign_class.replace('_', ' ').title()}\n"
            
            return response
        
        # Handle sign meaning queries
        if intent.startswith("sign_meaning:"):
            sign_name = intent.split(":")[1]
            sign_info = self.knowledge_base["traffic_signs"].get(sign_name, {})
            if sign_info:
                return f"**{sign_name.replace('_', ' ').title()}**\n\n" + \
                       f"**Category:** {sign_info.get('category', 'Unknown')}\n" + \
                       f"**Meaning:** {sign_info.get('meaning', 'N/A')}\n" + \
                       f"**Action Required:** {sign_info.get('action', 'N/A')}"
            else:
                return f"I don't have specific information about '{sign_name}' in my knowledge base."
        
        # Handle FAQ
        faq_mapping = {
            "how_to_use": "how_to_use",
            "model_selection": "model_selection",
            "performance": "accuracy",
            "confidence_threshold": "confidence_threshold",
            "supported_signs": "supported_signs"
        }
        
        if intent in faq_mapping:
            return self.knowledge_base["faq"][faq_mapping[intent]]
        
        # Handle sign categories
        if intent == "sign_categories":
            return """**Traffic Sign Categories:**

**Regulatory Signs (15 classes):** These signs give direct orders that must be obeyed. Examples include stop signs, speed limits, no entry, no parking, and one way.

**Warning Signs (10 classes):** These signs alert drivers to potential hazards ahead. Examples include pedestrian crossings, school zones, curves, and animal crossings.

**Mandatory Signs (4 classes):** These signs indicate required actions. Examples include roundabout directions, keep left/right, and bicycle paths.

Our system can detect 29 different traffic signs across these categories."""
        
        # General/fallback response
        system_info = self.knowledge_base["system"]
        return f"""Hello! I'm your AI assistant for the Bangladesh Traffic Sign Detection System.

**System Capabilities:**
- Model: {system_info['model']}
- Accuracy: {system_info['accuracy']}
- Speed: {system_info['speed']}
- Detects {system_info['classes']} traffic sign classes

**I can help you with:**
- Understanding detection results
- Explaining traffic sign meanings
- Guiding you on how to use the system
- Answering questions about model performance

Feel free to ask me anything about traffic signs or the detection system!"""
    
    def chat(self, user_message: str, detection_context: Optional[Dict] = None) -> str:
        """
        Main chat function - process user message and return response
        
        Args:
            user_message: User's input message
            detection_context: Optional context about current detection
            
        Returns:
            AI assistant's response
        """
        # Update context if provided
        if detection_context:
            self.set_detection_context(
                detection_context.get("num_detections", 0),
                detection_context.get("detected_classes", []),
                detection_context.get("model_type", "YOLOv11"),
                detection_context.get("confidence", 0.25)
            )
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Match intent and generate response
        intent = self._match_intent(user_message)
        response = self._generate_response(intent, user_message)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_detection_context = None


# Create a global instance
llm_chatbot = TrafficSignLLM()
