#!/usr/bin/env python3
"""
User Behavior Tracking Module

Tracks user interactions with the traffic sign detection system
for analysis and insights generation using Web-LLM.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field
import threading


@dataclass
class DetectionEvent:
    """Represents a single detection event."""
    timestamp: str
    detection_count: int
    classes_detected: List[str]
    avg_confidence: float
    inference_time_ms: float
    model_used: str
    image_source: str  # 'webcam', 'upload', 'video'


@dataclass
class InteractionEvent:
    """Represents a user interaction event."""
    timestamp: str
    action: str  # 'detect', 'compare', 'change_model', 'adjust_conf', etc.
    details: Dict[str, Any]


@dataclass
class SessionData:
    """Complete session data for a user."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_detections: int = 0
    total_frames: int = 0
    detection_events: List[DetectionEvent] = field(default_factory=list)
    interaction_events: List[InteractionEvent] = field(default_factory=list)
    models_used: Dict[str, int] = field(default_factory=dict)  # model_name: count
    common_signs: Dict[str, int] = field(default_factory=dict)  # sign_name: count
    avg_confidence: float = 0.0
    avg_inference_time: float = 0.0


class BehaviorTracker:
    """
    Tracks user behavior and detection patterns for analysis.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, storage_path: str = "data/analytics/sessions.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[SessionData] = None
        self._lock = threading.Lock()
        
        # Load existing sessions
        self.sessions: List[SessionData] = self._load_sessions()
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new tracking session."""
        with self._lock:
            if session_id is None:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_session = SessionData(
                session_id=session_id,
                start_time=datetime.now().isoformat()
            )
            
            return session_id
    
    def end_session(self):
        """End the current session and save data."""
        with self._lock:
            if self.current_session:
                self.current_session.end_time = datetime.now().isoformat()
                self.sessions.append(self.current_session)
                self._save_sessions()
                self.current_session = None
    
    def track_detection(
        self,
        detection_count: int,
        classes_detected: List[str],
        avg_confidence: float,
        inference_time_ms: float,
        model_used: str,
        image_source: str = "upload"
    ):
        """Track a detection event."""
        with self._lock:
            if not self.current_session:
                self.start_session()
            
            event = DetectionEvent(
                timestamp=datetime.now().isoformat(),
                detection_count=detection_count,
                classes_detected=classes_detected,
                avg_confidence=avg_confidence,
                inference_time_ms=inference_time_ms,
                model_used=model_used,
                image_source=image_source
            )
            
            self.current_session.detection_events.append(event)
            self.current_session.total_detections += detection_count
            self.current_session.total_frames += 1
            
            # Update model usage
            if model_used not in self.current_session.models_used:
                self.current_session.models_used[model_used] = 0
            self.current_session.models_used[model_used] += 1
            
            # Update common signs
            for sign in classes_detected:
                if sign not in self.current_session.common_signs:
                    self.current_session.common_signs[sign] = 0
                self.current_session.common_signs[sign] += 1
            
            # Update averages
            total = self.current_session.total_frames
            self.current_session.avg_confidence = (
                (self.current_session.avg_confidence * (total - 1) + avg_confidence) / total
            )
            self.current_session.avg_inference_time = (
                (self.current_session.avg_inference_time * (total - 1) + inference_time_ms) / total
            )
    
    def track_interaction(self, action: str, details: Dict[str, Any] = None):
        """Track a user interaction event."""
        with self._lock:
            if not self.current_session:
                self.start_session()
            
            event = InteractionEvent(
                timestamp=datetime.now().isoformat(),
                action=action,
                details=details or {}
            )
            
            self.current_session.interaction_events.append(event)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session for LLM analysis."""
        with self._lock:
            if not self.current_session:
                return {"error": "No active session"}
            
            # Calculate session duration
            start = datetime.fromisoformat(self.current_session.start_time)
            duration_seconds = (datetime.now() - start).total_seconds()
            
            # Get top signs
            top_signs = sorted(
                self.current_session.common_signs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Get most used model
            most_used_model = max(
                self.current_session.models_used.items(),
                key=lambda x: x[1]
            ) if self.current_session.models_used else ("N/A", 0)
            
            return {
                "session_id": self.current_session.session_id,
                "duration_seconds": duration_seconds,
                "total_detections": self.current_session.total_detections,
                "total_frames": self.current_session.total_frames,
                "avg_confidence": round(self.current_session.avg_confidence, 4),
                "avg_inference_time_ms": round(self.current_session.avg_inference_time, 2),
                "top_signs": [{"sign": k, "count": v} for k, v in top_signs],
                "most_used_model": most_used_model[0],
                "models_usage": self.current_session.models_used,
                "total_interactions": len(self.current_session.interaction_events)
            }
    
    def get_behavior_insights(self) -> Dict[str, Any]:
        """Generate behavior insights for LLM analysis."""
        # First get summary (which uses its own lock)
        summary = self.get_session_summary()
        
        if "error" in summary:
            return {"error": "No active session"}
        
        with self._lock:
            # Calculate usage patterns
            source_distribution = {}
            for event in self.current_session.detection_events:
                source = event.image_source
                if source not in source_distribution:
                    source_distribution[source] = 0
                source_distribution[source] += 1
            
            # Calculate interaction patterns
            action_distribution = {}
            for event in self.current_session.interaction_events:
                action = event.action
                if action not in action_distribution:
                    action_distribution[action] = 0
                action_distribution[action] += 1
            
            timeline = [
                {
                    "time": event.timestamp,
                    "count": event.detection_count,
                    "model": event.model_used,
                    "confidence": event.avg_confidence
                }
                for event in self.current_session.detection_events[-10:]  # Last 10
            ]
        
        return {
            "summary": summary,
            "source_distribution": source_distribution,
            "action_distribution": action_distribution,
            "detection_timeline": timeline
        }
    
    def _load_sessions(self) -> List[SessionData]:
        """Load sessions from storage."""
        if not self.storage_path.exists():
            return []
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return [SessionData(**session) for session in data]
        except Exception as e:
            print(f"Error loading sessions: {e}")
            return []
    
    def _save_sessions(self):
        """Save sessions to storage."""
        try:
            data = [asdict(session) for session in self.sessions]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving sessions: {e}")


# Global tracker instance
_global_tracker = None


def get_tracker() -> BehaviorTracker:
    """Get or create global behavior tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = BehaviorTracker()
    return _global_tracker
