"""
Database models and connection for user behavior analytics
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class AnalyticsDB:
    """SQLite database for tracking user behavior and analytics"""
    
    def __init__(self, db_path: str = "analytics.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """Create database tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_detections INTEGER DEFAULT 0,
                total_frames INTEGER DEFAULT 0,
                avg_fps REAL DEFAULT 0.0
            )
        """)
        
        # Detection events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_type TEXT NOT NULL,
                confidence_threshold REAL,
                num_detections INTEGER,
                inference_time_ms REAL,
                fps REAL,
                input_type TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # User interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                details TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # Chat messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        self.conn.commit()
    
    def create_session(self, session_id: str) -> bool:
        """Create a new user session"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, start_time) VALUES (?, ?)",
                (session_id, datetime.now().isoformat())
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Session already exists
    
    def log_detection(self, session_id: str, model_type: str, 
                      confidence_threshold: float, num_detections: int,
                      inference_time_ms: float, fps: float, input_type: str):
        """Log a detection event"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detection_events 
            (session_id, timestamp, model_type, confidence_threshold, 
             num_detections, inference_time_ms, fps, input_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), model_type, 
              confidence_threshold, num_detections, inference_time_ms, fps, input_type))
        
        # Update session statistics
        cursor.execute("""
            UPDATE sessions 
            SET total_detections = total_detections + ?,
                total_frames = total_frames + 1
            WHERE session_id = ?
        """, (num_detections, session_id))
        
        self.conn.commit()
    
    def log_interaction(self, session_id: str, interaction_type: str, details: Dict = None):
        """Log a user interaction"""
        cursor = self.conn.cursor()
        details_json = json.dumps(details) if details else None
        cursor.execute("""
            INSERT INTO user_interactions 
            (session_id, timestamp, interaction_type, details)
            VALUES (?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), interaction_type, details_json))
        self.conn.commit()
    
    def log_chat_message(self, session_id: str, role: str, message: str):
        """Log a chat message (user or assistant)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages 
            (session_id, timestamp, role, message)
            VALUES (?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), role, message))
        self.conn.commit()
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics for a session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT total_detections, total_frames, avg_fps, start_time
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'total_detections': row[0],
                'total_frames': row[1],
                'avg_fps': row[2],
                'start_time': row[3]
            }
        return None
    
    def get_all_sessions_stats(self) -> List[Dict]:
        """Get statistics for all sessions"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT session_id, start_time, end_time, 
                   total_detections, total_frames, avg_fps
            FROM sessions 
            ORDER BY start_time DESC
            LIMIT 100
        """)
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'start_time': row[1],
                'end_time': row[2],
                'total_detections': row[3],
                'total_frames': row[4],
                'avg_fps': row[5]
            })
        return sessions
    
    def get_detection_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get detection history for a session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, model_type, num_detections, 
                   inference_time_ms, fps, input_type
            FROM detection_events
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'timestamp': row[0],
                'model_type': row[1],
                'num_detections': row[2],
                'inference_time_ms': row[3],
                'fps': row[4],
                'input_type': row[5]
            })
        return history
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, role, message
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """, (session_id, limit))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'timestamp': row[0],
                'role': row[1],
                'message': row[2]
            })
        return history
    
    def get_analytics_summary(self) -> Dict:
        """Get overall analytics summary"""
        cursor = self.conn.cursor()
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        # Total detections
        cursor.execute("SELECT SUM(total_detections) FROM sessions")
        total_detections = cursor.fetchone()[0] or 0
        
        # Total frames processed
        cursor.execute("SELECT SUM(total_frames) FROM sessions")
        total_frames = cursor.fetchone()[0] or 0
        
        # Average FPS across all sessions
        cursor.execute("SELECT AVG(avg_fps) FROM sessions WHERE avg_fps > 0")
        avg_fps = cursor.fetchone()[0] or 0
        
        # Most popular model
        cursor.execute("""
            SELECT model_type, COUNT(*) as count 
            FROM detection_events 
            GROUP BY model_type 
            ORDER BY count DESC 
            LIMIT 1
        """)
        popular_model_row = cursor.fetchone()
        popular_model = popular_model_row[0] if popular_model_row else "N/A"
        
        # Most popular input type
        cursor.execute("""
            SELECT input_type, COUNT(*) as count 
            FROM detection_events 
            GROUP BY input_type 
            ORDER BY count DESC 
            LIMIT 1
        """)
        popular_input_row = cursor.fetchone()
        popular_input = popular_input_row[0] if popular_input_row else "N/A"
        
        return {
            'total_sessions': total_sessions,
            'total_detections': total_detections,
            'total_frames': total_frames,
            'avg_fps': round(avg_fps, 2),
            'popular_model': popular_model,
            'popular_input_type': popular_input
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
