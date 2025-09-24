from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import sqlite3
import json
import os
from contextlib import contextmanager
import logging
from collections import defaultdict
import asyncio
from fastapi import Query

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class WasteEvent(BaseModel):
    timestamp: str
    device_id: str
    category: str = Field(..., pattern="^(biodegradable|recyclable|landfill)$")
    confidence: float = Field(..., ge=0, le=1)
    inference_time: Optional[float] = None
    event_id: Optional[str] = None
    user_id: Optional[str] = None

class DeviceRegistration(BaseModel):
    device_id: str
    location: Dict[str, float]  # {"lat": 40.7128, "lng": -74.0060}
    description: Optional[str] = None

class UserReport(BaseModel):
    device_id: str
    user_id: Optional[str] = None
    report_type: str = Field(..., pattern="^(incorrect_classification|bin_full|maintenance)$")
    description: str
    image_url: Optional[str] = None

class RewardClaim(BaseModel):
    user_id: str
    event_id: str
    points: int = Field(default=10, ge=1, le=100)

class StatsResponse(BaseModel):
    total_events: int
    events_by_category: Dict[str, int]
    events_by_device: Dict[str, int]
    accuracy_metrics: Dict[str, float]
    time_series: List[Dict[str, Any]]

class WasteAPI:
    def __init__(self, db_path: str = "waste_backend.db"):
        self.db_path = db_path
        self.api_tokens = {"demo_token": "edge_device", "admin_token": "admin"}
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with required tables"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    inference_time REAL,
                    event_id TEXT,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Devices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    latitude REAL,
                    longitude REAL,
                    description TEXT,
                    last_seen TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Users and rewards table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    total_points INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    badges TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rewards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_id TEXT,
                    points INTEGER NOT NULL,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    user_id TEXT,
                    report_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    image_url TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_device ON events (device_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_category ON events (category)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def get_db(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials) -> str:
        """Verify API token and return user type"""
        token = credentials.credentials
        if token in self.api_tokens:
            return self.api_tokens[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    def add_event(self, event: WasteEvent) -> int:
        """Add a waste classification event"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (timestamp, device_id, category, confidence, 
                                  inference_time, event_id, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp,
                event.device_id,
                event.category,
                event.confidence,
                event.inference_time,
                event.event_id,
                event.user_id
            ))
            
            # Update device last seen
            cursor.execute('''
                UPDATE devices SET last_seen = CURRENT_TIMESTAMP 
                WHERE device_id = ?
            ''', (event.device_id,))
            
            conn.commit()
            return cursor.lastrowid
    
    def register_device(self, device: DeviceRegistration):
        """Register a new edge device"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO devices 
                (device_id, latitude, longitude, description, last_seen)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                device.device_id,
                device.location.get('lat'),
                device.location.get('lng'),
                device.description
            ))
            conn.commit()
    
    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get aggregated statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            # Total events
            cursor.execute('SELECT COUNT(*) FROM events WHERE timestamp > ?', 
                          (cutoff_time.isoformat(),))
            total_events = cursor.fetchone()[0]
            
            # Events by category
            cursor.execute('''
                SELECT category, COUNT(*) FROM events 
                WHERE timestamp > ?
                GROUP BY category
            ''', (cutoff_time.isoformat(),))
            events_by_category = dict(cursor.fetchall())
            
            # Events by device
            cursor.execute('''
                SELECT device_id, COUNT(*) FROM events 
                WHERE timestamp > ?
                GROUP BY device_id
            ''', (cutoff_time.isoformat(),))
            events_by_device = dict(cursor.fetchall())
            
            # Average confidence by category
            cursor.execute('''
                SELECT category, AVG(confidence) FROM events 
                WHERE timestamp > ?
                GROUP BY category
            ''', (cutoff_time.isoformat(),))
            avg_confidence = dict(cursor.fetchall())
            
            # Time series data (hourly buckets)
            cursor.execute('''
                SELECT 
                    datetime((strftime('%s', timestamp) / 3600) * 3600, 'unixepoch') as hour,
                    category,
                    COUNT(*) as count
                FROM events 
                WHERE timestamp > ?
                GROUP BY hour, category
                ORDER BY hour
            ''', (cutoff_time.isoformat(),))
            
            time_series_raw = cursor.fetchall()
            
            # Process time series data
            time_series = []
            hourly_data = defaultdict(lambda: defaultdict(int))
            
            for hour, category, count in time_series_raw:
                hourly_data[hour][category] = count
            
            for hour in sorted(hourly_data.keys()):
                time_series.append({
                    'timestamp': hour,
                    'biodegradable': hourly_data[hour].get('biodegradable', 0),
                    'recyclable': hourly_data[hour].get('recyclable', 0),
                    'landfill': hourly_data[hour].get('landfill', 0)
                })
            
            return {
                'total_events': total_events,
                'events_by_category': events_by_category,
                'events_by_device': events_by_device,
                'accuracy_metrics': {
                    'avg_confidence': avg_confidence,
                    'high_confidence_ratio': self._get_high_confidence_ratio(cursor, cutoff_time)
                },
                'time_series': time_series,
                'period_hours': hours
            }
    
    def _get_high_confidence_ratio(self, cursor, cutoff_time):
        """Calculate ratio of high confidence predictions"""
        cursor.execute('SELECT confidence FROM events WHERE timestamp > ?', 
                      (cutoff_time.isoformat(),))
        confidences = [row[0] for row in cursor.fetchall()]
        
        if not confidences:
            return 0.0
        
        high_confidence = sum(1 for c in confidences if c >= 0.8)
        return high_confidence / len(confidences)
    
    def get_devices(self) -> List[Dict[str, Any]]:
        """Get all registered devices with their status"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.device_id, d.latitude, d.longitude, d.description, 
                       d.last_seen, d.status, COUNT(e.id) as event_count
                FROM devices d
                LEFT JOIN events e ON d.device_id = e.device_id 
                    AND e.timestamp > datetime('now', '-24 hours')
                GROUP BY d.device_id
            ''')
            
            devices = []
            for row in cursor.fetchall():
                device_id, lat, lng, desc, last_seen, status, event_count = row
                devices.append({
                    'device_id': device_id,
                    'location': {'lat': lat, 'lng': lng} if lat and lng else None,
                    'description': desc,
                    'last_seen': last_seen,
                    'status': status,
                    'events_24h': event_count
                })
            
            return devices
    
    def add_user_report(self, report: UserReport) -> int:
        """Add a user report"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reports (device_id, user_id, report_type, description, image_url)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                report.device_id,
                report.user_id,
                report.report_type,
                report.description,
                report.image_url
            ))
            conn.commit()
            return cursor.lastrowid
    
    def award_points(self, reward: RewardClaim) -> bool:
        """Award points to a user"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            # Create user if doesn't exist
            cursor.execute('''
                INSERT OR IGNORE INTO users (user_id) VALUES (?)
            ''', (reward.user_id,))
            
            # Add reward
            cursor.execute('''
                INSERT INTO rewards (user_id, event_id, points, reason)
                VALUES (?, ?, ?, ?)
            ''', (reward.user_id, reward.event_id, reward.points, "Correct waste sorting"))
            
            # Update user total points
            cursor.execute('''
                UPDATE users SET total_points = total_points + ?
                WHERE user_id = ?
            ''', (reward.points, reward.user_id))
            
            conn.commit()
            return True

# Initialize API
waste_api = WasteAPI()
app = FastAPI(title="Smart Waste Sorting API", version="1.0.0")
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.post("/api/events")
async def create_event(
    event: WasteEvent,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Receive waste classification event from edge device"""
    user_type = waste_api.verify_token(credentials)
    
    try:
        event_id = waste_api.add_event(event)
        
        # Background task for additional processing
        background_tasks.add_task(process_event_background, event, event_id)
        
        return {
            "status": "success",
            "event_id": event_id,
            "message": "Event recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error adding event: {e}")
        raise HTTPException(status_code=500, detail="Failed to record event")

@app.get("/api/stats")
async def get_stats(
    hours: int = 24,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get system statistics"""
    waste_api.verify_token(credentials)
    
    try:
        stats = waste_api.get_stats(hours)
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.post("/api/devices")
async def register_device(
    device: DeviceRegistration,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Register a new edge device"""
    waste_api.verify_token(credentials)
    
    try:
        waste_api.register_device(device)
        return {"status": "success", "message": "Device registered successfully"}
    except Exception as e:
        logger.error(f"Error registering device: {e}")
        raise HTTPException(status_code=500, detail="Failed to register device")

@app.get("/api/devices")
async def get_devices(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get all registered devices"""
    waste_api.verify_token(credentials)
    
    try:
        devices = waste_api.get_devices()
        return {"devices": devices}
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get devices")

@app.post("/api/reports")
async def create_report(
    report: UserReport,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Create a user report"""
    waste_api.verify_token(credentials)
    
    try:
        report_id = waste_api.add_user_report(report)
        return {
            "status": "success",
            "report_id": report_id,
            "message": "Report submitted successfully"
        }
    except Exception as e:
        logger.error(f"Error creating report: {e}")
        raise HTTPException(status_code=500, detail="Failed to create report")

@app.post("/api/rewards")
async def award_points(
    reward: RewardClaim,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Award points to a user"""
    waste_api.verify_token(credentials)
    
    try:
        success = waste_api.award_points(reward)
        if success:
            return {"status": "success", "message": "Points awarded successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to award points")
    except Exception as e:
        logger.error(f"Error awarding points: {e}")
        raise HTTPException(status_code=500, detail="Failed to award points")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
@app.get("/api/events/recent")
async def recent_events(
    limit: int = Query(10, ge=1, le=100),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    waste_api.verify_token(credentials)
    try:
        with waste_api.get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, timestamp, device_id, category, confidence, user_id, created_at
                FROM events
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = c.fetchall()
        events = [{
            "event_id": f"evt_{r[0]}",
            "timestamp": r[1],
            "device_id": r[2],
            "category": r[3],
            "confidence": r[4],
            "user_id": r[5],
            "created_at": r[6]
        } for r in rows]
        return {"events": events}
    except Exception as e:
        logger.error(f"Error fetching recent events: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent events")

async def process_event_background(event: WasteEvent, event_id: int):
    """Background processing for events"""
    try:
        # Simulate some background processing
        await asyncio.sleep(0.1)
        
        # Could add logic here for:
        # - Anomaly detection
        # - Model performance monitoring
        # - Alert generation for full bins
        # - Route optimization triggers
        
        logger.info(f"Background processing completed for event {event_id}")
    except Exception as e:
        logger.error(f"Background processing failed for event {event_id}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)