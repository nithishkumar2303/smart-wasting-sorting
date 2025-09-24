import cv2
import numpy as np
import json
import requests
import time
import threading
import queue
from datetime import datetime
import os
import argparse
from collections import deque
import sqlite3
from datetime import datetime, timezone

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    print("TensorFlow Lite runtime not available, using TensorFlow")
    import tensorflow as tf
    TFLITE_AVAILABLE = False

def load_class_names(path="model_info.json", default=None):
        default = default or ["biodegradable", "recyclable", "landfill"]
        try:
            with open(path, "r") as f:
                return json.load(f).get("class_names", default)
        except Exception:
            return default

class EdgeInferenceSystem:
    def __init__(self, model_path, config_file='edge_config.json'):
        self.model_path = model_path
        info_path = os.path.join(os.path.dirname(self.model_path), "model_info.json")
        self.config = self.load_config(config_file)
        self.interpreter = None
        self.class_names = load_class_names(info_path, default=['biodegradable','recyclable','landfill'])
        self.tips = self.load_tips()
        
        # Initialize local database for offline storage
        self.init_local_db()
        
        # Event queue for background processing
        self.event_queue = queue.Queue()
        self.running = False
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        
        self.load_model()
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            "device_id": "edge_device_001",
            "api_endpoint": "http://localhost:8000/api/events",
            "api_token": "demo_token",
            "confidence_threshold": 0.7,
            "input_resolution": [224, 224],
            "camera_index": 0,
            "inference_interval": 2.0,
            "offline_mode": False,
            "max_offline_events": 1000
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def load_tips(self):
        """Load educational tips for each category"""
        return {
            'biodegradable': [
                "Great! Organic waste will decompose naturally.",
                "Tip: Remove any stickers or plastic wrapping first.",
                "Composting reduces methane emissions from landfills."
            ],
            'recyclable': [
                "Perfect! This helps create new products.",
                "Tip: Clean containers before recycling.",
                "Recycling saves energy and natural resources."
            ],
            'landfill': [
                "This goes to landfill. Consider reducing such items.",
                "Tip: Look for reusable alternatives next time.",
                "Landfill waste takes decades to decompose."
            ]
        }
    
    def init_local_db(self):
        """Initialize SQLite database for offline event storage"""
        self.db_path = 'edge_events.db'
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                device_id TEXT,
                category TEXT,
                confidence REAL,
                image_path TEXT,
                synced BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_model(self):
        """Load TensorFlow Lite model"""
        try:
            if TFLITE_AVAILABLE:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            else:
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Model loaded successfully: {self.model_path}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        target_size = tuple(self.config.get('input_resolution', [224,224]))
        
        # Resize image
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Normalize to [0, 1]
        image= (image-127.5)/127.5
        
        return np.expand_dims(image, axis=0)
    
    def run_inference(self, image):
        """Run inference on preprocessed image"""
        start_time = time.time()
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output[0]  # Remove batch dimension
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return predictions, inference_time
    
    def classify_image(self, image):
        processed_image = self.preprocess_image(image)
        predictions, inference_time = self.run_inference(processed_image)
        probs = predictions  # 1D array of softmax scores

        # Initial top-1
        predicted_class_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_class_idx])
        category = self.class_names[predicted_class_idx]

        # Guardrail: prefer landfill if recyclable is uncertain
        if 'recyclable' in self.class_names and 'landfill' in self.class_names:
            i_rec = self.class_names.index('recyclable')
            i_lf  = self.class_names.index('landfill')
            p_rec = float(probs[i_rec])
            p_lf  = float(probs[i_lf])

            # Allow thresholds from config; use defaults if missing
            rec_hi = float(self.config.get('recycle_decision_threshold', 0.70))
            lf_min = float(self.config.get('landfill_guardrail_threshold', 0.30))

            if p_rec < rec_hi and p_lf > lf_min:
                category = 'landfill'
                confidence = p_lf

        return {
            'category': category,
            'confidence': confidence,
            'inference_time': inference_time,
            'all_predictions': {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
        }
    
    def save_event_locally(self, event_data, image_path=None):
        """Save event to local database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO events (timestamp, device_id, category, confidence, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            event_data['timestamp'],
            event_data['device_id'],
            event_data['category'],
            event_data['confidence'],
            image_path
        ))
        
        conn.commit()
        event_id = cursor.lastrowid
        conn.close()
        
        return event_id
    
    def send_to_backend(self, event_data):
        """Send event data to backend API"""
        try:
            headers = {
                'Authorization': f"Bearer {self.config['api_token']}",
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.config['api_endpoint'],
                json=event_data,
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"HTTP {response.status_code}: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def sync_offline_events(self):
        """Sync stored offline events with backend"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unsynced events
        cursor.execute('SELECT * FROM events WHERE synced = FALSE LIMIT 50')
        events = cursor.fetchall()
        
        synced_count = 0
        for event in events:
            event_id, timestamp, device_id, category, confidence, image_path, synced, created_at = event
            
            event_data = {
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            'device_id': self.config['device_id'],
            'category': result['category'],
            'confidence': result['confidence'],
            'inference_time': result['inference_time']
            }
            
            success, response = self.send_to_backend(event_data)
            if success:
                # Mark as synced
                cursor.execute('UPDATE events SET synced = TRUE WHERE id = ?', (event_id,))
                synced_count += 1
            else:
                print(f"Failed to sync event {event_id}: {response}")
                break  # Stop syncing on first failure
        
        conn.commit()
        conn.close()
        
        if synced_count > 0:
            print(f"Synced {synced_count} offline events")
    
    def process_event_queue(self):
        """Background thread to process events"""
        while self.running:
            try:
                event_data = self.event_queue.get(timeout=1)
                
                # Save locally first
                self.save_event_locally(event_data)
                
                # Try to send to backend if not in offline mode
                if not self.config['offline_mode']:
                    success, response = self.send_to_backend(event_data)
                    if not success:
                        print(f"Failed to send event: {response}")
                
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def display_result(self, image, result):
        """Display result on image with tip"""
        category = result['category']
        confidence = result['confidence']
        
        # Choose tip
        tip = np.random.choice(self.tips[category])
        
        # Color coding
        colors = {
            'biodegradable': (34, 139, 34),    # Green
            'recyclable': (30, 144, 255),      # Blue
            'landfill': (128, 128, 128)        # Gray
        }
        
        color = colors.get(category, (255, 255, 255))
        
        # Add text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Category and confidence
        text1 = f"{category.upper()}: {confidence:.1%}"
        cv2.putText(image, text1, (10, 30), font, 0.8, color, 2)
        
        # Tip (wrap long text)
        words = tip.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) > 40:  # Approximate character limit
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw tip lines
        for i, line in enumerate(lines[:2]):  # Max 2 lines
            cv2.putText(image, line, (10, 70 + i * 25), font, 0.5, (255, 255, 255), 1)
        
        # Performance info
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            perf_text = f"FPS: {fps:.1f} | Avg: {avg_time*1000:.1f}ms"
            cv2.putText(image, perf_text, (10, image.shape[0] - 20), font, 0.4, (200, 200, 200), 1)
        
        return image

    def run_camera_demo(self, camera_index=None):
        """Run live camera demo"""
        if camera_index is None:
            camera_index = self.config['camera_index']
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("Starting camera demo. Press 'q' to quit, 's' to save prediction")
        
        # Start background processing thread
        self.running = True
        processing_thread = threading.Thread(target=self.process_event_queue, daemon=True)
        processing_thread.start()
        
        last_inference_time = 0
        inference_interval = self.config['inference_interval']
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Run inference at specified interval
                if current_time - last_inference_time >= inference_interval:
                    result = self.classify_image(frame)
                    last_inference_time = current_time
                    
                    # Create event data
                    event_data = {
                        'timestamp': datetime.now().isoformat(),
                        'device_id': self.config['device_id'],
                        'category': result['category'],
                        'confidence': result['confidence'],
                        'inference_time': result['inference_time']
                    }
                    
                    # Queue for background processing
                    if result['confidence'] >= self.config['confidence_threshold']:
                        self.event_queue.put(event_data)
                    
                    # Display result
                    frame_with_result = self.display_result(frame.copy(), result)
                else:
                    frame_with_result = frame
                
                cv2.imshow('Smart Waste Sorting - Edge Demo', frame_with_result)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'captured_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame: {filename}")
        
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            processing_thread.join(timeout=2)
    
    def get_stats(self):
        """Get device statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM events')
        total_events = cursor.fetchone()[0]
        
        cursor.execute('SELECT category, COUNT(*) FROM events GROUP BY category')
        category_counts = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM events WHERE synced = FALSE')
        unsynced_events = cursor.fetchone()[0]
        
        conn.close()
        
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        return {
            'device_id': self.config['device_id'],
            'total_events': total_events,
            'category_counts': category_counts,
            'unsynced_events': unsynced_events,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'model_path': self.model_path
        }

    def preprocess_image(self, image):
        target_size = tuple(self.config['input_resolution'])
        # Convert BGR -> RGB to match training
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Smart Waste Sorting Edge System')
    parser.add_argument('--model', default='model/waste_model.tflite', help='Path to TFLite model')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--config', default='edge_config.json', help='Configuration file')
    parser.add_argument('--stats', action='store_true', help='Show device statistics')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train the model first using the training script")
        return
    
    # Initialize edge system
    edge_system = EdgeInferenceSystem(args.model, args.config)
    
    if args.stats:
        # Show statistics
        stats = edge_system.get_stats()
        print("\nEdge Device Statistics:")
        print("=" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        # Run camera demo
        print("Smart Waste Sorting Edge System")
        print(f"Model: {args.model}")
        print(f"Device ID: {edge_system.config['device_id']}")
        
        edge_system.run_camera_demo(args.camera)

if __name__ == "__main__":
    main()