"""
FaceScan Pro - Flask Web Application
Simple face recognition web interface using OpenCV LBPH
"""

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import base64

app = Flask(__name__)

# Global state
lbph = cv2.face.LBPHFaceRecognizer_create()
labels = {}
detection_logs = []
camera = None
training_images = {}
camera_active = False  # Control flag for live feed

# Initialize
KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DIR.mkdir(exist_ok=True)
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.txt"

def load_model():
    """Load trained LBPH model and labels"""
    global lbph, labels
    
    if os.path.exists(TRAINER_FILE) and os.path.exists(LABELS_FILE):
        lbph.read(TRAINER_FILE)
        
        with open(LABELS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    labels[int(parts[0])] = parts[1]
        
        print(f"✅ Loaded model with {len(labels)} people")
        return True
    return False

def check_blur(image, threshold=100):
    """Check if image is blurry"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold, variance

def get_camera():
    """Get camera instance"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def generate_frames():
    """Generate video frames with face detection"""
    global camera, camera_active
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Get camera
    camera = get_camera()
    
    while camera_active:
        success, frame = camera.read()
        
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        # Process each face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))
            
            if os.path.exists(TRAINER_FILE):
                try:
                    label, conf = lbph.predict(face_resized)
                    
                    if conf < 150:
                        name = labels.get(label, "Unknown")
                        color = (0, 255, 0)
                        text = f"{name} ({conf:.0f}%)"
                        
                        # Log detection
                        detection_logs.append({
                            'name': name,
                            'confidence': f"{conf:.0f}",
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        if len(detection_logs) > 20:
                            detection_logs.pop(0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        text = "Unknown"
                except:
                    color = (0, 0, 255)
                    text = "No Model"
            else:
                color = (255, 165, 0)
                text = "Train First"
            
            # Draw box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, text, (x+6, y-6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Release camera when stopped
    if camera is not None:
        camera.release()
        camera = None

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """Capture image from webcam"""
    data = request.json
    name = data.get('name')
    image_data = data.get('image')
    
    if not name or not image_data:
        return jsonify({'success': False, 'error': 'Missing data'})
    
    # Decode base64 image
    try:
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check blur
        is_sharp, blur_score = check_blur(img)
        
        if not is_sharp:
            return jsonify({'success': False, 'error': 'Image too blurry', 'blur_score': blur_score})
        
        # Detect face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected'})
        elif len(faces) > 1:
            return jsonify({'success': False, 'error': 'Multiple faces detected'})
        
        # Save image
        if name not in training_images:
            training_images[name] = []
        
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200, 200))
        
        training_images[name].append(face_resized)
        
        return jsonify({
            'success': True,
            'count': len(training_images[name]),
            'blur_score': blur_score
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train LBPH model with collected images"""
    global lbph, labels
    
    if not training_images:
        return jsonify({'success': False, 'error': 'No training data'})
    
    try:
        faces = []
        face_labels = []
        label_dict = {}
        current_label = 0
        
        # Prepare training data
        for person_name, images in training_images.items():
            person_dir = KNOWN_FACES_DIR / person_name
            person_dir.mkdir(exist_ok=True)
            
            label_dict[current_label] = person_name
            
            for idx, img in enumerate(images):
                faces.append(img)
                face_labels.append(current_label)
                
                # Save image
                cv2.imwrite(str(person_dir / f"img_{idx}.jpg"), img)
            
            current_label += 1
        
        # Train model
        lbph = cv2.face.LBPHFaceRecognizer_create()
        lbph.train(faces, np.array(face_labels))
        lbph.save(TRAINER_FILE)
        
        # Update labels
        labels.update(label_dict)
        
        # Save labels
        with open(LABELS_FILE, 'w') as f:
            for label, name in labels.items():
                f.write(f"{label}:{name}\n")
        
        # Clear training data
        training_images.clear()
        
        return jsonify({
            'success': True,
            'message': f'Trained {len(label_dict)} people successfully!'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logs')
def get_logs():
    """Get detection logs"""
    return jsonify({'logs': detection_logs[-20:]})

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """Clear detection logs"""
    detection_logs.clear()
    return jsonify({'success': True})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera feed"""
    global camera_active
    camera_active = True
    return jsonify({'success': True})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera feed"""
    global camera_active, camera
    camera_active = False
    
    # Release camera
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({'success': True})

if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("🚀 FaceScan Pro - Starting Web Server")
    print("="*60)
    print("📱 Open browser: http://localhost:5000")
    print("⚡ Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
