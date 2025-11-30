"""
Ultra-Simple Live Face Recognition
Uses OpenCV LBPH for real-time detection
"""

import cv2
import os
import time
from datetime import datetime
from pathlib import Path

def load_labels():
    """Load label mapping from file."""
    label_dict = {}
    
    if not os.path.exists('labels.txt'):
        return label_dict
    
    with open('labels.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                label_dict[int(parts[0])] = parts[1]
    
    return label_dict

def main():
    """Main live recognition loop."""
    print("=" * 60)
    print("Ultra-Simple Live Face Recognition")
    print("=" * 60)
    
    # Check for trained model
    if not os.path.exists('trainer.yml'):
        print("\n❌ No trainer.yml found!")
        print("Run 'python train.py' first to train the model.")
        return
    
    # Load model and labels
    print("\n✅ Loading LBPH model...")
    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.read('trainer.yml')
    
    labels = load_labels()
    if not labels:
        print("⚠️  No labels found!")
    else:
        print(f"👥 Looking for: {', '.join(labels.values())}")
    
    # Open webcam with DirectShow backend on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("\n❌ No webcam! Check connection.")
        print("💡 TIP: Close other apps using webcam (Zoom, Teams, etc.)")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warm up camera
    print("\n🔄 Warming up webcam...")
    for _ in range(5):
        cap.read()
    
    # Load face detector
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("✅ Webcam ready! Press Q or ESC to quit.\n")
    
    # Create unknown folder
    unknown_dir = Path("known_faces") / "unknown"
    unknown_dir.mkdir(parents=True, exist_ok=True)
    
    # FPS tracking
    prev_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️  Failed to read frame, retrying...")
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        # Process each face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))
            
            # Predict
            label, conf = lbph.predict(face_resized)
            
            # Confidence threshold
            if conf < 150:  # Lower is better
                name = labels.get(label, "Unknown")
                color = (0, 255, 0)  # Green
                text = f"{name} ({conf:.0f}%)"
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red
                text = "Unknown"
                
                # Save unknown face
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unknown_path = unknown_dir / f"{timestamp}.jpg"
                cv2.imwrite(str(unknown_path), face_resized)
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, text, (x+6, y-6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Console output
        if len(faces) > 0:
            print(f"Detected {len(faces)} face(s)", end='\r')
        
        # Show frame (static window name to prevent multiple windows)
        cv2.imshow('Live Face Recognition - Press Q to quit', frame)
        
        # Exit on Q or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            print("\n\n👋 Exiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Live recognition stopped")
    print("=" * 60)

if __name__ == "__main__":
    main()
