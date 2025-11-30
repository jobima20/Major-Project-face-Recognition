"""
Ultra-Simple Face Recognition Training
Uses OpenCV LBPH (no dlib, fast install)
"""

import cv2
import numpy as np
import os
from pathlib import Path

def check_blur(image, threshold=100):
    """Check if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold, variance

def collect_images(person_name, num_images=20):
    """Collect training images for a person."""
    person_dir = Path("known_faces") / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📸 Collecting {num_images} images for {person_name}")
    print("Press SPACE to capture (20 times)")
    
    # Use DirectShow backend on Windows to avoid hanging
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam!")
        print("💡 TIP: Close other apps using webcam (Zoom, Teams, etc.)")
        return 0
    
    # Warm up the camera (read a few frames to initialize)
    print("🔄 Warming up webcam...")
    for _ in range(5):
        cap.read()
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    collected = 0
    
    print("✅ Webcam ready! Look at the window and press SPACE.\n")
    
    while collected < num_images:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read frame, retrying...")
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        display = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(display, f"Captured: {collected}/{num_images}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press SPACE to capture", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show face detection status
        if len(faces) == 0:
            cv2.putText(display, "NO FACE DETECTED!", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif len(faces) > 1:
            cv2.putText(display, f"{len(faces)} FACES - Only 1 person please!", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(display, "READY - Press SPACE", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(f'Training: {person_name}', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            if len(faces) == 0:
                print("⚠️  No face detected! Position yourself in front of webcam.")
            elif len(faces) > 1:
                print(f"⚠️  {len(faces)} faces detected! Only one person should be in frame.")
            elif len(faces) == 1:
                is_sharp, blur_score = check_blur(gray)
                
                if is_sharp:
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (200, 200))
                    
                    img_path = person_dir / f"img_{collected}.jpg"
                    cv2.imwrite(str(img_path), face_resized)
                    collected += 1
                    print(f"✅ Image {collected}/{num_images} (blur: {blur_score:.1f})")
                else:
                    print(f"⚠️  Blurry - retake (score: {blur_score:.1f})")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return collected

def train_model():
    """Train LBPH model from collected images."""
    known_faces_dir = Path("known_faces")
    
    if not known_faces_dir.exists():
        print("❌ No known_faces folder found!")
        return False
    
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    print("\n🔍 Loading images...")
    
    for person_dir in known_faces_dir.iterdir():
        if not person_dir.is_dir() or person_dir.name == "unknown":
            continue
        
        person_name = person_dir.name
        label_dict[current_label] = person_name
        
        for img_file in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(current_label)
        
        print(f"  ✅ {person_name}: {len([l for l in labels if l == current_label])} images")
        current_label += 1
    
    if len(faces) == 0:
        print("❌ No training images found!")
        return False
    
    print(f"\n🤖 Training LBPH model ({len(faces)} total images)...")
    
    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.train(faces, np.array(labels))
    lbph.save('trainer.yml')
    
    # Save label mapping
    with open('labels.txt', 'w') as f:
        for label, name in label_dict.items():
            f.write(f"{label}:{name}\n")
    
    print(f"✅ Trained {len(label_dict)} people successfully!")
    return True

def main():
    """Main training workflow."""
    print("=" * 60)
    print("Ultra-Simple Face Recognition Training")
    print("=" * 60)
    
    while True:
        person_name = input("\n👤 Person name (or 'done'): ").strip()
        
        if person_name.lower() == 'done':
            break
        
        if not person_name:
            print("❌ Name cannot be empty!")
            continue
        
        collected = collect_images(person_name, num_images=20)
        
        if collected > 0:
            done = input(f"\n✅ Collected {collected} images. Done with {person_name}? (y/n): ").lower()
            if done != 'y':
                continue
    
    # Train the model
    if train_model():
        print("\n" + "=" * 60)
        print("✅ Training complete! Run 'python live.py' to test.")
        print("=" * 60)
    else:
        print("\n⚠️  Training failed. Add more people and try again.")

if __name__ == "__main__":
    main()
