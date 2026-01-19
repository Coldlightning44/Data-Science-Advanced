import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from datetime import datetime
import json
import winsound
import face_recognition
import pickle

class FocusTracker:
    def __init__(self):
        self.log_dir = Path("data/focus_logs")
        self.att_dir = Path("data/attendance")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.att_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load AI Models
        # Face Landmarker (Attention)
        face_base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # Object Detector (Phone - Int8 recommended)
        obj_base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
        obj_options = vision.ObjectDetectorOptions(
            base_options=obj_base_options,
            running_mode=vision.RunningMode.VIDEO,
            score_threshold=0.4,
            category_allowlist=["cell phone", "mobile phone"]
        )
        self.object_detector = vision.ObjectDetector.create_from_options(obj_options)
        
        # 2. Load Known Faces for Recognition
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()

    def load_encodings(self):
        enc_path = Path("data/encodings/encodings.pickle")
        if enc_path.exists():
            with open(enc_path, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]

    def draw_hud(self, frame, status, color, score, present_students, session_duration_sec):
        # --- 1. Top Bar (Status) ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FOCUS: {int(score)}%", (frame.shape[1]-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- 2. Top-Left Floating Panel (Attendance) ---
        # Draw a semi-transparent box for the list
        panel_h = 40 + (len(present_students) * 30)
        panel_w = 250
        panel_x, panel_y = 10, 70
        
        sub_overlay = frame.copy()
        cv2.rectangle(sub_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 50, 50), -1)
        cv2.addWeighted(sub_overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "In Class:", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        for idx, (name, stats) in enumerate(present_students.items()):
            y_pos = panel_y + 55 + (idx * 30)
            
            # Calculate % of class attended so far
            # Using session_duration_sec as the 'total expected' for real-time progress calc
            # Or use fixed threshold logic:
            is_present = stats['duration'] >= (session_duration_sec * 0.3)
            
            icon = " [P]" if is_present else " [ ]"
            text_color = (0, 255, 0) if is_present else (150, 150, 150)
            
            cv2.putText(frame, f"{name}{icon}", (panel_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    def track_focus_session(self, session_name: str, duration_min: int):
        cap = cv2.VideoCapture(0)
        # Set Resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        start_time = datetime.now()
        expected_duration_sec = duration_min * 60
        
        # Tracking Variables
        frame_count = 0
        history = []
        
        # Phone Logic
        phone_episodes = []
        phone_start_time = None 
        
        # Focus Logic
        distraction_episodes = []
        is_distracted = False
        distraction_start_time = None
        last_alert_time = None
        
        # Attendance Logic
        active_students = {} # {name: {'first_seen': time, 'last_seen': time, 'duration': 0}}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > expected_duration_sec: break
            
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(elapsed * 1000)
            
            # --- 1. Face Recognition (Run every 30 frames/1 sec to save CPU) ---
            if frame_count % 30 == 0:
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
                faces = face_recognition.face_locations(small_frame)
                encodings = face_recognition.face_encodings(small_frame, faces)
                
                current_names = []
                for encoding in encodings:
                    matches = face_recognition.compare_faces(self.known_encodings, encoding)
                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_names[first_match_index]
                    current_names.append(name)
                
                # Update Attendance Durations
                for name in current_names:
                    if name not in active_students:
                        active_students[name] = {'duration': 0}
                    active_students[name]['duration'] += 1 # Add 1 second roughly

            # --- 2. AI Detections ---
            face_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            obj_result = self.object_detector.detect_for_video(mp_image, timestamp_ms)
            
            # Calculate Focus Score
            score = 0
            if face_result.face_landmarks:
                nose = face_result.face_landmarks[0][1]
                dist = np.sqrt((nose.x - 0.5)**2 + (nose.y - 0.5)**2)
                score = max(0, 100 - (dist * 450))

            # Check Phones
            phone_visible = False
            for detection in obj_result.detections:
                bbox = detection.bounding_box
                cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE", (bbox.origin_x, bbox.origin_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                phone_visible = True

            # --- 3. Episode Logic (Phone) ---
            if phone_visible and phone_start_time is None:
                phone_start_time = datetime.now() # Start Episode
            elif not phone_visible and phone_start_time is not None:
                # End Episode
                p_dur = (datetime.now() - phone_start_time).total_seconds()
                if p_dur > 1.0: # Filter flickers
                    phone_episodes.append({
                        "start_time": phone_start_time.strftime("%H:%M:%S"),
                        "end_time": datetime.now().strftime("%H:%M:%S"),
                        "duration": round(p_dur, 1)
                    })
                phone_start_time = None

            # --- 4. Distraction & Alert Logic ---
            # Phone is now just another cause for distraction state
            # Trigger if Score is low OR Phone is visible
            current_distraction_status = (score <= 15) or phone_visible
            
            if current_distraction_status:
                status = "DISTRACTED"
                color = (0, 0, 255)
                
                if not is_distracted:
                    is_distracted = True
                    distraction_start_time = datetime.now()
                
                # Unified Alert Timer (Same for Phone or Gaze)
                curr_dist_dur = (datetime.now() - distraction_start_time).total_seconds()
                
                # Alert after 10s initially, then every 5s
                if curr_dist_dur >= 10:
                    if last_alert_time is None or (datetime.now() - last_alert_time).total_seconds() >= 5:
                        try:
                            winsound.Beep(1000, 500) 
                            last_alert_time = datetime.now()
                        except: pass
            else:
                status = "FOCUSED"
                color = (0, 255, 0)
                if is_distracted:
                    # Log Gaze Distraction Episode
                    d_dur = (datetime.now() - distraction_start_time).total_seconds()
                    if d_dur > 5:
                        distraction_episodes.append({
                            "type": "Gaze/Phone",
                            "start": distraction_start_time.strftime("%H:%M:%S"),
                            "duration": round(d_dur, 1)
                        })
                    is_distracted = False
                    distraction_start_time = None
                    last_alert_time = None

            # --- 5. Draw HUD ---
            self.draw_hud(frame, status, color, score, active_students, expected_duration_sec)
            
            cv2.imshow("Smart Classroom Manager", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

        # Save Focus Report
        if phone_start_time: # Close any open phone episode
            p_dur = (datetime.now() - phone_start_time).total_seconds()
            phone_episodes.append({"start_time": phone_start_time.strftime("%H:%M:%S"), "duration": round(p_dur, 1)})

        report = {
            "session": session_name,
            "date": str(datetime.now().date()),
            "phone_episodes": phone_episodes,
            "distraction_episodes": distraction_episodes,
            "data": history}