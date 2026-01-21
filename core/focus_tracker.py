import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from datetime import datetime
import json
import time
import math
import face_recognition
from collections import deque

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class FocusTracker:
    def __init__(self):
        self.log_dir = Path("data/focus_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("data")
        self.profiles_dir = self.data_dir / "profiles"
        self.enc_file = self.data_dir / "encodings/global_encodings.json"
        
        data = json.loads(self.enc_file.read_text()) if self.enc_file.exists() else {}
        self.known_ids = list(data.keys())
        self.known_encs = [np.array(v) for v in data.values()]

        face_model_path = Path("models/face_landmarker.task")
        if not face_model_path.exists():
            raise FileNotFoundError(f"Face landmarker model not found at {face_model_path}")

        face_base_opts = python.BaseOptions(model_asset_path=str(face_model_path))
        face_opts = vision.FaceLandmarkerOptions(
            base_options=face_base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(face_opts)
        
        self.phone_detector = None
        obj_model_path = Path("models/efficientdet_lite0.tflite") 
        if obj_model_path.exists():
            obj_base_opts = python.BaseOptions(model_asset_path=str(obj_model_path))
            obj_opts = vision.ObjectDetectorOptions(
                base_options=obj_base_opts,
                score_threshold=0.45,
                category_allowlist=['cell phone', 'mobile phone'] 
            )
            self.phone_detector = vision.ObjectDetector.create_from_options(obj_opts)
        else:
            print("Warning: 'efficientdet_lite0.tflite' not found. Phone detection disabled.")
            
        self.last_detected_phones = []
        self.phone_persistence_counter = 0

    def get_name(self, sid):
        try:
            return json.loads((self.profiles_dir / f"{sid}.json").read_text())["name"]
        except:
            return None

    def get_bbox_from_landmarks(self, landmarks, image_width, image_height):
        min_x = min(lm.x for lm in landmarks)
        min_y = min(lm.y for lm in landmarks)
        max_x = max(lm.x for lm in landmarks)
        max_y = max(lm.y for lm in landmarks)
        margin = 0.2
        width = max_x - min_x
        height = max_y - min_y
        min_x = max(0, min_x - width * margin)
        min_y = max(0, min_y - height * margin)
        max_x = min(1, max_x + width * margin)
        max_y = min(1, max_y + height * margin)
        left = int(min_x * image_width)
        top = int(min_y * image_height)
        right = int(max_x * image_width)
        bottom = int(max_y * image_height)
        return (top, right, bottom, left)

    def matrix_to_euler(self, matrix):
        matrix = np.array(matrix)
        R = matrix[:3, :3]
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = -math.asin(np.clip(R[2, 0], -1.0, 1.0))
        roll = math.atan2(R[2, 1], R[2, 2])
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

    def play_alert(self):
        if HAS_WINSOUND:
            winsound.Beep(1000, 300) 
        else:
            print('\a') 

    def track_focus_session(self, session_name: str, duration_min: int):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        start_time = datetime.now()
        attendance_threshold_sec = (duration_min * 60) * 0.30
        
        active_faces = {}
        display_faces = {}
        
        unknown_buffer = deque(maxlen=15)
        smoothed_unknowns = 0

        last_alert_time = 0
        last_frame_time = time.time()
        
        frame_count = 0
        face_detection_skip = 5  # Face/recognition runs every 5th frame
        phone_detection_skip = 1  # Phone detection runs every frame (no skip)
        
        cached_face_data = []
        cached_matrices = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_count += 1
                curr_time_obj = datetime.now()
                elapsed = (curr_time_obj - start_time).total_seconds()
                
                if elapsed > duration_min * 60: break
                remaining = max(0, duration_min * 60 - elapsed)

                current_time = time.time()
                dt = current_time - last_frame_time
                last_frame_time = current_time

                height, width, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(elapsed * 1000)

                # ===============================
                # PHONE DETECTION (EVERY FRAME - NO SKIP)
                # ===============================
                phones_detected = False
                current_phone_boxes = []
                
                run_phone_detection = (frame_count % phone_detection_skip == 0)
                
                if run_phone_detection and self.phone_detector:
                    det_res = self.phone_detector.detect(mp_img)
                    if det_res.detections:
                        phones_detected = True
                        self.phone_persistence_counter = 5
                        for detection in det_res.detections:
                            bbox = detection.bounding_box
                            current_phone_boxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))
                        self.last_detected_phones = current_phone_boxes
                    elif self.phone_persistence_counter > 0:
                        phones_detected = True
                        self.phone_persistence_counter -= 1
                        current_phone_boxes = self.last_detected_phones
                    else:
                        self.last_detected_phones = []

                # ===============================
                # FACE DETECTION + RECOGNITION (SKIPPED FRAMES)
                # ===============================
                run_face_detection = (frame_count % face_detection_skip == 0)

                if run_face_detection:
                    face_res = self.landmarker.detect_for_video(mp_img, ts_ms)
                    landmarks = face_res.face_landmarks or []
                    matrices = face_res.facial_transformation_matrixes or []
                    bboxes = [self.get_bbox_from_landmarks(lm, width, height) for lm in landmarks]

                    face_encs = []
                    if bboxes:
                        small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
                        small_bboxes = [(int(t*0.5), int(r*0.5), int(b*0.5), int(l*0.5)) for (t, r, b, l) in bboxes]
                        face_encs = face_recognition.face_encodings(small_rgb, small_bboxes, model='small')

                    curr_unknowns = 0
                    new_face_data = []

                    for i, bbox in enumerate(bboxes):
                        sid = None
                        name = None

                        if i < len(face_encs):
                            distances = face_recognition.face_distance(self.known_encs, face_encs[i])
                            best_match_idx = np.argmin(distances)
                            if distances[best_match_idx] < 0.6:
                                sid = self.known_ids[best_match_idx]
                                name = self.get_name(sid)

                        if not sid:
                            curr_unknowns += 1
                            continue

                        new_face_data.append((sid, name, i, i))

                    cached_face_data = new_face_data
                    cached_matrices = matrices
                    unknown_buffer.append(curr_unknowns)
                    smoothed_unknowns = int(np.mean(unknown_buffer)) if unknown_buffer else 0
                else:
                    # Use cached matrices for pose estimation
                    matrices = cached_matrices

                # ===============================
                # LOGIC PHASE - UPDATE FACE STATES
                # ===============================
                for sid, name, bbox_idx, matrix_idx in cached_face_data:
                    if sid not in active_faces:
                        active_faces[sid] = {
                            "name": name,
                            "status": "Focused",
                            "dist_start": None,
                            "total_seen": 0.0,
                            "total_focus": 0.0,
                            "total_dist": 0.0,
                            "episodes": [],
                            "last_seen_ts": time.time()
                        }

                    st = active_faces[sid]
                    st["last_seen_ts"] = time.time()
                    st["total_seen"] += dt

                    is_phone_dist = phones_detected
                    is_pose_dist = False
                    
                    if run_face_detection and matrix_idx < len(matrices):
                        y, p, r = self.matrix_to_euler(matrices[matrix_idx])
                        if abs(y) > 25 or abs(p) > 15:
                            is_pose_dist = True
                    elif not run_face_detection and st.get("last_pose_dist"):
                        is_pose_dist = st["last_pose_dist"]

                    current_distracted = is_phone_dist or is_pose_dist
                    dist_type = "Phone" if is_phone_dist else ("Head Turn" if is_pose_dist else "None")
                    
                    st["last_pose_dist"] = is_pose_dist

                    if current_distracted:
                        st["total_dist"] += dt
                        if st["status"] == "Focused":
                            st["status"] = "Distracted"
                            st["dist_start"] = datetime.now()
                            st["dist_type"] = dist_type

                        if st["dist_start"]:
                            dur_so_far = (datetime.now() - st["dist_start"]).total_seconds()
                            if dur_so_far >= 5 and time.time() - last_alert_time > 5:
                                self.play_alert()
                                last_alert_time = time.time()
                    else:
                        st["total_focus"] += dt
                        if st["status"] == "Distracted":
                            if st["dist_start"]:
                                s_dt = st["dist_start"]
                                e_dt = datetime.now()
                                dur = (e_dt - s_dt).total_seconds()
                                if dur >= 5:
                                    st["episodes"].append({
                                        "start_time": s_dt.strftime("%H:%M:%S"),
                                        "end_time": e_dt.strftime("%H:%M:%S"),
                                        "duration_sec": round(dur, 1),
                                        "type": st.get("dist_type", "General")
                                    })
                            st["status"] = "Focused"
                            st["dist_start"] = None

                    is_present = st["total_seen"] >= attendance_threshold_sec

                    display_faces[sid] = {
                        "name": name,
                        "focus": st["status"] == "Focused",
                        "present": is_present,
                        "last_seen": time.time()
                    }

                # Remove faces not seen in 5 seconds from display
                now = time.time()
                for sid in list(display_faces.keys()):
                    if now - display_faces[sid]["last_seen"] > 5:
                        del display_faces[sid]

                # ===============================
                # DISPLAY PHASE
                # ===============================
                display_frame = frame.copy()
                
                for (px, py, pw, ph) in current_phone_boxes:
                    start_point = (px, py)
                    end_point = (px + pw, py + ph)
                    cv2.rectangle(display_frame, start_point, end_point, (0, 0, 255), 2)
                    cv2.putText(display_frame, "PHONE", (px, py - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                self.draw_overlay(display_frame, display_faces, smoothed_unknowns, elapsed, remaining, attendance_threshold_sec)
                cv2.imshow("Smart Classroom - Class Session", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_logs(session_name, duration_min, active_faces, attendance_threshold_sec)

    def draw_overlay(self, frame, display_faces, unknowns, elapsed, remaining, thresh_sec):
        overlay = frame.copy()
        card_width = 320
        line_height = 25
        card_height = 80 + max(len(display_faces), 1)*line_height + 50
        
        cv2.rectangle(overlay, (10, 10), (10+card_width, 10+card_height), (30,30,30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        SCALE = 0.5
        THICK = 1
        WHITE = (255, 255, 255)

        cv2.putText(frame, "LIVE MONITOR", (20, 35), FONT, 0.6, WHITE, 2)
        
        cv2.circle(frame, (200, 30), 5, (0,255,0), -1)
        cv2.putText(frame, "Pres", (210, 35), FONT, 0.4, (200,200,200), 1)
        cv2.circle(frame, (260, 30), 5, (0,0,255), -1)
        cv2.putText(frame, "Abs", (270, 35), FONT, 0.4, (200,200,200), 1)

        y = 65
        for sid, data in display_faces.items():
            col = (0, 255, 0) if data["present"] else (0, 0, 255)
            cv2.circle(frame, (25, y-5), 6, col, -1)
            cv2.putText(frame, data["name"], (45, y), FONT, SCALE, WHITE, THICK)
            
            f_txt = "FOCUSED" if data["focus"] else "DISTRACTED"
            f_col = (0, 255, 0) if data["focus"] else (0, 0, 255)
            cv2.putText(frame, f_txt, (210, y), FONT, 0.45, f_col, 1)
            y += line_height
            
        y += 15
        cv2.line(frame, (20, y), (300, y), (100,100,100), 1)
        y += 25
        cv2.putText(frame, f"Unknowns: {unknowns}", (20, y), FONT, SCALE, (200, 200, 255), 1)
        y += 25
        mins, secs = divmod(int(remaining), 60)
        cv2.putText(frame, f"Time Left: {mins:02}:{secs:02}", (20, y), FONT, SCALE, (0, 255, 255), 1)

    def save_logs(self, session_name, duration_min, active_faces, thresh_sec):
        final_data = {
            "session": session_name,
            "date": str(datetime.now().date()),
            "duration_min": duration_min,
            "attendance_threshold_seconds": thresh_sec,
            "students": {},
            "distraction_episodes": []
        }
        
        for sid, st in active_faces.items():
            # Finalize any ongoing distraction at session end
            if st["status"] == "Distracted" and st["dist_start"]:
                s_dt = st["dist_start"]
                e_dt = datetime.now()
                dur = (e_dt - s_dt).total_seconds()
                if dur >= 5:
                    st["episodes"].append({
                        "start_time": s_dt.strftime("%H:%M:%S"),
                        "end_time": e_dt.strftime("%H:%M:%S"),
                        "duration_sec": round(dur, 1),
                        "type": st.get("dist_type", "General")
                    })

            attendance_status = "Present" if st["total_seen"] >= thresh_sec else "Absent"
            
            final_data["students"][sid] = {
                "name": st["name"],
                "total_seen_duration": round(st["total_seen"], 1),
                "total_focus_duration": round(st["total_focus"], 1),
                "total_distracted_duration": round(st["total_dist"], 1),
                "attendance_status": attendance_status,
                "distraction_episodes": st["episodes"]
            }
            
            # Add each episode to the global distraction episodes list
            for episode in st["episodes"]:
                final_data["distraction_episodes"].append({
                    "student_id": sid,
                    "student_name": st["name"],
                    "start_time": episode["start_time"],
                    "end_time": episode["end_time"],
                    "duration_sec": episode["duration_sec"],
                    "distraction_type": episode["type"]
                })
        
        # Sort distraction episodes by start time
        final_data["distraction_episodes"].sort(key=lambda x: x["start_time"])
            
        (self.log_dir / f"{session_name}.json").write_text(json.dumps(final_data, indent=2))