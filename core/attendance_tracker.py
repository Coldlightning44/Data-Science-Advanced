import cv2
import face_recognition
import json
import numpy as np
from pathlib import Path
from datetime import datetime

class AttendanceTracker:
    def __init__(self):
        self.data_dir = Path("data")
        self.profiles_dir = self.data_dir / "profiles"
        self.enc_file = self.data_dir / "encodings/global_encodings.json"
        self.att_dir = self.data_dir / "attendance"
        
        data = json.loads(self.enc_file.read_text()) if self.enc_file.exists() else {}
        self.known_ids = list(data.keys())
        self.known_encs = [np.array(v) for v in data.values()]

    def start_session(self, session_name):
        cap = cv2.VideoCapture(0)
        start_time = datetime.now()
        session_stats = {} # {id: frames_seen}
        total_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            total_frames += 1
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb)
            face_encs = face_recognition.face_encodings(rgb, face_locs)

            for loc, enc in zip(face_locs, face_encs):
                matches = face_recognition.compare_faces(self.known_encs, enc, 0.5)
                name = "Unknown"
                if True in matches:
                    idx = matches.index(True)
                    sid = self.known_ids[idx]
                    session_stats[sid] = session_stats.get(sid, 0) + 1
                    try:
                        name = json.loads((self.profiles_dir / f"{sid}.json").read_text())['name']
                    except: name = sid
                
                top, right, bottom, left = loc
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Attendance Tracker (30% Rule)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

        # Finalize Logic: Mark Present if visibility >= 30%
        final_records = {}
        for sid, seen in session_stats.items():
            presence_rate = (seen / total_frames) if total_frames > 0 else 0
            final_records[sid] = {
                "name": json.loads((self.profiles_dir / f"{sid}.json").read_text())['name'],
                "status": "Present" if presence_rate >= 0.30 else "Absent (Low Presence)",
                "presence_percentage": round(presence_rate * 100, 1)
            }
        
        log_data = {
            "session": session_name,
            "date": str(datetime.now().date()),
            "duration_frames": total_frames,
            "records": final_records
        }
        (self.att_dir / f"{session_name}_{datetime.now():%H%M}.json").write_text(json.dumps(log_data, indent=2))