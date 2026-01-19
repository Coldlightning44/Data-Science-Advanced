import face_recognition
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import os

class StudentRegistration:
    def __init__(self):
        self.base_data = Path("data")
        self.profiles_dir = self.base_data / "profiles"
        self.photos_dir = self.base_data / "photos"
        self.encodings_dir = self.base_data / "encodings"
        self.encodings_file = self.encodings_dir / "global_encodings.json"
        
        # Log directories for purging
        self.att_dir = self.base_data / "attendance"
        self.foc_dir = self.base_data / "focus_logs"

        # Ensure all directories exist
        for folder in [self.profiles_dir, self.photos_dir, self.encodings_dir, self.att_dir, self.foc_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def _load_encodings(self):
        if not self.encodings_file.exists(): return {}
        try:
            data = json.loads(self.encodings_file.read_text())
            return {k: np.array(v) for k, v in data.items()}
        except: return {}

    def register_student(self, student_id, name, email, photo_path):
        if (self.profiles_dir / f"{student_id}.json").exists():
            return False, f"ID {student_id} already exists."

        try:
            image = face_recognition.load_image_file(photo_path)
            encs = face_recognition.face_encodings(image)
            if not encs: return False, "No face detected in the photograph."

            profile = {
                "id": student_id, "name": name, "email": email,
                "photo_path": photo_path, "registered_at": datetime.now().isoformat()
            }
            (self.profiles_dir / f"{student_id}.json").write_text(json.dumps(profile, indent=2))

            # Update global face database
            current_encs = self._load_encodings()
            current_encs[student_id] = encs[0].tolist()
            self.encodings_file.write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                                       for k, v in current_encs.items()}, indent=2))
            return True, f"Student {name} registered successfully."
        except Exception as e:
            return False, str(e)

    def delete_student(self, student_id):
        """Removes student files and cleans up all historical logs."""
        # 1. Remove Profile and Photo
        p_path = self.profiles_dir / f"{student_id}.json"
        img_path = self.photos_dir / f"{student_id}.jpg"
        if p_path.exists(): os.remove(p_path)
        if img_path.exists(): os.remove(img_path)

        # 2. Update Global Encodings
        if self.encodings_file.exists():
            data = json.loads(self.encodings_file.read_text())
            if student_id in data:
                del data[student_id]
                self.encodings_file.write_text(json.dumps(data, indent=2))

        # 3. Purge Logs
        self._purge_attendance(student_id)
        self._purge_focus(student_id)
        return True

    def _purge_attendance(self, student_id):
        for log_file in self.att_dir.glob("*.json"):
            data = json.loads(log_file.read_text())
            records = data.get("records", {})
            if student_id in records:
                if len(records) <= 1:
                    os.remove(log_file) # Delete file if they were the only participant
                else:
                    del records[student_id] # Remove only their entry
                    log_file.write_text(json.dumps(data, indent=2))

    def _purge_focus(self, student_id):
        for log_file in self.foc_dir.glob("*.json"):
            data = json.loads(log_file.read_text())
            if data.get("student_id") == student_id:
                os.remove(log_file)