# filename: student_registration.py
import os
import json
from pathlib import Path
import shutil

# Directories
DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
PHOTOS_DIR = DATA_DIR / "photos"
ATTENDANCE_DIR = DATA_DIR / "attendance"
FOCUS_DIR = DATA_DIR / "focus_logs"
ENC_FILE = DATA_DIR / "encodings/global_encodings.json"

def get_all_students():
    """Returns list of student IDs based on profiles."""
    if not PROFILES_DIR.exists(): return []
    return [f.stem for f in PROFILES_DIR.glob("*.json")]

def delete_student_completely(student_id):
    """
    Removes student profile, photo, encoding, and scrubs them from ALL attendance and focus logs.
    Does NOT delete the log files themselves, just the student's entries inside them.
    """
    # 1. Delete Profile & Photo
    profile_path = PROFILES_DIR / f"{student_id}.json"
    photo_path = PHOTOS_DIR / f"{student_id}.jpg"
    
    if profile_path.exists(): os.remove(profile_path)
    if photo_path.exists(): os.remove(photo_path)
    
    # 2. Remove from encodings
    if ENC_FILE.exists():
        try:
            enc_data = json.loads(ENC_FILE.read_text())
            if student_id in enc_data:
                del enc_data[student_id]
                ENC_FILE.write_text(json.dumps(enc_data, indent=2))
        except Exception as e:
            print(f"Error updating encodings: {e}")
    
    # 3. Scrub from Attendance Logs
    if ATTENDANCE_DIR.exists():
        for log_file in ATTENDANCE_DIR.glob("*.json"):
            try:
                data = json.loads(log_file.read_text())
                if 'records' in data and student_id in data['records']:
                    del data['records'][student_id]
                    log_file.write_text(json.dumps(data, indent=2))
            except Exception as e:
                print(f"Error scrubbing attendance {log_file}: {e}")

    # 4. Scrub from Focus Logs
    if FOCUS_DIR.exists():
        for log_file in FOCUS_DIR.glob("*.json"):
            try:
                data = json.loads(log_file.read_text())
                if 'students' in data and student_id in data['students']:
                    del data['students'][student_id]
                    log_file.write_text(json.dumps(data, indent=2))
            except Exception as e:
                print(f"Error scrubbing focus {log_file}: {e}")
        
    return True

def delete_session_log(filename, type="attendance"):
    """Deletes a specific session file."""
    if type == "attendance":
        target = ATTENDANCE_DIR / filename
    else:
        target = FOCUS_DIR / filename
        
    if target.exists():
        os.remove(target)
        return True
    return False