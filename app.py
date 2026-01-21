# filename: app.py
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import cv2
import face_recognition
import numpy as np

# Import Core Logic
from core.focus_tracker import FocusTracker
from core.student_registration import delete_student_completely, delete_session_log, get_all_students

# --- CONFIG ---
st.set_page_config(page_title="Smart Classroom", layout="wide")

# Paths
DATA_DIR = Path("data")
PROFILE_DIR = DATA_DIR / "profiles"
ATTENDANCE_DIR = DATA_DIR / "attendance"
FOCUS_DIR = DATA_DIR / "focus_logs"
PHOTOS_DIR = DATA_DIR / "photos"
ENC_FILE = DATA_DIR / "encodings/global_encodings.json"

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Classroom Admin")
    st.markdown("---")
    page = st.radio("Navigation", 
                    ["Overview", "Register Student", "Class Session", "Class Analytics", "Students", "Sessions"])

# --- PAGE: OVERVIEW ---
if page == "Overview":
    st.title("System Overview")
    s_count = len(list(PROFILE_DIR.glob("*.json"))) if PROFILE_DIR.exists() else 0
    sess_count = len(list(FOCUS_DIR.glob("*.json"))) if FOCUS_DIR.exists() else 0
    
    col1, col2 = st.columns(2)
    col1.metric("Registered Students", s_count)
    col2.metric("Total Sessions", sess_count)

# --- PAGE: REGISTER STUDENT ---
elif page == "Register Student":
    st.title("Student Registration")
    s_id = st.text_input("Student ID")
    name = st.text_input("Full Name")
    img_file = st.camera_input("Take Student Photo")
    
    if st.button("Save Student Record"):
        if s_id and name and img_file:
            try:
                photo_path = PHOTOS_DIR / f"{s_id}.jpg"
                photo_path.parent.mkdir(parents=True, exist_ok=True)
                photo_path.write_bytes(img_file.getbuffer())
                
                img = cv2.imdecode(np.frombuffer(img_file.getbuffer(), np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                
                if encs:
                    enc = encs[0].tolist()
                    ENC_FILE.parent.mkdir(parents=True, exist_ok=True)
                    enc_data = json.loads(ENC_FILE.read_text()) if ENC_FILE.exists() else {}
                    enc_data[s_id] = enc
                    ENC_FILE.write_text(json.dumps(enc_data))
                    
                    profile = {"id": s_id, "name": name, "date": str(datetime.now().date())}
                    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
                    (PROFILE_DIR / f"{s_id}.json").write_text(json.dumps(profile))
                    st.success(f"Successfully registered {name}")
                else:
                    st.warning("No face detected in photo. Please retake the photo.")
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
        else:
            st.error("Please provide ID, Name, and Photo")

# --- PAGE: CLASS SESSION ---
elif page == "Class Session":
    st.title("Start Session")
    with st.container():
        sess_name = st.text_input("Session Name", placeholder="Enter a unique session name...")
        duration = st.number_input("Duration (minutes)", 1, 180, 45)
        
        if st.button("Launch Camera & Track"):
            if not sess_name.strip():
                st.error("Please enter a Session Name before launching.")
            else:
                try:
                    tracker = FocusTracker()
                    with st.spinner("Session in progress... Press 'q' in the camera window to stop."):
                        tracker.track_focus_session(sess_name, duration)
                    st.success("Session concluded.")
                except Exception as e:
                    st.error(f"Session error: {str(e)}")

# --- PAGE: CLASS ANALYTICS ---
# --- PAGE: CLASS ANALYTICS ---
elif page == "Class Analytics":
    st.title("Analytics & Reports")
    
    f_files = list(FOCUS_DIR.glob("*.json")) if FOCUS_DIR.exists() else []
        
    if f_files:
        choice = st.selectbox("Select Session", [f.name for f in f_files])
        data = json.loads((FOCUS_DIR / choice).read_text())
        
        # Session Summary
        st.subheader("📊 Session Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Session Name", data.get('session', 'N/A'))
        col2.metric("Date", data.get('date', 'N/A'))
        col3.metric("Duration (min)", data.get('duration_min', 'N/A'))
        
        st.markdown("---")
        
        # Student Performance Table
        students = data.get('students', {})
        if students:
            st.subheader("👥 Student Attendance & Performance")
            report_data = []

            for sid, info in students.items():
                report_data.append({
                    "Student ID": sid,
                    "Name": info['name'],
                    "Presence Duration (s)": info.get('total_seen_duration', 0),
                    "Status": info.get('attendance_status', 'Absent'),
                    "Focus Time (s)": info.get('total_focus_duration', 0),
                    "Distracted Time (s)": info.get('total_distracted_duration', 0)
                })
            
            df = pd.DataFrame(report_data)
            st.table(df)
        
        st.markdown("---")
        
        # Distraction Episodes Section
        distraction_episodes = data.get('distraction_episodes', [])
        if distraction_episodes:
            st.subheader("⚠️ Distraction Episodes (5+ seconds)")
            st.write(f"Total Episodes: **{len(distraction_episodes)}**")
            
            # Create DataFrame for distraction episodes
            episodes_data = []
            for episode in distraction_episodes:
                episodes_data.append({
                    "Student Name": episode.get('student_name', 'Unknown'),
                    "Student ID": episode.get('student_id', 'N/A'),
                    "Start Time": episode.get('start_time', 'N/A'),
                    "End Time": episode.get('end_time', 'N/A'),
                    "Duration (s)": episode.get('duration_sec', 0),
                    "Distraction Type": episode.get('distraction_type', 'Unknown')
                })
            
            episodes_df = pd.DataFrame(episodes_data)
            st.dataframe(episodes_df, use_container_width=True)
            
            # Summary statistics
            st.markdown("#### Episode Summary")
            col1, col2, col3 = st.columns(3)
            
            total_duration = sum(ep.get('duration_sec', 0) for ep in distraction_episodes)
            phone_episodes = sum(1 for ep in distraction_episodes if ep.get('distraction_type') == 'Phone')
            head_turn_episodes = sum(1 for ep in distraction_episodes if ep.get('distraction_type') == 'Head Turn')
            
            col1.metric("Total Distraction Time", f"{total_duration:.1f}s")
            col2.metric("Phone Distractions", phone_episodes)
            col3.metric("Head Turn Distractions", head_turn_episodes)
        else:
            st.info("No distraction episodes recorded (all distractions were under 5 seconds)")
    else:
        st.info("No session logs available. Start a class session to generate reports.")


# --- PAGE: STUDENTS (MANAGEMENT) ---
elif page == "Students":
    st.title("Student Directory")
    
    students_ids = get_all_students()
    
    if not students_ids:
        st.info("No students registered.")
    else:
        st.subheader("Registered Students")
        
        for s_id in students_ids:
            try:
                prof_path = PROFILE_DIR / f"{s_id}.json"
                prof = json.loads(prof_path.read_text())
                s_name = prof.get("name", "Unknown")
            except:
                s_name = "Error Loading Profile"

            with st.expander(f"👤 {s_name} (ID: {s_id})"):
                col1, col2 = st.columns([3, 1])
                col1.write(f"**Registered:** {prof.get('date', 'N/A')}")
                
                if col2.button("Delete Student", key=f"del_btn_{s_id}"):
                    st.session_state[f"confirm_del_{s_id}"] = True
                
                if st.session_state.get(f"confirm_del_{s_id}"):
                    st.warning("Are you sure? This cannot be undone.")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("Yes, Delete", key=f"yes_{s_id}"):
                        delete_student_completely(s_id)
                        del st.session_state[f"confirm_del_{s_id}"]
                        st.success(f"Deleted {s_name}")
                        st.rerun()
                    if c_no.button("No, Cancel", key=f"no_{s_id}"):
                        del st.session_state[f"confirm_del_{s_id}"]
                        st.rerun()

# --- PAGE: SESSIONS (RECORDS) ---
elif page == "Sessions":
    st.title("Session History")
    
    focus_logs = list(FOCUS_DIR.glob("*.json")) if FOCUS_DIR.exists() else []
    
    if not focus_logs:
        st.info("No logs available.")
    else:
        log_to_del = st.selectbox("Select a session file to delete", [f.name for f in focus_logs])
        
        if st.button("Delete Session Log"):
             st.session_state["del_sess_active"] = True
        
        if st.session_state.get("del_sess_active"):
            st.warning(f"Permanently delete {log_to_del}?")
            cy, cn = st.columns([1, 6])
            if cy.button("Yes", key="sess_yes"):
                try:
                    delete_session_log(log_to_del, "focus")
                    st.session_state["del_sess_active"] = False
                    st.success("Deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            if cn.button("No", key="sess_no"):
                st.session_state["del_sess_active"] = False
                st.rerun()