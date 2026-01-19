import streamlit as st
import pandas as pd
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.express as px
from PIL import Image

# --- Core Imports ---
from core.focus_tracker import FocusTracker

# --- Page Config ---
st.set_page_config(page_title="EduVision AI", page_icon="🎓", layout="wide")

# --- Directory Setup ---
DIRS = ["data/profiles", "data/photos", "data/attendance", "data/focus_logs", "data/encodings"]
for d in DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)

# --- Custom Theme ---
st.markdown("""
    <style>
    .stApp { background-color: #FBFBFE; }
    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .metric-card {
        background: #FFFFFF;
        border-left: 5px solid #4F46E5;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=80)
    st.sidebar.title("EduVision AI")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox("Go to", 
        ["Dashboard", "Live Class Monitor", "Student Enrollment", "Attendance Logs", "Focus Analytics"])

    # ---------------------------------------------------------
    # DASHBOARD
    # ---------------------------------------------------------
    if page == "Dashboard":
        st.title("🏫 Smart Classroom Overview")
        c1, c2, c3 = st.columns(3)
        
        total_students = len(list(Path("data/profiles").glob("*.json")))
        total_sessions = len(list(Path("data/attendance").glob("*.json")))
        
        with c1: st.metric("Enrolled Students", total_students)
        with c2: st.metric("Classes Held", total_sessions)
        with c3: st.metric("System Status", "Ready", delta="Healthy")
        
        st.markdown("### Recent Activity")
        st.info("The system is optimized for EfficientDet-Lite0 and 720p Video Input.")
        

    # ---------------------------------------------------------
    # STUDENT ENROLLMENT (Webcam Photo)
    # ---------------------------------------------------------
    elif page == "Student Enrollment":
        st.title("👤 Student Enrollment")
        st.write("Complete the form and capture a clear photo to register.")
        
        with st.container():
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name", placeholder="John Doe")
                student_id = st.text_input("Student ID", placeholder="STU-001")
                email = st.text_input("Email Address", placeholder="john@university.edu")
            
            with col2:
                # NEW: Camera Input instead of File Uploader
                img_file = st.camera_input("Take Enrollment Photo")
            
            if st.button("Register Student", type="primary", use_container_width=True):
                if name and student_id and email and img_file:
                    # Save Metadata
                    meta = {"name": name, "id": student_id, "email": email, "date": str(datetime.now())}
                    Path(f"data/profiles/{student_id}.json").write_text(json.dumps(meta))
                    
                    # Save Image
                    img = Image.open(img_file)
                    img.save(f"data/photos/{student_id}.jpg")
                    
                    st.success(f"Successfully registered {name}!")
                    st.balloons()
                else:
                    st.error("Missing information or photo. Please try again.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # LIVE CLASS MONITOR
    # ---------------------------------------------------------
    elif page == "Live Class Monitor":
        st.title("📡 Live Class Session")
        
        with st.sidebar:
            st.subheader("Session Settings")
            session_id = st.text_input("Session Code", f"BIO_{datetime.now().strftime('%H%M')}")
            duration = st.slider("Duration (mins)", 5, 120, 45)
        
        st.warning("Ensure lighting is adequate for face recognition.")
        
        if st.button("🔴 Start Live Monitoring", use_container_width=True):
            try:
                tracker = FocusTracker()
                tracker.track_focus_session(session_id, duration)
                st.success("Session completed. Data synced.")
            except Exception as e:
                st.error(f"Hardware Error: {e}")

    # ---------------------------------------------------------
    # ATTENDANCE LOGS
    # ---------------------------------------------------------
    elif page == "Attendance Logs":
        st.title("📋 Attendance History")
        logs = list(Path("data/attendance").glob("*.json"))
        
        if logs:
            selected = st.selectbox("Select Session", [f.stem for f in logs])
            data = json.loads(Path(f"data/attendance/{selected}.json").read_text())
            
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index.name = "Student ID"
            
            # Display metrics
            present = len(df[df['status'] == 'Present'])
            st.subheader(f"Session Summary: {present}/{len(df)} Present")
            
            st.table(df.style.applymap(
                lambda x: 'color: green' if x == 'Present' else 'color: red', subset=['status']
            ))
        else:
            st.info("No records found.")

    # ---------------------------------------------------------
    # FOCUS ANALYTICS
    # ---------------------------------------------------------
    elif page == "Focus Analytics":
        st.title("🧠 Engagement Insights")
        logs = list(Path("data/focus_logs").glob("*.json"))
        
        if logs:
            selected = st.selectbox("Select Analysis Report", [f.stem for f in logs])
            data = json.loads(Path(f"data/focus_logs/{selected}.json").read_text())
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Avg Focus Score", f"{data.get('average_focus_score', 0)}%")
            with c2:
                phone_count = data.get("phone_incidents_count", 0)
                st.metric("Phone Violations", phone_count, delta="Alert" if phone_count > 0 else "None", delta_color="inverse")
            
            # Focus Chart
            history = data.get("data", [])
            if history:
                df_chart = pd.DataFrame(history)
                fig = px.line(df_chart, x="time", y="score", title="Session Engagement Curve")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No focus data available.")

if __name__ == "__main__":
    main()