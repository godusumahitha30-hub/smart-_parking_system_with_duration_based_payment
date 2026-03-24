import streamlit as st
import cv2
import tempfile
import pandas as pd
from main import process_video

st.title("🚗 Smart Parking System")

# Initialize session state
if "run_demo" not in st.session_state:
    st.session_state.run_demo = False

uploaded_file = st.file_uploader("Upload Parking Video (Optional)", type=["mp4"])

# Button
if st.button("▶️ Click to Run Demo Video"):
    st.session_state.run_demo = True

video_path = None

# If upload
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

# If demo clicked
elif st.session_state.run_demo:
    video_path = "demo.mp4"

# If nothing selected
if video_path is None:
    st.warning("Upload a video OR click 'Run Demo Video'")
    st.stop()

# UI placeholders
stframe = st.empty()
data_placeholder = st.empty()
revenue_placeholder = st.empty()

# Process video
for frame, rec, revenue in process_video(video_path):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame, channels="RGB")

    if len(rec) > 0:
        df = pd.DataFrame(rec)
        data_placeholder.dataframe(df)

    revenue_placeholder.markdown(f"### 💰 Total Revenue: {revenue}")

st.success("Processing Complete")