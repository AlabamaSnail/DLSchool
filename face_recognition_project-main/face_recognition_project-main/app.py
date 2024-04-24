import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time 
from datetime import datetime
# Try to read the CSV file
try:
    df = pd.read_csv("face_recognition_project-main/Attendance/Attendance_.csv")

except FileNotFoundError:
    st.error("File not found or empty.")


ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")



count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")


df=pd.read_csv("face_recognition_project-main/Attendance/Attendance_.csv")

st.dataframe(df.style.highlight_max(axis=0))


