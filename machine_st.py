#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess

#model = tf.keras.models.load_model('modelqw.h5')

def main():
    # Set the layout
    st.set_page_config(page_title="Machine Status App", page_icon="🤖", layout="wide")

    # Main title
    st.title("Machine Status Monitoring App")

    # Load the saved model
    if not os.path.isfile('model.h5'):
        subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/subhabaha/MachineRun/main/machine_model.h5"'], shell=True)
        st.write(print("ok"))
    else:
        st.write(print("not ok"))

    # Button to execute the code
    if st.button("Execute Code"):
        # Call the function to get the machine status and log
        machine_status = get_machine_status_and_log()

        # Display the machine status
        st.subheader("Machine Status")
        st.info(machine_status)

        # Display the machine status log
        st.subheader("Machine Status Monitoring Log")
        st.info(print(get_machine_status_and_log()))

def get_machine_status_and_log(): 
    frame_count = 0
    running_frames = 0
    not_running_frames = 0
    skip_frames = 2  # Skip 2 frames in between each prediction
    consecutive_frames_threshold = 15
    machine_status = None
    status_chk = 0
    status1 = ""
    status = ""
    
    # Open a connection to the webcam (0 represents the default webcam)
    video_path = "sample video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the interval for capturing screenshots in seconds
    screenshot_interval = 1  # 1-second interval
    screenshot_frames = int(screenshot_interval * fps)  # Number of frames to wait for 1 second

    # Initialize variables
    frame_count = 0
    running_frames = 0
    not_running_frames = 0
    skip_frames = 2  # Skip 2 frames in between each prediction
    consecutive_frames_threshold = 15
    machine_status = None
    status_chk = 0
    status1 = ""

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Resize the frame to match the input size of the model
        frame = cv2.resize(frame, (224, 224))

        # Preprocess the frame
        img_array = image.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        prediction = model.predict(img_array, verbose=0)

        # Classify as running if the prediction is above 0.5 (you can adjust this threshold)
        if prediction > 0.5:
            running_frames += 1
            not_running_frames = 0
        else:
            running_frames = 0
            not_running_frames += 1

        # Capture a screenshot at the specified interval
        if frame_count % screenshot_frames == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            screenshot_filename = f"screenshot_{current_time}.png"
            cv2.imwrite(screenshot_filename, frame)
            if prediction > 0.5:
                status = "Running"
            else:
                status = "Not Running"
            print(f"{current_time} - Screenshot captured: {screenshot_filename}, Machine {status}")
            if status == status1:
                status_chk += 1
            else:
                status_chk = 0
            status1 = status


        # Check for consecutive frames and update machine status
        if status_chk >= consecutive_frames_threshold:
            print(f"Machine Status: {status}")
            status_chk = 0

        # Display the frame with prediction
        cv2.imshow('Webcam', frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release the webcam capture object and close the OpenCV window
    cap.release()
    
    return status

if __name__ == "__main__":
    main()

