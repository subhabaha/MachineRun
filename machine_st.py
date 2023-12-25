#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_option_menu import option_menu

def get_log(model, video): 
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
    video_path = video
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
            st.write(f"{current_time} - Screenshot captured: {screenshot_filename}, Machine {status}")
            if status == status1:
                status_chk += 1
            else:
                status_chk = 0
            status1 = status


        # Check for consecutive frames and update machine status
        if status_chk >= consecutive_frames_threshold:
            print(f"Machine Status: {status}")
            status_chk = 0

        frame_count += 1

    # Release the webcam capture object and close the OpenCV window
    cap.release()

def get_machine_status(model, video): 
    frame_count = 0
    running_frames = 0
    not_running_frames = 0
    skip_frames = 2  # Skip 2 frames in between each prediction
    consecutive_frames_threshold = 15
    machine_status = None
    status_chk = 0
    status1 = ""
    status = "Loading ..."
    clear = st.empty()
    count = 0
    
    # Open a connection to the webcam (0 represents the default webcam)
    video_path = video
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the interval for capturing screenshots in seconds
    screenshot_interval = 1  # 1-second interval
    screenshot_frames = int(screenshot_interval * fps)  # Number of frames to wait for 1 second

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
            if status == status1:
                status_chk += 1
            else:
                status_chk = 0
            status1 = status


        # Check for consecutive frames and update machine status
        if status_chk >= consecutive_frames_threshold or count == 0:
            with clear.container():
                st.write(f"Machine Status: {status}")
                status_chk = 0
        count = 1
        frame_count += 1

    # Release the webcam capture object and close the OpenCV window
    cap.release()

# Set the layout
st.set_page_config(page_title="Machine Status App", page_icon="🤖", layout="wide")

# Main title
st.title("Machine Status Monitoring App")

# Load the saved model
model = tf.keras.models.load_model('machine_model_temp.h5')
video_path = "sample video.mp4"

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Machine Status", "Machine runtime log"],
        icons = ["lightning-charge-fill", "list-columns"],
        default_index = 0)

if selected == "Machine Status":
    # Display the machine status
    st.subheader("Machine Status")
    # Call the function to get the machine status and log
    machine_status = get_machine_status(model, video_path)

if selected == "Machine runtime log":
    # Display the machine status
    st.subheader("Machine runtime log")
    get_log(model, video_path)
