# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
!pip install -r requirements.txt

import numpy as np
import pickle
import streamlit as st
import cv2

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#Creating function for prediction

def diabest_prediction(input_data):
    #input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The patient is not diabetic'
    else:
      return 'The patiant is diabetic'

def main():
   
   #title
   st.title('Diabetes Prediction Web Application')

   #Getting the inoput data from the user

def main():
   
   #title
   st.title('Diabetes Prediction Web Application')

   #Getting the inoput data from the user
   Pregnancies = st.text_input('Number of Pregnancies')
   Glucose = st.text_input('Glucose Level')
   BloodPressure = st.text_input('Blood Pressure value')
   SkinThickness = st.text_input('Skin Thickness value')
   Insulin = st.text_input('Insulin Level Value')
   BMI = st.text_input('BMI value')
   DiabetesPedigreeFunction = st.text_input('Diabestes Pedigree Function Value')
   Age = st.text_input('Age of Patiant')

   #Code for Prediction
   diagnosis = ''

   #Creating a button for prediction

   if st.button('Diabetes Test Result') == True:
      diagnosis = diabest_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
      print(diagnosis)

      st.success(diagnosis)

if __name__ == '__main__':
   main()
