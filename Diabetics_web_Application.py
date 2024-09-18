# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:13:16 2024

@author: USER
"""

import numpy as np 
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/Machine learning model/trained_model.sav','rb'))


#Create a function

def diabetics_prediction(input_data):
 
    input_data_as_numpy = np.asarray(input_data)
    input_data_reshape = input_data_as_numpy.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)
    if (prediction[0] == 0 ):
     return "The Person is No Diabetics"
    else :
        return "The Person is Diabetics"

def web():
    st.title('Diabetics Prediction By Mobassir')
    
    Pregnancies = st.text_input('Number Of  Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BloodPressure value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin  = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction value')
    Age = st.text_input(',Age Of The Person') 
    
    diagnosis = ''
    #creatintg a function
    if st.button('diabetics Test Result'):
        diagnosis = diabetics_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    

if __name__== '__web__':
    web()