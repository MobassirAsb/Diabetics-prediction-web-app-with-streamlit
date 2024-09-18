# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
loaded_model = pickle.load(open('D:/Machine learning model/trained_model.sav','rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy = np.asarray(input_data)
input_data_reshape = input_data_as_numpy.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshape)
print(prediction)
if (prediction[0] == 0 ):
  print("The Person is No Diabetics")
else :
  print("The Person is Diabetics")