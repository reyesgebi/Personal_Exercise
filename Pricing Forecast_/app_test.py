import pandas as pd
import numpy as np
import json
import joblib
import streamlit as st

# Open Files

with open('LinearRegression.pkl', 'rb') as file_1:
   regression_model = joblib.load(file_1)

with open('scaling_mod.pkl', 'rb') as file_2:
   scaling_model = joblib.load(file_2)

with open('ordinal_mod.pkl', 'rb') as file_3:
   ordinal_model = joblib.load(file_3)

with open('OHE_mod.pkl', 'rb') as file_4:
    ohe_model = joblib.load(file_4)

with open('num_features.txt', 'r') as file_5:
    numcol = json.load(file_5)

with open('cat_features.txt', 'r') as file_6:
    catcol = json.load(file_6)

name = st.selectbox('Service Name',tuple(ohe_model.categories_[0].tolist()))
distance = st.slider('Distance (miles)',0,20)
surge_multiplier = st.slider('Surge Multiplier',[1.  , 1.75, 1.25, 1.5 , 2.  , 2.5 , 3.  ])
st.write(type(surge_multiplier))