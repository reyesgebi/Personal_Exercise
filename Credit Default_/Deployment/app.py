import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from PIL import Image

# Set page title and icon
st.set_page_config(page_title='Predicting Death Event', page_icon=':skull:')

# Add a header image
header_image = Image.open('5-consequences-of-a-credit-card-default.png')
st.image(header_image, use_column_width=True)

# Add some background information
st.write("""
# Predicting Credit Default

This app uses a Random Forest Classifier to predict the likelihood of a customer to defaulting their credit
""")

# Load the saved model
with open('random.pkl', 'rb') as file_1:
    randfor = joblib.load(file_1)

# Add input fields for the patient's data

credit_type = st.selectbox('Credit Type:',('CIB','CRIF','EXP','EQUI'))
lump_sum_payment = st.selectbox('Lump Sum Payment',('lpsm','not_lpsm'))
Interest_rate_spread = st.number_input('Interest Rate Spread')
rate_of_interest = st.number_input('Rate of Interest')
dtir = st.number_input('dtir')

# Create a pandas DataFrame with the patient's data
df = pd.DataFrame({
    'credit_type': [credit_type], 
    'lump_sum_payment': [lump_sum_payment], 
    'Interest_rate_spread': [Interest_rate_spread], 
    'rate_of_interest': [rate_of_interest],
    'dtir1': [dtir],
})

# Add a button to trigger the prediction
if st.button('Predict'):
    # Use the model to predict the patient's outcome
    label = randfor.predict(df)
    
    # Display the prediction
    if label == 1:
        st.subheader(f'Patient is at high risk of defaulting. Please take necessary action.')
    else:
        st.subheader(f'Patient is at low risk of defaulting. Continue to monitor.')