import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct paths dynamically
model_path = os.path.join(BASE_DIR, "saved", "car_price__model.sav")
label_encoder_name_path = os.path.join(BASE_DIR, "saved", "label_encoder_name.sav")
label_encoder_fuel_path = os.path.join(BASE_DIR, "saved", "label_encoder_fuel.sav")
label_encoder_seller_path = os.path.join(BASE_DIR, "saved", "label_encoder_seller.sav")
label_encoder_trans_path = os.path.join(BASE_DIR, "saved", "label_encoder_trans.sav")
label_encoder_owner_path = os.path.join(BASE_DIR, "saved", "label_encoder_owner.sav")

# Load the models safely
model = pickle.load(open(model_path, 'rb'))
label_encoder_name = pickle.load(open(label_encoder_name_path, 'rb'))
label_encoder_fuel = pickle.load(open(label_encoder_fuel_path, 'rb'))
label_encoder_seller = pickle.load(open(label_encoder_seller_path, 'rb'))
label_encoder_trans = pickle.load(open(label_encoder_trans_path, 'rb'))
label_encoder_owner = pickle.load(open(label_encoder_owner_path, 'rb'))


# Define the columns used during training
training_columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']

# Function to encode categorical columns using the pre-trained label encoders
def encode_features(data, label_encoder_name, label_encoder_fuel, label_encoder_seller, label_encoder_trans, label_encoder_owner):
    for col, encoder in zip(['name', 'fuel', 'seller_type', 'transmission', 'owner'], 
                             [label_encoder_name, label_encoder_fuel, label_encoder_seller, label_encoder_trans, label_encoder_owner]):
        try:
            data[col] = encoder.transform(data[col])
        except ValueError:
            known_classes = encoder.classes_
            data[col] = data[col].apply(lambda x: encoder.transform([x])[0] if x in known_classes else 0)
    
    return data


list_Cars = ['Mercedes-Benz', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel','Maruti']

fuel_types = ['Diesel', 'Petrol', 'LPG', 'CNG']
owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
Seller_Types = ['Individual', 'Dealer', 'Trustmark Dealer']
transmission_types = ['Manual', 'Automatic']
seats_types = [2, 4, 5, 6, 7, 8, 9, 10, 14]
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Streamlit App UI

st.set_page_config(
    page_title="Car Price Prediction"
)
  
st.title('Car Price Prediction 🚀')


image_path = os.path.join(BASE_DIR, "assets", "image.webp")
st.image(image_path, use_container_width=True)

   
# Input fields for user data
with st.container(height=600):
    st.header('Enter Car Details')
    col1 , col2 = st.columns(2)
    with col1:
        name = st.selectbox('Brand of Car', list_Cars)
        fuel = st.selectbox('Type of Fuel', fuel_types)
        seller_type = st.selectbox('Type of Sellers', Seller_Types)
        transmission = st.selectbox('Type of Transmission', transmission_types)
        owner = st.selectbox('Type of Owner', owner_types)
        seats = st.selectbox('Number of Seat', seats_types)
    with col2:
        km_driven = st.number_input('KM Driven of Car', min_value=0, value=50000)
        year = st.number_input('Year of Manufacture', value=2000)
        mileage = st.slider('Mileage of Car' ,0,40,20)
        engine = st.slider('Engine of Car (CC)',0,4000,1200 )
        max_power = st.slider('Max Power of Car (hp)',0,1000,120 )


# Prepare the input data for prediction
input_data = pd.DataFrame({
    'name': [name],
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats]
})

# Encode the categorical features using the pre-trained label encoders
input_data_encoded = encode_features(input_data, label_encoder_name, label_encoder_fuel, label_encoder_seller, label_encoder_trans, label_encoder_owner)

# Ensure the columns are in the same order as during model training
input_data_encoded = input_data_encoded[training_columns]

# Button for Predicting
if st.button('Predict Price'):
        # Predict the price 
        predicted_price = model.predict(input_data_encoded)[0]
        
        st.write(f'The Predicted Price is ₹{predicted_price:.2f}')
        # Load feature importance from model
        feature_importance = pd.DataFrame({
            'feature': ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Feature Importance Visualization
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='importance', y='feature', data=feature_importance, palette="viridis", ax=ax)
        ax.set_title('Feature Importance in Predicting Car Price ')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        st.pyplot(fig) 

      
      
     
