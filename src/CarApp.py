import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "saved", "car_price__model.sav")
label_paths = {
    "name": os.path.join(BASE_DIR, "saved", "label_encoder_name.sav"),
    "fuel": os.path.join(BASE_DIR, "saved", "label_encoder_fuel.sav"),
    "seller": os.path.join(BASE_DIR, "saved", "label_encoder_seller.sav"),
    "trans": os.path.join(BASE_DIR, "saved", "label_encoder_trans.sav"),
    "owner": os.path.join(BASE_DIR, "saved", "label_encoder_owner.sav")
}
image_path = os.path.join(BASE_DIR, "assets", "image.webp")

try:
    model = pickle.load(open(model_path, 'rb'))
    label_encoders = {key: pickle.load(open(path, 'rb')) for key, path in label_paths.items()}
except FileNotFoundError as e:
    st.error(f"⚠️ ملف مفقود: {e.filename}")
    st.stop()

training_columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',
                    'owner', 'mileage', 'engine', 'max_power', 'seats']

list_Cars = ['Mercedes-Benz', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
             'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mitsubishi', 'Audi',
             'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo',
             'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel', 'Maruti']

fuel_types = ['Diesel', 'Petrol', 'LPG', 'CNG']
owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
transmission_types = ['Manual', 'Automatic']
seats_types = [2, 4, 5, 6, 7, 8, 9, 10, 14]

st.set_page_config(page_title="Car Price Prediction")
st.title('🚗 Car Price Prediction')

if os.path.exists(image_path):
    try:
        st.image(image_path)
    except Exception:
        st.warning("⚠️ تعذر عرض الصورة.")
else:
    st.warning("⚠️ صورة السيارة غير موجودة.")

with st.container():
    st.header('أدخل تفاصيل السيارة')
    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox('ماركة السيارة', list_Cars)
        fuel = st.selectbox('نوع الوقود', fuel_types)
        seller_type = st.selectbox('نوع البائع', seller_types)
        transmission = st.selectbox('نوع ناقل الحركة', transmission_types)
        owner = st.selectbox('عدد الملاك السابقين', owner_types)
        seats = st.selectbox('عدد المقاعد', seats_types)
    with col2:
        km_driven = st.number_input('المسافة المقطوعة (كم)', min_value=0, value=50000)
        year = st.number_input('سنة الصنع', value=2010)
        mileage = st.slider('الكيلومترات لكل لتر', 0, 40, 20)
        engine = st.slider('سعة المحرك (CC)', 0, 4000, 1200)
        max_power = st.slider('القدرة القصوى (hp)', 0, 1000, 120)

def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return 0  # قيمة بديلة لو العلامة مش موجودة

def encode_data(df):
    df['name'] = safe_transform(label_encoders['name'], df['name'][0])
    df['fuel'] = safe_transform(label_encoders['fuel'], df['fuel'][0])
    df['seller_type'] = safe_transform(label_encoders['seller'], df['seller_type'][0])
    df['transmission'] = safe_transform(label_encoders['trans'], df['transmission'][0])
    df['owner'] = safe_transform(label_encoders['owner'], df['owner'][0])
    return df

input_df = pd.DataFrame([{
    'name': name,
    'year': year,
    'km_driven': km_driven,
    'fuel': fuel,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner,
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power,
    'seats': seats
}])

input_encoded = encode_data(input_df)
input_encoded = input_encoded[training_columns]

if st.button("🔮 توقع السعر"):
    try:
        predicted_price = model.predict(input_encoded)[0]
        st.success(f"✅ السعر المتوقع: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"⚠️ خطأ أثناء التنبؤ: {str(e)}")
    if hasattr(model, "feature_importances_"):
        st.subheader("🔍 أهمية الخصائص")
        importance_df = pd.DataFrame({
            "feature": training_columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.info("ℹ️ النموذج لا يدعم عرض أهمية الخصائص.")
