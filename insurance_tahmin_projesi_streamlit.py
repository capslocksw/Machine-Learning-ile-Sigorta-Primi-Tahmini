import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Başlık
st.title("Sigorta Primi Tahmini")

# Temizlenmiş veriyi yükleme
with open("C:\\Users\\ASUS\Desktop\\acun medya akademi\\benim notlarım\\insurance_dataset_tahmin_projesi\\processed_insurance_data_enson.pkl", 'rb') as file:
    df = pickle.load(file)

# Modeli ve dönüştürücüleri yükleme
model = joblib.load("C:\\Users\\ASUS\Desktop\\acun medya akademi\\benim notlarım\\insurance_dataset_tahmin_projesi\\lasso_regression_model_insurance_enson.joblib")
poly = joblib.load("C:\\Users\\ASUS\Desktop\\acun medya akademi\\benim notlarım\\insurance_dataset_tahmin_projesi\\poly_transformer.pkl")
scaler = joblib.load("C:\\Users\\ASUS\Desktop\\acun medya akademi\\benim notlarım\\insurance_dataset_tahmin_projesi\\scaler.pkl")

# Özellik adlarını yükleme
with open("c:\\Users\\ASUS\\Desktop\\acun medya akademi\\benim notlarım\\insurance_dataset_tahmin_projesi\\feature_names.pkl", 'rb') as f:
    feature_names = pickle.load(f)

# Kullanıcıdan veri girişi alma
st.sidebar.header("Girdi Değerlerini Ayarlayın")

def user_input_features():
    age = st.sidebar.slider('Yaş', 18, 100, 30)
    bmi = st.sidebar.slider('BMI', 15.0, 50.0, 25.0)
    children = st.sidebar.slider('Çocuk Sayısı', 0, 5, 0)
    sex_male = st.sidebar.selectbox('Cinsiyet', ('Kadın', 'Erkek'))
    smoker_yes = st.sidebar.selectbox('Sigara İçiyor mu?', ('Hayır', 'Evet'))
    region = st.sidebar.selectbox('Bölge', ('Güneydoğu', 'Kuzeybatı', 'Kuzeydoğu', 'Güneybatı'))
    
    data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex_male == 'Erkek' else 0,
        'smoker_yes': 1 if smoker_yes == 'Evet' else 0,
        'region_northwest': 1 if region == 'Kuzeybatı' else 0,
        'region_southeast': 1 if region == 'Güneydoğu' else 0,
        'region_southwest': 1 if region == 'Güneybatı' else 0,
        'age_bmi': age * bmi,
        'age_category_genç': 1 if age < 35 else 0,
        'age_category_orta yaş': 1 if 35 <= age < 55 else 0,
        'age_category_yaşlı': 1 if age >= 55 else 0,
        'bmi_category_normal': 1 if 18.5 <= bmi < 25 else 0,
        'bmi_category_fazla kilolu': 1 if 25 <= bmi < 30 else 0,
        'bmi_category_obez': 1 if bmi >= 30 else 0,
        'smoker_bmi': (1 if smoker_yes == 'Evet' else 0) * bmi
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Özellikleri doğru sırayla ayarlayın
input_df = input_df[feature_names]

# Polinomsal özellikler ve ölçeklendirme
input_df_poly = poly.transform(input_df)
input_df_poly_scaled = scaler.transform(input_df_poly)

# Tahmin
prediction = model.predict(input_df_poly_scaled)

# Sonuçları gösterme
st.subheader('Tahmin Edilen Sigorta Primi')
st.write(prediction[0])