import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

model = pk.load(open('model.pkl', 'rb'))

st.header("Car Price Prediction Machine Learning Model")

df = pd.read_csv("Car_Price_Prediction.csv")


def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()


df['name'] = df['name'].apply(get_brand_name)

name = st.selectbox("Select Car Brand", df['name'].unique())
year = st.slider("Year Manufactured", 1994, 2024)
km_driven = st.slider("No of Kms Driven", 10, 300000)
fuel = st.selectbox("Fuel type", df['fuel'].unique())
seller_type = st.selectbox("Seller_type", df['seller_type'].unique())
transmission = st.selectbox("Transmission Type", df['transmission'].unique())
owner = st.selectbox("Owner", df["owner"].unique())
mileage = st.slider("Car Mileage", 0, 45)
engine = st.slider("Engine", 600, 5000)
max_power = st.slider("Max Power", 30, 500)
torque_num = st.slider("Torque", 4, 800)
seats = st.slider("Number of Seats", 2, 14)

if st.button("Predict"):
    data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine,
                 max_power, seats, torque_num]],
        columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine',
                 'max_power', 'seats', 'torque_num'])
    data_model['fuel'].replace(["Diesel", "Petrol", "LP", "CNG"], [1, 2, 3, 4], inplace=True)
    data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], inplace=True)
    data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5], inplace=True)
    price = model.predict(data_model)
    st.markdown('Car Price is going to be ' + str(price[0]))










