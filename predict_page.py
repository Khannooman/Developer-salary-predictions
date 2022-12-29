import streamlit as st
import pandas as ps
import numpy as np
import pickle


def load_model():

    with open("models.pkl",'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_Education = data["le_Education"]
le_Employment = data["le_Employment"]

def show_predict_page():
    st.title("Softare Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
    'Canada', 'United Kingdom', 'Israel', 'United States of America',
        'Germany', 'India', 'Netherlands', 'others', 'Australia',
        'Russian Federation', 'Czech Republic', 'Austria', 'Italy',
        'Poland', 'Sweden', 'Mexico', 'France', 'Brazil', 'Denmark',
        'Spain', 'Turkey', 'Ukraine', 'Romania', 'Portugal', 'Belgium',
        'Greece', 'Switzerland', 'China', 'Argentina', 'Bangladesh',
        'Pakistan', 'Iran', 'Indonesia', 'Nigeria'
    )

    education = (
        "Bachelor’s degree", "Master’s degree", "Others", "Post graduated"
    )

    employment = (
        'Full Time', 'Others'
    )

    country = st.selectbox("Country" , countries)

    education_level = st.selectbox("Education Level" , education)
    Employment = st.selectbox("Employement" , employment)
    experience = st.slider("Year oF Experience",0,50,3)

    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[Employment,education_level, country,experience]])
        x[:,0] = le_Employment.transform(x[:,0])
        x[:,1] = le_Education.transform(x[:,1])
        x[:, 2] = le_country.transform(x[:,2])
        x = x.astype("float64")

        Salary = regressor.predict(x)
        st.subheader(f"The estimated salary is ${Salary[0][0]:.2f}")
         





        

