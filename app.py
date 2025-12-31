import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

def load_picle_obj(file_path):
    with open(file_path,'rb') as file_obj:
        object=pickle.load(file_obj)
    return object


model=tf.keras.models.load_model(r'C:\Users\jaive\OneDrive\Documents\deep_learning\models\tesormodel.h5')

gen_label_encoder:LabelEncoder=load_picle_obj(r'encoder\label_encoder.pkl')
geo_onehot_encoder:OneHotEncoder=load_picle_obj(r'encoder\onehot_encoder.pkl')
scallar:StandardScaler=load_picle_obj(r'scallar\scalar.pkl')

# starting stream liteapp
st.title("customer chrn prediction using Artifical Nural network")

CreditScore=st.number_input('CreditScore')
Geography = st.selectbox('Geography', geo_onehot_encoder.categories_[0])
Gender = st.selectbox('Gender', gen_label_encoder.classes_)
Age=st.number_input("Age",min_value=1,max_value=120)
Tenure=st.slider('Tenure', 0, 10)
Balance=st.number_input("Balance",min_value=100)
NumOfProducts=st.slider('Number of Products', 1, 4)
HasCrCard=st.selectbox('Has Credit Card', [0, 1])
IsActiveMember=st.selectbox('Is Active Member', [0, 1])
EstimatedSalary=st.number_input('Estimated Salary')

input_data = {
    "CreditScore": CreditScore,
    "Geography": Geography,
    "Gender": Gender,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary
}
df=pd.DataFrame([input_data])
df['Gender']=gen_label_encoder.transform(df["Gender"])
geo_encoded = geo_onehot_encoder.transform([[Geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=geo_onehot_encoder.get_feature_names_out(['Geography']))
df=pd.concat([df.drop(['Geography'],axis=1),geo_df],axis=1)
x=scallar.transform(df)
y=model.predict(x)
st.write(f'Churn Probability: {y}')

if y[0]> 0.5:
    st.write('The cutomer will exist the bank')
else:
    st.write('The customer is not likely  to Exit.')