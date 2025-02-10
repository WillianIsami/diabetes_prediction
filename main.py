import streamlit as st
import numpy as np
import pickle

def load_model():
    with open("diabetes_model.sav", "rb") as model_file:
        model = pickle.load(model_file)
    with open("diabetes_scaler.sav", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

def predict_diabetes(model, scaler, input_data):
    # Scale the input data
    scaled_data = scaler.transform(np.array(input_data).reshape(1, -1))
    
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]
    
    return prediction[0], probability

def main():
    st.title('Diabetes Prediction App')
    
    # Input fields
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    
    # Prediction button
    if st.button('Predict Diabetes'):
        model, scaler = load_model()        
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                      insulin, bmi, diabetes_pedigree, age]
        prediction, probability = predict_diabetes(model, scaler, input_data)
        
        if prediction == 1:
            st.error(f'Diabetes Predicted (Probability: {probability:.2%})')
        else:
            st.success(f'No Diabetes Predicted (Probability: {probability:.2%})')

if __name__ == "__main__":
    main()