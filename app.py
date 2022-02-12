import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
# creating a function for Prediction


def cancer_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The Breast Cancer is Malignant'
    else:
        return 'The Breast Cancer is Benign'


def main():
    # giving a title
    st.title('Devloper@Shubham Sharma')
    st.title('Breast Cancer Prediction Web App')
    # getting the input data from the user
    # mean radius	mean texture	mean perimeter	mean area	perimeter error	area error	worst radius	worst texture	worst perimeter	worst area
    MeanRadius = st.text_input('Enter the Mean Radius')
    MeanTexture = st.text_input('Enter the Mean Texture')
    MeanPerimeter = st.text_input('Enter the Mean Perimeter')
    MeanArea = st.text_input('Enter the Mean Area')
    PerimeterError = st.text_input('Enter the Perimeter Error')
    AreaError = st.text_input('Enter the Area Error')
    WorstRadius = st.text_input('Enter the Worst Radius')
    WorstTexture = st.text_input('Enter the Worst Texture')
    WorstPerimeter = st.text_input('Enter the Worst Perimeter')
    WorstArea = st.text_input('Enter the Worst Area')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Breast Cancer Test Result'):
        diagnosis = cancer_prediction([MeanRadius, MeanTexture, MeanPerimeter, MeanArea,
                                      PerimeterError, AreaError, WorstRadius, WorstTexture, WorstPerimeter, WorstArea])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
