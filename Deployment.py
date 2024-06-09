import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import OneHotEncoder


page_bg_img = """

<style>
@import url('https://fonts.googleapis.com/css2?family=Aptos&display=swap');
[data-testid="stAppViewContainer"]{
background-color: #f1f8f7;
# linear-gradient(to left, , white);
}
h1{
    text-align: center;
    color:#153a6d;
    font-family: 'Aptos', sans-serif;  
}

h2{
    color: #F0F0F0;
    font-family: 'Aptos', sans-serif; 
}
h3{
    color: white;
    font-family: 'Aptos', sans-serif; 
}
h4{
    color: white;
    font-family: 'Aptos', sans-serif; 
}
p{
    color:#10244f;
    # font-size: 50px;
    font-family: 'Aptos', sans-serif; 
}
input {
    background-color: #C1C7C9;
    color: #white;
    border: 1px solid black;
    border-radius: 5px;
    padding: 10px;
}


# textarea {
#     background-color: black;
#     color: #000000;
#     border: 1px solid #4E65FF;
#     border-radius: 5px;
#     padding: 10px;
# }
.st-bc.st-bx.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-b9.st-c7.st-c8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-ae.st-af.st-ag.st-cf.st-ai.st-aj.st-bw.st-cg.st-ch.st-ci {
    background-color: white;
    color: black; 
}

.stButton button {
    background-color: #4FC0D0;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
}

.stButton button:hover {
    # background-color: #3B50CC;
    background-color: #1B6B93;
    color:white !important;
}
.st-emotion-cache-ul70r3.e1nzilvr4 {
    color: lightgreen;
}


</style>

"""
st.markdown(page_bg_img, unsafe_allow_html=True)

#loading the saved model
with open('D:/document/Binus/Semester4/ML/Assessment/Deployment/trained_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# Access the model and encoder
model = loaded_model['model']
one_hot_encoder = loaded_model['one_hot_encoder']
ordinal_encoder = loaded_model['ordinal_encoder']
label_encoder = loaded_model['label_encoder']
min_max_scaler = loaded_model['min_max_scaler']
x_train = loaded_model['x_train']

#creating a function for obesity prediction
def obesity_prediction(input_datatest):
    # input_datatest = (1, 1, 0, 0, 0, 0, 1, 0, 1, 24.44301, 1.699998, 81.66995, 2, 2.983297, 1, 2.763573,0, 0.976473, 1)

    # Mengubah input_data menjadi array numpy
    input_data_as_numpy_array = np.asarray(input_datatest)

    # Melakukan transformasi data dengan minMax_scaler
    input_data_transformed = min_max_scaler.transform([input_data_as_numpy_array])

    # Reshape array karena kita memprediksi untuk satu instance
    input_data_reshaped = input_data_transformed.reshape(1, -1)
    # Melakukan prediksi
    prediction = model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
        # st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'Insufficient_Weight'
    if (prediction[0] == 1):
        # st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'Normal_Weight'
    if (prediction[0] == 2):
        # st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'Obesity_Type_I'
    if (prediction[0] == 3):
        # st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'Obesity_Type_II'
    if (prediction[0] == 4):
        # st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'Obesity_Type_III'
    if (prediction[0] == 5):
        # st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'Overweight_Level_I'
    if (prediction[0] == 6):
        # st.markdown('<style> .st-au {background-color: #32CD32} </style>', unsafe_allow_html=True)
        return 'Overweight_Level_II'


    
def main():
    #giving a title 
    st.title('Obesity Levels Prediction Web App')
    
    #getting the input data from the user 
    
    # Combine input variables into x_test
    input_data = {
        'Gender': st.radio('Gender:',['Male','Female']),
        'Age': st.number_input("Age:",min_value=0),
        'Height': st.number_input("Height(meter):",min_value=0.00),
        'Weight': st.number_input("Weight(kg):",min_value=0.00),
        'FHWO': st.radio("Do you have family history of weight issues?",['yes','no']),
        'FAVC': st.radio("Do you often consume high calorie food?",['yes','no']),
        'FCVC': st.slider("How much frequency do you consume vegetables every day?",0,4,0),
        'NCP': st.slider("How much main meals do you consume every day?",0,4,0),
        'CAEC': st.radio("How many consumption of food between meals?",['Sometimes','Frequently','Always','no']),
        'SMOKE': st.radio("Do you smoke?",['yes','no']),
        'CH2O': st.slider("How much consumption of water daily every day?", 0, 3, 0),
        'SCC': st.radio("Do you have calories consumption monitoring?",['yes','no']),
        'FAF': st.slider("How often do you engage in exercise or any physical activities?", 0, 3, 0),
        'TUE': st.slider("How much time do you spend daily using technology devices such as smartphones, computers, or tablets?",0,2,0),
        'CALC': st.radio("How often do you consume alcohol?",['Sometimes','Frequently','Always','no']),
        'MTRANS': st.radio("What mode of transportation do you primarily use to commute?",['Public_Transportation','Automobile', 'Walking', 'Bike'])
    }
    # Convert input_data to DataFrame
    x_test = pd.DataFrame([input_data])    
    

    #code for prediction 
    diagnosis = ''
    
    #Creating a button for prediction
    if st.button('Obesity Test Result'):
        # 1.Columns to be one-hot encoded
        columns_to_encode = ['FHWO', 'FAVC', 'SMOKE', 'SCC', 'MTRANS', 'Gender']
        
        # Initialize an empty DataFrame to store encoded features
        test_encoded_features = pd.DataFrame(index=x_test.index)

        # Iterate over each column to encode
        for column in columns_to_encode:
            # One-hot encode the current column
            one_hot_encoder = OneHotEncoder(sparse_output=True, drop='first')  # drop='first' to drop the first level for each feature
            # Fit the encoder to the training data
            one_hot_encoder.fit(x_train[[column]])
            # Apply transform
            test_encoded_column = one_hot_encoder.transform(x_test[[column]])

            # Convert the encoded features into DataFrame and concatenate with the existing encoded features
            encoded_df = pd.DataFrame(test_encoded_column.toarray(), columns=one_hot_encoder.get_feature_names_out([column]), index=x_test.index)
            test_encoded_features = pd.concat([test_encoded_features, encoded_df], axis=1)

        # Reset Index is done to preserve the indices
        test_encoded_features.reset_index(drop=True, inplace=True)

        # Create a dataframe without the encoded features
        test_df = x_test.drop(columns=columns_to_encode, inplace=False)
        test_df.reset_index(drop=True, inplace=True)  # Reset Index is done to preserve the indices

        # Concatenate the encoded features with the original DataFrame
        x_test_encoded = pd.concat([test_encoded_features, test_df], axis=1)

        #2 Ordinal Encoding
        x_test_encoded[['CALC', 'CAEC']] = ordinal_encoder.transform(x_test_encoded[['CALC', 'CAEC']])        
        diagnosis = obesity_prediction(x_test_encoded.values[0])    
        
    if diagnosis:        
        st.markdown(
        f"""
        <div style="text-align: center; background-color: #e4f1ee;">
            <span style="color: #10244f; font-size: 24px;font-family: 'Aptos', sans-serif; ">{diagnosis}</span>
        </div>
        """,
        unsafe_allow_html=True,
        )
    
if __name__ == '__main__':
    main()
    
    
