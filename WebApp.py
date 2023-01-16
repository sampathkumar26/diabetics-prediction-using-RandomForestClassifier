#Description: This program detects if someone has diabetes using mavhine learning and python

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python !
""")

#Open and display an image
image = Image.open('images.jpg')
st.image(image, caption='ML', use_column_width=True)


#the data
df = pd.read_csv('diabetes.csv')
st.subheader('Data Information:')

#data as a table
st.dataframe(df)

st.write(df.describe())

chart = st.line_chart(df)

#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y= df.iloc[:,-1].values
# Split the dataset into 75% Training set and 25% Testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17,3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure',0, 122,72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    
    
    #store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }
    
    #transform the data into adata frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#store the user input into a variable  
user_input = get_user_input()

#set a subheader and diplay the user input
st.subheader('User Input :')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%' )

#Store the models prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)
