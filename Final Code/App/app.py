from matplotlib.pyplot import stem, step
from numpy.core.defchararray import encode
from scipy.sparse.construct import random
import streamlit as st
import joblib
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import time
from streamlit.elements import number_input
from streamlit.proto.Button_pb2 import Button
import numpy as np
from sklearn.preprocessing import LabelEncoder
sys.setrecursionlimit(10000)
#server.enableWebsocketCompression = False
# STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION = False

st.set_page_config(layout="wide")
model = pickle.load(open('ca_vclf.pkl','rb'))
#print(model)


nav = st.sidebar.radio("Navigation",['Home','Cardiac Arrythmia','Diabetes','Kidney Disease','Stroke'])
df_ca = pd.read_csv('processedheartdata.csv')
X = df_ca.iloc[:,1:-1]
y = df_ca.iloc[:,-1]
X_train_ca,X_test_ca, y_train_ca , y_test_ca = train_test_split(X,y,test_size=0.20,random_state=42)


df_d = pd.read_csv('diabetes.csv')
X = df_d.iloc[:,1:-1]
y = df_d.iloc[:,-1]
X_train_d,X_test_d, y_train_d , y_test_d = train_test_split(X,y,test_size=0.20,random_state=42)



df_s = pd.read_csv('stroke-data.csv')

if nav == 'Home':
    st.title('Disease Prediction App')
    st.markdown('''
    #### With this app, you check the risk of getting a disease just by entering some details. <br><br><br><br>
    # Services provided by us:<br>
    ## 1. Risk Prediction of Cardiac Arrythmia<br>
    ## 2. Probability of getting Diabetes in Future<br>
    ## 3. Probability of getting a Kidney Disease in Future<br>
    ## 4. Probability of having a stroke
    ''')

if nav == 'Cardiac Arrythmia':
    st.title('Cardiac Arrythmia Risk Prediction')
    st.markdown('''
    ### Here you can check the risk of getting Cardiac Arrythmia in the future.
    ''')
    model_ca = pickle.load(open(r'ca_fitted_vclf.pkl','rb'))
    ccf = 0
    age = st.number_input('Age',step=5)
    sex = st.number_input('Gender (1 = Male, 0 = Female) ',step=1)
    binary = st.number_input('Have you felt any pain in the past? yes = 1, no = 0',step=1) # not to include in predictions
    if binary == 1:
        pain_location = st.number_input('Enter pain location (0 or 1)',step=1)
        pain_with_exertion = st.number_input('Did you feel pain with exertion? (1 = yes, 0= no)',step=1)
        relieved_after_rest = st.number_input('Was the pain relieved after rest? (1 = yes, 0 = no)',step=1)
        chest_pain_type = st.number_input('On a scale of 1 to 4, how painful did you feel? (1 being low and 4 being high)',step=1)
    else:
        pain_location = 0
        pain_with_exertion = 0
        relieved_after_rest = 0
        chest_pain_type = 1
    resting_bp = st.number_input('Enter your Resting Blood Pressure',step=5)
    hypertension = st.number_input('Do you have hypertension? (1 = yes, 0 = no)',step=1)
    cholesterol = st.number_input('Enter your choleterol level (a number between 125 to 603)',step=10)
    fasting_blood_sugar = 1
    resting_ecg = 1
    digitalis = 1
    beta_blocker = 1
    nitrates = 1
    calcium_channel_blocker = 1
    diuretic = 1
    exercise_protocol = 50
    duration_of_exercise = st.number_input('How many hours in a week do you exercise?',step=2)
    mets_achieved = 10
    max_heart_rate = st.number_input('Maximum heart rate achieved during exercise? (a number between 69 to 190)',step=5)
    resting_heart_rate = st.number_input('Resting heart rate (normal = 60-100 beats per minute)',step=5)
    peak_exercise_bp_1 = 150
    peak_exercise_bp_2= 160
    resting_bp_d = 200
    exercise_angina = 1
    xhypo = 1
    oldpeak = 0.79
    ST_height_at_rest = 14
    ST_height_at_peak = 14
         
    #model_ca.fit(X_train_ca,y_train_ca)
    # from sklearn.linear_model import LogisticRegression
    # model = LogisticRegression()
    
    if st.button('Predict'):
        # with st.spinner(text = 'In Progress'):
        #     time.sleep(100)
        #     st.success('Done')
        model_ca.fit(X_train_ca,y_train_ca)
        a = [[ccf,age,sex,pain_location,pain_with_exertion,relieved_after_rest,chest_pain_type,resting_bp,hypertension,
        cholesterol,fasting_blood_sugar,resting_ecg,digitalis,beta_blocker,nitrates,calcium_channel_blocker,diuretic,exercise_protocol
        ,duration_of_exercise,mets_achieved,max_heart_rate,resting_heart_rate,peak_exercise_bp_1, peak_exercise_bp_2,resting_bp_d,exercise_angina,xhypo,oldpeak,ST_height_at_rest,ST_height_at_peak]]
        a = np.array(a)
        y_pred_ca = model_ca.predict(a)
        # y_pred_ca = model_ca.predict(X_test_ca)
        # st.write(y_pred_ca)
        if y_pred_ca == 0:
            st.markdown('''
            ### You have nothing to worry about.
            ''')
        if y_pred_ca == 1:
            st.markdown('''
            ### You have very low risk of getting Arrythmia. Here are some precautions: (some link)
            ''')
        if y_pred_ca == 2 or y_pred_ca == 3:
            st.markdown('''
            ### You have high risk of getting Arrythmia. Please take some precautionary steps: (some link)
            ''')
        if y_pred_ca == 4:
            st.markdown('''
            ### You have high risk of getting Arrythmia. Please take some precautionary steps: (some link)
            ''')
    #ccf,age,sex,pain location,pain w exertion,relieved after rest,chest pain type,resting bp s,hypertension,cholesterol,fasting blood sugar,resting ecg,digitalis,beta blocker,nitrates,calcium channel blocker,diuretic,exercise protocol,duration of exercise,mets achieved,max heart rate,resting heart rate,peak exercise bp 1,peak exercise bp 2,resting bp d,exercise angina,xhypo,oldpeak,ST height at rest,ST heaight at peak,target
    


if nav == 'Diabetes':
    st.title('Diabetes Risk Prediction')
    st.markdown('''
    ### Here you can check the risk of having Diabetes in the future.
    ''')
    model_d = pickle.load(open(r'd_clfrf.pkl','rb'))
    age = st.number_input('Enter your age', step=5)
    pregnancies = 2
    glucose = st.number_input('Glucose level (a no. between 100 to 200)', step = 5)
    blood_pressure = st.number_input('Blood Pressure', step = 5)
    skin_thickness = 25
    insulin = st.number_input('Insulin level (a no. between 50 to 850)', step = 5)
    bmi = st.number_input('Body Mass Index (a no. between 32 to 65)', step = 2)
    dpf = 0.5
    if st.button('Predict'):
        model_d.fit(X_train_d,y_train_d)
        a = [[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,age]]
        a = np.array(a)
        y_pred_d = model_d.predict_proba(a)
        st.write(y_pred_d[0][1])
        if y_pred_d[0][1] <= 0.4:
            st.markdown('### You have low risk of getting diabetes')
        if y_pred_d[0][1] > 0.4 and y_pred_d[0][1] <= 0.42:
            st.markdown('### You have medium risk of getting diabetes')       
        if y_pred_d[0][1] > 0.42:
            st.markdown('### You have high risk of getting diabetes')



if nav == 'Kidney Disease':
    st.title('Kidney Disease Risk Prediction')
    st.markdown('''
    ### Here you can check the risk of having a Kidney Malfunction in the future.
    ''')   

if nav == 'Stroke':
    st.title('Stroke Risk Prediction')
    st.markdown('''
    ### Here you can check the risk of having a Stroke in the future.
    ''') 
    df_s['bmi'].fillna(df_s['bmi'].mean(),inplace = True)
    X_s = df_s.iloc[:,1:-1]
    y_s = df_s.iloc[:,-1]
    encoder = LabelEncoder()
    X_s['gender'] = encoder.fit_transform(X_s['gender'])
    X_s['ever_married'] = encoder.fit_transform(X_s['ever_married'])
    X_s['work_type'] = encoder.fit_transform(X_s['work_type'])
    X_s['Residence_type'] = encoder.fit_transform(X_s['Residence_type'])
    X_s['smoking_status'] = encoder.fit_transform(X_s['smoking_status'])
    X_train_s, X_test_s ,y_train_s, y_test_s= train_test_split(X_s,y_s,test_size=0.20, random_state=42)

    model_s = pickle.load(open(r's_clfrf.pkl','rb'))  
    age = st.number_input('Enter your age',step = 5) #put at #2
    gender = st.number_input('Gender (1= Male, 0 = Female)',step = 1) # put at #1
    hypertension = st.number_input('Do you have Hypertension? (1 = yes, 0 = no)',step = 1)
    heart_disease = st.number_input('Do you have any heart problem? (1 = yes, 0 = no)',step = 1)
    ever_married = st.number_input('Marital Status (1 = Married , 0 = Unmarried)',step=1)
    work_type = st.number_input('Work Type (0 = Private, 1 = Self-employed, 2 = Govt_job, 3 = children, 4 = Never_worked)',step = 1)
    residence_type = st.number_input('Residence type (0 = Urban, 1 = Rural', step=1)
    avg_glucose_level = st.number_input('Glucose level (a number between 100 to 200',step=10)
    bmi = st.number_input('Body Mass Index (a number between 32 to 65)', step = 5)
    smoking_status = st.number_input('Do you Smoke? (0 = formerly smoked, 1 = never smoked, 2 = smokes, 3 = Unknown)', step=1)
    model_s.fit(X_train_s,y_train_s)
    a = [[gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]]
    a = np.array(a)
    if st.button('Predict'):
        y_pred_s = model_s.predict_proba(a)
        st.write('Probaility of getting a stroke = ', y_pred_s[0][1])

