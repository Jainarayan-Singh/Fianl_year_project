import streamlit as st
import pickle
import numpy as np
import sklearn

# page icon
st.set_page_config(page_title='Pile Blows Count Prediction', page_icon=':hammer:', layout='wide')

tab1,tab2 = st.tabs(['Pile Blows Count Prediction', 'About App'])

with tab1:
    # import the model
    model = pickle.load(open('ex.pkl','rb'))
    df = pickle.load(open('df.pkl','rb'))

    st.title("Pile Blows Count Prediction")



    depth = st.number_input('Depth (in metres)', 2.00, 40.00)
    qc = st.number_input('qc (MPa)', 1.00, 100.00)
    ENTRHU = st.number_input('ENTRHU (Normalized between 0-1)', 0.050, 0.990)
    hammer_energy = st.number_input('Hammer energy (Normalized between 0-1)', 0.100, 0.900)
    n_blows = st.number_input('No of blows', 1, 2500)


    if st.button('Blows Count Prediction'):

        query = np.array([depth, qc, ENTRHU, hammer_energy, n_blows])

        query = query.reshape(1,5)
        st.text("The predicted Blows Count is :  " + str(int((model.predict(query)[0]))))

with tab2:
    st.info('Pile driveability prediction is the process of estimating the resistance that a pile will face during \n'
            'installation. It involves the use of analytical approaches, numerical models, and empirical data \n'
            'to determine the capacity of the pile to withstand various loads during installation. \n'
            'Here, Model is made by using ensemble technique (Extra Trees Regression) of machine learning.')

    st.warning('User should be careful while taking the direct values from this model as these will \n'
               'be the tentative values not the actual/exact for designing of pile foundations as \n '
               'values need to be further recheck through experiment. ThankYou!')
    st.text('Created by JAINARAYAN SINGH, PIYUSH SHEJWAR, CV KALYAN' )
