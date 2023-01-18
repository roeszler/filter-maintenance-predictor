""" Prediction functions for User Interfaces """
# flake8: noqa
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from src.data_management import load_pkl_file
import numpy as np

# load RUL files
rul_pipe = load_pkl_file('deployed/rfr_pipeline.pkl')
scaler = load_pkl_file('deployed/scaler.pkl')
rul_feat_importance = plt.imread('deployed/features_importance.png')
rul_reg_evaluation = plt.imread('deployed/reg_eval_plot.png')
X_train = pd.read_csv('deployed/X_train_deployed.csv')
X_test = pd.read_csv('deployed/X_test_deployed.csv')
y_train = pd.read_csv('deployed/y_train_deployed.csv')
y_test = pd.read_csv('deployed/y_test_deployed.csv')
X_validate = pd.read_csv('deployed/X_validate_deployed.csv')
y_validate = pd.read_csv('deployed/y_validate_deployed.csv')

def rul_regression_predictor():
    """
    Predictor based on Random Forest Regression
    """
    st.subheader('Calculate Remaining Useful Life (RUL)\n'
                 '_(in relative time units)_:')
    X_diff_p = st.slider('Differential Pressure', 0.0, 540.0)
    X_dust_f = st.slider('Dust Feed', 60.0, 380.0)
    X_time = st.slider('Filter Age', 0, 400)
    X_dust_s = st.selectbox('Dust Grain Size', ('0.9', '1.025', '1.2'))
    st.write('---')
    data = np.array([X_diff_p, X_dust_f, X_dust_s, X_time]).reshape(1, -1)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, y_train)
    Y_pred = model.predict(np.array([data]).reshape(1, 4))
    st.write(f'# RUL = {round(Y_pred[0], 2)}')
    st.write(f'Differential Pressure: {X_diff_p}')
    st.write(f'Dust Feed Rate: {X_dust_f}')
    st.write(f'Filter Age: {X_time}')
    st.write(f'Dust Type: {X_dust_s}')
    st.write('Regressor: Random Forest')
    st.write('---')


if __name__ == '__main__':
    rul_regression_predictor()
