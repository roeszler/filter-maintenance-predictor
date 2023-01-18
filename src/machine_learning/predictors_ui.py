""" Prediction functions for User Interfaces """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from src.data_management import load_pkl_file
import numpy as np

# load RUL files
rul_pipe = load_pkl_file(f'deployed/rfr_pipeline.pkl')
scaler = load_pkl_file(f'deployed/scaler.pkl')
rul_feat_importance = plt.imread(f'deployed/features_importance.png')
rul_reg_evaluation = plt.imread(f'deployed/reg_eval_plot.png')
X_train = pd.read_csv(f'deployed/X_train_deployed.csv')
X_test = pd.read_csv(f'deployed/X_test_deployed.csv')
y_train = pd.read_csv(f'deployed/y_train_deployed.csv')
y_test = pd.read_csv(f'deployed/y_test_deployed.csv')
X_validate = pd.read_csv(f'deployed/X_validate_deployed.csv')
y_validate = pd.read_csv(f'deployed/y_validate_deployed.csv')

def rul_regression_predictor():
    """
    Predictor based on Random Forest Regression
    """
    st.subheader(f'Remaining Useful Life (RUL)\n_(in relative time units)_:')
    X_diff_p = st.slider('Differential Pressure', 0.0, 540.0)
    X_dust_f = st.slider('Dust Feed', 60.0, 380.0)
    X_dust_s = st.selectbox('Dust Grain Size', ('0.9', '1.025', '1.2'))
    st.write('---')
    data = np.array([X_diff_p, X_dust_f, X_dust_s]).reshape(1,-1)
    
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, y_train)
    Y_pred = model.predict(np.array([data]).reshape(1,3))
    
    st.write(f'# RUL = {round(Y_pred[0],2)}')
    st.write(f'Differential Pressure: {X_diff_p}')
    st.write(f'Dust Feed Rate: {X_dust_f}')
    st.write(f'Dust Type: {X_dust_s}')
    st.write('Regressor: Random Forest')
    st.write('---')

if __name__=='__main__':
    rul_regression_predictor()
