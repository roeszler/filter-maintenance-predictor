import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from src.data_management import load_pkl_file
import numpy as np

# load RUL files
model = load_pkl_file(f'deployed/rfr_pipeline.pkl')
X_train = pd.read_csv(f'deployed/X_train_deployed.csv')
y_train = pd.read_csv(f'deployed/y_train_deployed.csv')
X_test = pd.read_csv(f'deployed/X_test_deployed.csv')
y_test = pd.read_csv(f'deployed/y_test_deployed.csv')

X_diff_p = X_train['Differential Pressure']
X_dust_f = X_train['Dust Feed']
X_dust_s = X_train['Dust']
data = np.array([X_diff_p, X_dust_f, X_dust_s]).reshape(1,-1)
model = RandomForestRegressor(n_jobs=-1)
model.fit(X_train, y_train)
Y_pred = model.predict(np.array([data]).reshape(1,3))
st.write(f'Predicted Remaining Useful Life: {Y_pred[0]}')

def page0_body():
    """ Testing Page """

    # st.write("* Train Set")
    # regression_evaluation(X=X_train, y=y_train, pipeline=model)
    # st.write("* Test Set")
    # regression_evaluation(X=X_test, y=y_test, pipeline=model)


# def regression_evaluation(X, y, pipeline):
prediction = model.predict(X_train)
R2_Score = r2_score(y_train, prediction).round(4)
st.write('R² Score:', r2_score(y_train, prediction).round(4))
mae = mean_absolute_error(y_train, prediction).round(4)
st.write('Mean Absolute Error:', mean_absolute_error(y_train, prediction).round(4))
medAe = median_absolute_error(y_train, prediction).round(4)
st.write('Median Absolute Error:', median_absolute_error(y_train, prediction).round(4))
mse = mean_squared_error(y_train, prediction).round(4)
st.write('Mean Squared Error:', mean_squared_error(y_train, prediction).round(4))
rmse =  np.sqrt(mean_squared_error(y_train, prediction)).round(4)
st.write('Root Mean Squared Error:', np.sqrt(
    mean_squared_error(y_train, prediction)).round(4))
st.write("\n")
