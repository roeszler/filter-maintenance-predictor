#  flake8: noqa
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file

# load RUL files
rul_pipe = load_pkl_file('deployed/rfr_pipeline.pkl')
rul_feat_importance = plt.imread('deployed/features_importance.png')
rul_reg_evaluation = plt.imread('deployed/reg_eval_plot.png')
reg_eval_pic_test = plt.imread('deployed/reg_eval_pic_test.png')
reg_eval_pic_train = plt.imread('deployed/reg_eval_pic_train.png')
reg_eval_pic_validate = plt.imread('deployed/reg_eval_pic_validate.png')
cross_val_plot = plt.imread('deployed/cross_val_plot.png')
X_train = pd.read_csv('deployed/X_train_deployed.csv')
X_test = pd.read_csv('deployed/X_test_deployed.csv')
y_train = pd.read_csv('deployed/y_train_deployed.csv')
y_test = pd.read_csv('deployed/y_test_deployed.csv')
X_validate = pd.read_csv('deployed/X_validate_deployed.csv')
y_validate = pd.read_csv('deployed/y_validate_deployed.csv')


def page0_body():
    """ Testing Page """

    data = st.radio('Select Evaluation Data to View',
        (
            'Model Regression Images', 'Model Evaluation Metrics',
            'Cross Validation Metrics'
        )
    )

    if data == 'Model Regression Images':
        st.write('You selected Regression Images.')
        st.image(rul_reg_evaluation)
    elif data == 'Model Evaluation Metrics':
        st.write('You selected Model Evaluation Metrics')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Train Set')
            # st.image('https://res.cloudinary.com/yodakode/image/upload/YodaKode/freepik-com-Designed-by-stories-Freepik_wkfvq1.jpg')
            st.image(reg_eval_pic_train, width=600)
        with col2:
            st.write('Test Set')
            # st.image('https://res.cloudinary.com/yodakode/image/upload/YodaKode/freepik-com-Designed-by-stories-Freepik_wkfvq1.jpg')
            st.image(reg_eval_pic_test, width=600)
    else:
        st.write('You selected Cross Validation Metrics')
        st.image(cross_val_plot)
        st.image(reg_eval_pic_validate)
