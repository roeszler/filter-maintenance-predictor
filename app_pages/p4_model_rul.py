""" Page answers business requirement 1 """
#  flake8: noqa
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file


def page4_body():
    """ Defines the p4_predict_rul page """

    # load RUL files
    rul_pipe = load_pkl_file(f'deployed/rfr_pipeline.pkl')
    rul_feat_importance = plt.imread(f'deployed/features_importance.png')
    rul_reg_evaluation = plt.imread(f'deployed/reg_eval_plot.png')
    reg_eval_pic_test = plt.imread(f'deployed/reg_eval_pic_test.png')
    reg_eval_pic_train = plt.imread(f'deployed/reg_eval_pic_train.png')
    reg_eval_pic_validate = plt.imread(f'deployed/reg_eval_pic_validate.png')
    cross_val_plot = plt.imread(f'deployed/cross_val_plot.png')
    X_train = pd.read_csv(f'deployed/X_train_deployed.csv')
    X_test = pd.read_csv(f'deployed/X_test_deployed.csv')
    y_train = pd.read_csv(f'deployed/y_train_deployed.csv')
    y_test = pd.read_csv(f'deployed/y_test_deployed.csv')
    X_validate = pd.read_csv(f'deployed/X_validate_deployed.csv')
    y_validate = pd.read_csv(f'deployed/y_validate_deployed.csv')
    st.subheader('ML Prediction Pipeline: Remaining Useful Life')
    # display pipeline training summary conclusions
    st.info(
        f"* We have a strong Regressor model to predict RUL for a given "
        f"filter and/or dust type and/or an accurate way to measure differential pressure. "
        f"The model more than satisfied the business and project requirement of: an R² Score "
        f"> **0.7** of on **both the train and test sets**. \n"
        
        f"* The regressor performance was greater than +0.94% on train, test and validation sets.\n"
        f"* We notice that 'Coarse dust' class and 'Dust feed' rate also adds reasonable "
        f"performance. The absence of definitive information on filter type is a "
        f"limitation of this project.\n"

        f"* We would recommend the R² Score tolerance may be better served between a tolerance up to" 
        f"**0.85** to **0.95** as the performance of the train and test sets exceeds 90 with most "
        "regression models. This information would be fed back to the business team for consideration "
        "as the subject matter experts in this field. \n\n"
        )
    
    st.info(
        f"* We proceeded to perform a Regressor + PCA and primarily for project demonstration purposes. "
        f"This increased the accuracy of the models and demonstrated how we would reduce model over or "
        f"under fitting. \n"

        f"* The pipeline was tuned on **Dust Feed** rate, **Differential Pressure** and **Type of dust**, "
        f"using the train and test sets. A hybrid dataset was used where all the tests without RUL "
        f"were calculated derivatively and included if their test bin reached **600 pa** or more. \n" 
        
        f"* We also demonstrated how to convert the target and predictor variables to classes within the "
        f"**Filter Feature Study** to answer Business Requirement 2. \n\n"
        )
    st.write("---")

    # show pipeline steps
    st.info('ML pipeline to predict RUL')
    st.code(rul_pipe)
    st.write("---")

    # show best features
    st.info('The features the model was trained and their importance.')
    st.write(X_train.columns.to_list())
    st.image(rul_feat_importance)
    st.write("---")

    # data inspection
    st.info('Inspect Input Data')
    version = st.selectbox('Select set:', ('X_train', 'y_train', 'X_test', 'y_test', 'X_validate', 'y_validate'))
    df_sample = pd.read_csv(f'deployed/{version}_deployed.csv')
    st.write(f'* The dataset has {df_sample.shape[0]} rows, {df_sample.shape[1]} columns\n')
    st.write(df_sample.head(6))
    st.write('---')

    # evaluate performance on both sets
    st.subheader('Pipeline Performance')
    data = st.radio('Select Evaluation Data to View',
        ('Model Evaluation Metrics', 'Model Regression Images', 'Cross Validation Metrics')
        )

    if data == 'Model Regression Images':
        st.write('You selected Regression Images.')
        st.image(rul_reg_evaluation)
    elif data == 'Model Evaluation Metrics':
        st.write('You selected Model Evaluation Metrics')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Train Set')
            st.image(reg_eval_pic_train, width=600)
        with col2:
            st.write('Test Set')
            st.image(reg_eval_pic_test, width=600)
    else:
        st.write('You selected Cross Validation Metrics')
        st.image(cross_val_plot)
        st.image(reg_eval_pic_validate)
