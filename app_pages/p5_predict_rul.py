""" Page answers business requirement 1 """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_filter_test_data, load_pkl_file
from src.machine_learning.evaluate_sets import (
    clf_performance, regression_performance, regression_evaluation,
    regression_evaluation_plots
    )

def page5_body():
    st.write("This is page 5")

    # Version Selector
    version = st.selectbox('Select Version:', ('v1', 'v2'))
    st.write('Current Version:', version)

    # load RUL pipeline files
    rul_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_rul/{version}/RandomForestRegressor_pipeline.pkl")
    # rul_labels_map = load_pkl_file(
    #     f"outputs/ml_pipeline/predict_rul/{version}/label_map.pkl")
    rul_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_rul/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_rul/{version}/X_train.csv").dropna()
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_rul/{version}/X_test.csv").dropna()
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_rul/{version}/y_train.csv").dropna()
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_rul/{version}/y_test.csv").dropna()
    X_validate = pd.read_csv(
        f"outputs/ml_pipeline/predict_rul/{version}/X_validate.csv").dropna()
    y_validate = pd.read_csv(
        f"outputs/ml_pipeline/predict_rul/{version}/y_validate.csv").dropna()

    st.write("### ML Prediction Pipeline: Remaining Useful Life")
    # display pipeline training summary conclusions
    st.info(
        f"* We have an extremely strong Regressor model to predict RUL for a given "
        f"filter and/or dust type and/or an accurate way to measure differential pressure. "
        f"The model more than satisfied the business and project requirement of: an R² Score "
        f"> **0.7** of on **both the train and test sets**. \n"
        
        f"* The regressor performance was greater than +0.96 on both sets.\n"
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

        f"* The pipeline was tuned on **Dust Feed** rate, **Differential Pressure** and **Type of dust, "
        f"using the train and test sets, derived from a hybrid dataset where all the tests without RUL "
        f"were included with it calculated, if their test bin reached 600 pa or more. \n" 
        
        f"* We also demonstrated how to convert the target and predictor variables to classes within the "
        f"**Filter Feature Study** to answer Business Requirement 2. \n\n"
    )
    st.write("---")

    # show pipeline steps
    st.write("* ML pipeline to predict RUL.")
    st.write(rul_pipe)
    st.write("---")

    # show best features
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(rul_feat_importance)
    st.write("---")

    # evaluate performance on both sets
    st.write("### Pipeline Performance")
    # clf_performance(X_train=X_train, y_train=y_train,
    #                 X_test=X_test, y_test=y_test,
    #                 pipeline=rul_pipe,
    #                 label_map=rul_labels_map)
    
    regression_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=rul_pipe)
                    # label_map=rul_labels_map)
    
    regression_evaluation_plots(X=X_train, y=y_train, pipeline=rul_pipe)
    regression_evaluation_plots(X=X_test, y=y_test, pipeline=rul_pipe)
