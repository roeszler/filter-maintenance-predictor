""" Page Allows users to interact with model """
import streamlit as st
from src.machine_learning.predictors_ui import rul_regression_predictor

def page3_body():
    """ Defines the p3_interface_rul page """
    st.write("This is page 3")

    # Run Predictor
    rul_regression_predictor()

