""" File manages the loading of datasets """
#  flake8: noqa
import streamlit as st
import pandas as pd
import joblib


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_filter_test_data():
    """
    Load test data for filter
    """
    df = pd.read_csv('deployed/dfCleanTotal.csv')
    return df


def load_ohe_data():
    """
    Load data from one hot encoder
    """
    df_ohe = pd.read_csv(f'deployed/dfOhe.csv')
    return df_ohe


def load_pkl_file(file_path):
    """
    Load .pkl files
    """
    return joblib.load(filename=file_path)
