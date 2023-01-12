""" File manages the loading of datasets """
import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_filter_test_data():
    # df = pd.read_csv('outputs/datasets/transformed/dfCombinedHybrid.csv')
    # df = pd.read_csv('outputs/datasets/cleaned/dfCleanTotal.csv')
    df = pd.read_csv('deployed/dfCleanTotal.csv')
    return df


def load_ohe_data():
    # df_ohe = pd.read_csv(f'outputs/datasets/transformed/dfOhe.csv')
    df_ohe = pd.read_csv(f'deployed/dfOhe.csv')
    return df_ohe


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)
