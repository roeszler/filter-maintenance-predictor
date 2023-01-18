""" Page Allows users to interact with model """
#  flake8: noqa
import streamlit as st
from src.machine_learning.predictors_ui import rul_regression_predictor


def page3_body():
    """ Defines the p3_interface_rul page """

    # Information Panel
    st.subheader('RUL Prediction Interface')
    st.info(
        f'* The client is interested in determining the remaining useful life of a replaceable '
        f'machine part (in this case an industrial air filter). to make a determination of when '
        f'a given replaceable part is likely to reach the final 10% of its remaining useful '
        f'life (known as the zone of failure).\n\n'

        f'* The client also interested in determining what the primary features that correlate '
        f'to RUL and calculating the RUL for each type of dust as specified in the testing database'
    )
    st.write("---")
    # Run Predictor
    rul_regression_predictor()
