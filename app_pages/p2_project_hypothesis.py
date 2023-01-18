""" Page states the Project Hypotheses """
import streamlit as st
#  flake8: noqa


def page2_body():
    """ Defines the p2_project_hypothesis page """

    st.subheader('Project Hypothesis and Validation')
    # conclusions taken from '05 Modelling & Evaluation'
    # and '06 Filter Feature' notebooks
    st.write(
        f'We suspect Remaining Useful Life can be projected from **the supplied data**'
        )
    st.success('**Correct**:'
        f'The Correlation Analysis at the Filter Feature Study supports this hypothesis. \n\n'
        )
    st.write(
        f'We suspect the Useful Life of a Filter shortens with **Dust Feed**')
    st.success('**Correct**:'
        f'The Correlation Analysis at the Filter Feature Study supports this hypothesis. \n\n'
        )
    st.write(
        f'We suspect the Useful Life of a Filter is highly affected by **Flow rate**')
    st.success('**Correct**:'
        f'The Correlation Analysis at the Filter Feature Study supports this hypothesis. \n\n'
        )
    st.write(
        f'We suspect the Useful Life of a Filter is highly affected by **Dust type**')
    st.warning('**Mixed**:'
        f'The Correlation Analysis at the Filter Feature Study moderately supports this hypothesis. \n\n')
    st.subheader('Observations')
    st.success(
        f'* A correlation and PCA analysis showed that the rate of change in **differential pressure** '
        f'and dust feed rate are strong indicators of a filters current **Remaining Useful Life**. \n\n')
    st.success(
        f'* A Filter Feature Study demonstrated that a Shortened Useful Filter Life typically '
        f'corresponds with **A4 Coarse Dust** and/or a **higher air flow rate**.\n\n')
    st.warning(
        f'* A Filter Feature Study demonstrated that the variations in Dust Type filtered affect '
        f'its power to assist in the prediction of Remaining Useful Life.\n\n')
    st.info(
        f'* These insights will be referred to the business management and sales teams for further '
        f'discussions, investigations required and conclusions.'
        )
