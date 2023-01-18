""" Page answers business requirement 2 """
#  flake8: noqa
import streamlit as st
from src.data_management import load_filter_test_data, load_ohe_data
from src.machine_learning.plot_sets import (
    dust_per_variable, parallel_plot_rul
    )


def page5_body():
    """ Defines the p5_feature_study page """

    # load data
    df = load_filter_test_data()
    df_ohe = load_ohe_data()

    # hard copied from feature study notebook
    vars_to_study = ['Dust_feed', 'Differential_pressure', 'Flow_rate',
                     'Dust_ISO 12103-1, A2 Fine Test Dust',
                     'Dust_ISO 12103-1, A3 Medium Test Dust',
                     'Dust_ISO 12103-1, A4 Coarse Test Dust']
    
    st.subheader('Filter Feature Study')
    st.success(
        f'**Project Business Requirement 2**\n\n'
        f'The client is interested in determining the **primary features that '
        f'correlate to Remaining Useful Life** of a replaceable part '
        f'so that the client can confirm the most relevant variables correlated '
        f'to its operational life.')
    
    # data inspection
    if st.checkbox('Inspect Filter Test Data'):
        st.write(
            f'* The dataset has {df.shape[0]} rows, {df.shape[1]} columns, '
            f'contained in {len(df["Data_No"].unique())} separate test bins.\n'
            f'* See 6 rows of the last observations in each bin.')

        st.write(df[df.Data_No != df.Data_No.shift(-1)].tail(6))

    st.write('---')

    # correlation study summary
    # vars_list = print(vars_to_study, sep='\n')
    st.write(
        f'A correlation study was conducted in the notebook to **better understand how '
        f'the variables are correlated to Remaining Useful Life** of a part. \n'
        f"Excluding '**Time**', the top 6 correlated variables are:"
        )
    st.write(vars_to_study)

    # Text sourced from on '06_Filter_Feature_Study' notebook - 'Conclusions and Next steps' section
    st.info(
        f'The correlation indications and plots below '
        f'indicate that: \n'
        f'* The rate of change in **RUL** is mostly affected by **dust feed** \n'
        f'* The rate of change in **RUL** is typically affected by **the change in differential pressure** \n'
        f'* The rate of change in **RUL** is often affected by **dust type** and /or **the rate of flow** \n'
        f'* The rate of change in **RUL** is affected more when the **coarseness of the dust** increases \n'
        f'* **Time** has been excluded as it is included in the [calculation of **RUL**](https://github.com/roeszler/filter-maintenance-predictor/blob/main/README.md#details-of-calculations) regardless \n'
    )

    # Code sourced from '06_Filter_Feature_Study' notebook - 'EDA on selected variables' section
    df_eda = df_ohe.filter(vars_to_study + ['RUL'])

    # Individual plots per variable
    target_var = st.selectbox('Dust Type per Variable?', ('Dust_ISO 12103-1, A2 Fine Test Dust', 'Dust_ISO 12103-1, A3 Medium Test Dust', 'Dust_ISO 12103-1, A4 Coarse Test Dust'))
    if st.checkbox('Run Variable Plots'):
        dust_per_variable(df_eda, target_var)
    
    # Parallel plot
    if st.checkbox('Run Parallel Plot of important variables to RUL'):
        st.write(
            f'* Note the bight colored **dust feed** information indicate the dust feed profiles that '
            f'have the greatest effect on remaining useful life. '
            f'The higher the feed rate, the greater negative effect on remaining useful life.')
        parallel_plot_rul(df_eda)
