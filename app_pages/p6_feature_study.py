""" Page answers business requirement 2 """
import plotly.express as px
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from src.data_management import load_filter_test_data, load_ohe_data
from feature_engine.discretisation import ArbitraryDiscretiser

def page6_body():
    st.write('This is page 6')

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
            f'* Find the last observations of the last 6 bins below.')

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
        f'* The rate of change in **RULe** is mostly affected by **dust feed** \n'
        f'* The rate of change in **RUL** is typically affected by **the change in differential pressure** \n'
        f'* The rate of change in **RUL** is often affected by **dust type** and /or **the rate of flow** \n'
        f'* The rate of change in **RUL** is affected more when the **coarseness of the dust** increases \n'
        f'* **Time** has been excluded as it is included in the [calculation of **RUL**](https://github.com/roeszler/filter-maintenance-predictor/blob/main/README.md#details-of-calculations) regardless \n'
    )

    # Code sourced from '06_Filter_Feature_Study' notebook - 'EDA on selected variables' section
    df_eda = df_ohe.filter(vars_to_study + ['RUL'])

    # Individual plots per variable
    target_var = st.selectbox('Dust Type per Variable?', ('Dust_ISO 12103-1, A2 Fine Test Dust', 'Dust_ISO 12103-1, A3 Medium Test Dust', 'Dust_ISO 12103-1, A4 Coarse Test Dust'))
    if st.checkbox('Run Variable Plot'):
        dust_per_variable(df_eda, target_var)
    
    # Parallel plot
    if st.checkbox('Run Parallel Plot of important variables to RUL'):
        st.write(
            f"* Information in bright orange and yellow indicates the profile of dust feed")
        parallel_plot_rul(df_eda)


# FUNCTIONS
# function created using '06_Filter_Feature_Study' notebook code - "Variables Distribution by RUL" section
def dust_per_variable(df_eda, target_var):

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        # if df_eda[col].dtype == 'object':
        if df_eda[col].dtype == df_eda[target_var].dtype:
            # plot_categorical(df_eda, col, target_var)
            pass
        else:
            plot_numerical(df_eda, col, target_var)


# function created using '06_Filter_Feature_Study' notebook code - "Variables Distribution by RUL" section
def plot_categorical(df, col, target_var):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    sns.countplot(data=df, x=col, hue=target_var, order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


# function created using '06_Filter_Feature_Study' notebook code - "Variables Distribution by RUL" section
def plot_numerical(df, col, target_var):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)

# function created using '06_Filter_Feature_Study' notebook code - "Parallel Plot" section
def parallel_plot_rul(df_eda):

    # hard coded from "disc.binner_dict_['RUL']"" result,
    tenure_map = [-np.inf, 31, 62, 93, 124, 155, 186, 217, 248, 279, np.inf]

    # sourced from '06_Filter_Feature_Study' notebook within the "Parallel Plot" section
    disc = ArbitraryDiscretiser(binning_dict={'RUL': tenure_map})
    df_parallel = disc.fit_transform(df_eda)

    n_classes = len(tenure_map) - 1
    classes_ranges = disc.binner_dict_['RUL'][1:-1]
    LabelsMap = {}
    for n in range(0, n_classes):
        if n == 0:
            LabelsMap[n] = f"<{classes_ranges[0]}"
        elif n == n_classes-1:
            LabelsMap[n] = f"+{classes_ranges[-1]}"
        else:
            LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"

    df_parallel['RUL'] = df_parallel['RUL'].replace(LabelsMap)
    fig = px.parallel_categories(
        df_parallel, color='Dust_feed')
    # to render chart
    # st.pyplot(fig)
    st.plotly_chart(fig)
