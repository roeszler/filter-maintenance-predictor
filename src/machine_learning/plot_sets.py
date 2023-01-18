""" Functions for data evaluation """
# flake8: noqa
import plotly.express as px
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.discretisation import ArbitraryDiscretiser


# FUNCTIONS
# function created using '06_Filter_Feature_Study' notebook code - "Variables Distribution by RUL" section
def dust_per_variable(df_eda, target_var):
    """
    Calculates if dust per variable and plots it
    """
    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == df_eda[target_var].dtype:
            pass
        else:
            plot_numerical(df_eda, col, target_var)


# function created using '06_Filter_Feature_Study' notebook code - "Variables Distribution by RUL" section
def plot_categorical(df, col, target_var):
    """
    Plots categorical variables
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    sns.countplot(data=df,
                  x=col,
                  hue=target_var,
                  order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


# function created using '06_Filter_Feature_Study' 
# notebook code - "Variables Distribution by RUL" section
def plot_numerical(df, col, target_var):
    """
    Plots numerical values
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


# function created using '06_Filter_Feature_Study' 
# notebook code - "Parallel Plot" section
def parallel_plot_rul(df_eda):
    """
    Parallel plot of RUL as a range
    """
    # hard coded from "disc.binner_dict_['RUL']" result,
    rul_map = [-np.inf, 31, 62, 93, 124, 155, 186, 217, 248, 279, np.inf]

    # sourced from '06_Filter_Feature_Study' notebook within 
    # the "Parallel Plot" section
    disc = ArbitraryDiscretiser(binning_dict={'RUL': rul_map})
    df_parallel = disc.fit_transform(df_eda)

    n_classes = len(rul_map) - 1
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
    st.plotly_chart(fig)
