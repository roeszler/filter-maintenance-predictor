""" Functions for data evaluation """
# flake8: noqa 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    median_absolute_error, classification_report, confusion_matrix
    )

# Regression Reports
def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Handles the data for input into train and test regression reports
    """
    st.info("Train Set")
    regression_evaluation(X_train, y_train, pipeline)

    st.info("Test Set")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    """
    Displays summary of important regression variables
    """
    prediction = pipeline.predict(X)
    st.write('R² Score:', r2_score(y, prediction).round(4))
    st.code(r2_score(y, prediction).round(4))
    st.write('Mean Absolute Error:', mean_absolute_error(y, prediction).round(4))
    st.code(mean_absolute_error(y, prediction).round(4))
    st.write('Median Absolute Error:', median_absolute_error(y, prediction).round(4))
    st.code(median_absolute_error(y, prediction).round(4))
    st.write('Mean Squared Error:', mean_squared_error(y, prediction).round(4))
    st.code(mean_squared_error(y, prediction).round(4))
    st.write('Root Mean Squared Error:', np.sqrt(mean_squared_error(y, prediction)).round(4))
    st.code(np.sqrt(mean_squared_error(y, prediction)).round(4))
    st.write('\n')


def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
    """
    Creates regression evaluation plots
    """
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title('Train Set')

    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predictions')
    axes[1].set_title('Test Set')

    plt.show()


# Classification Reports
# code copied from "Modeling and Evaluation" notebooks
def confusion_matrix_and_report(X, y, pipeline, label_map):
    """
    Creates a confusion matrix report from X and y inputs
    """
    prediction = pipeline.predict(X)

    st.write('#### Confusion Matrix')
    st.code(pd.DataFrame(confusion_matrix(y_true=prediction, y_pred=y),
                         columns=[["Actual " + sub for sub in label_map]],
                         index=[["Prediction " + sub for sub in label_map]]
                         ))

    st.write('#### Classification Report')
    st.code(classification_report(y, prediction, target_names=label_map), "\n")


# code copied from "Modeling and Evaluation" notebooks
def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    """
    Handles inputs for confusion matrix reporting for train and test datasets
    """
    st.info("Train Set")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)
    st.info("Test Set")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)


def consolidate_df_test_dust(df_test):
    """
    Consolidated this specific df_test data into three equal dust classes
    """
    # Calculating standard deviation
    std_group = df_test.groupby('Data_No')['Differential_pressure'].std()
    df_test['std_DP'] = df_test['Data_No'].map(std_group)

    # Calculating coefficient of variation
    var_group = df_test.groupby('Data_No')['Differential_pressure'].apply(lambda data: np.std(data, ddof=1) / np.mean(data, axis=0) * 100)
    df_test['cv_DP'] = df_test['Data_No'].map(var_group)

    # Calculating median
    median_group = df_test.groupby('Data_No')['Differential_pressure'].median()
    df_test['median_DP'] = df_test['Data_No'].map(median_group)

    # Get count of entries for each Data_No
    bin_sum = df_test.groupby('Data_No')['Data_No'].count()

    # Map the count of entries for each Data_No
    df_test['bin_size'] = df_test['Data_No'].map(bin_sum)

    # Filter rows with Dust
    dust_A3 = df_test[df_test['Dust'] == 1.025]
    dust_A4 = df_test[df_test['Dust'] == 1.200]

    # Select rows which are not the last ones
    filter_A3 = dust_A3[dust_A3.Data_No != dust_A3.Data_No.shift(-1)]
    filter_A4 = dust_A4[dust_A4.Data_No != dust_A4.Data_No.shift(-1)]

    # Sort values by filter_balance in ascending order
    df_test_A3 = filter_A3.sort_values(by='filter_balance', ascending=True)
    df_test_A4 = filter_A4.sort_values(by='filter_balance', ascending=True)

    # Compute cumulative sum
    df_test_A3['c_sum'] = df_test_A3['bin_size'].cumsum()
    df_test_A4['c_sum'] = df_test_A4['bin_size'].cumsum()

    bin_no = df_test_A3['Data_No'].head(3)
    dn_fb = df_test_A3.loc[df_test_A3['Data_No'].isin(bin_no), 'Data_No'].head(14).sort_values(ascending=True).reset_index(drop=True)
    df_test_cleaned_A3 = df_test[df_test['Data_No'].isin(bin_no)]
    df_test_cleaned_A3.head(14).style.hide(['Time', 'Dust_feed', 'Flow_rate', 'Dust', 'mass_g', 'cumulative_mass_g', 'Tt'], axis="columns")

    # Repeat fot A4 Coarse Dust
    bin_no = df_test_A4['Data_No'].head(4)
    bin_no.to_frame()

    # Get all values from df_test that are in bin_no
    df_test_copy = df_test
    df_test_cleaned_A4 = df_test_copy[df_test_copy['Data_No'].isin(bin_no)]

    # Create dataframes for each dust class
    dust_A2 = df_test[df_test['Dust'] == 0.900]
    dust_A3 = df_test_cleaned_A3
    dust_A4 = df_test_cleaned_A4

    # Concatenate the dataframes
    global df_test_even_dist
    df_test_even_dist = pd.concat([dust_A2, dust_A3, dust_A4], ignore_index = True)

    # Replace df_test with df_test_even_dist
    df_test_even_dist = df_test_even_dist.reset_index(drop=True)

    # Plot a bar graph of the dust classes
    category_totals = df_test_even_dist.groupby('Dust')['Differential_pressure'].count().sort_values()
    category_totals.plot(kind="barh", title='Proportion of Dust Classes in "df_test_even_dist"\n', xlabel='\nObservations', ylabel='Dust Class')
