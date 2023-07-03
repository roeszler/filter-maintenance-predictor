# Filter Replacement Predictor
## A Machine Learning Project

We consider a fictitious business case where **predictive analytics** is applied to a real life project. 

The scenario surrounds an industrial workplace looking to employ a **predictive maintenance model** in place of a **preventative** one as the primary strategy to maintain heavy equipment. 

This dataset has been selected as an example to train a Machine Learning model to predict the degradation process (gradual clogging) of a replaceable part (in this case a dust filter).

The project is aimed to provide an example of added value to a variety of industrial users looking to optimize maintenance programs and/or industrial processes that require some sort of filtering (the screening for ore concentrate, gas particles, materials in water or any other type of slurry).

* Repository link: [github.com/roeszler/filter-maintenance-predictor](https://github.com/roeszler/filter-maintenance-predictor)
* Deployed: [ml-maintenancepredictor.onrender.com](https://ml-maintenancepredictor.onrender.com/)

<details>
<summary style="font-size: 1.2rem;"><strong>Added value of a Predictive Model</strong><br><i style="font-size: 1rem;">(Dropdown List)</i></summary>
<br>

* Identify patterns that lead to potential problems or failures
* Identifying trends to aid future business decisions and/or investments
* Confidently predict the frequency of required maintenance
* Shorter equipment downtimes
* Actively preventing failures whilst optimizing the value of replacement parts
* Lowering the cost of preventive maintenance
* Avoiding the cost of repair/corrective maintenance
* Avoiding cost of replacing equipment
* Increasing the expected useful life of equipment
* Minimizing energy consumption

</details>

<!-- ### Summary of Terms & Jargon -->

<details>
<summary style="font-size: 1.2rem;"><strong>Summary of Terms & Jargon</strong></summary>

* **The client** is the business we are providing a Machine Learning (ML) solution for.
* **The stakeholder** is a team, business or entity involved with the development of the machine learning model.
* **A user** is a person or business looking to use the model to inform business decisions.
* **A prospect** is a potential new customer
* **The project** is the plan and delivery of a ML solution to meet a variety of requirements to Predict Maintenance of a replaceable part.
* The **replaceable part** for this project, is considered a filter mat made out of randomly oriented, non-woven otherwise unspecified 'fibre' material.
* **Differential Pressure** is a measure of the change in air pressures before and after the filtering process.
* **RUL** is Remaining Useful Life; the amount of time an asset (machine, replaceable part, filter, etc.) is likely to operate before it requires repair or replacement.
* The **filter degradation process** is the gradual performance decline over time, which can be quantified and used by statistical models.
* The **Model Threshold** is the minimum value of the **R<sup>2</sup> Regression Score** (coefficient of determination) that we will accept. It indicates how well a model fits the data and has been estimated by the client business team to be at **0.7**. Anything below this score will not be considered accurate enough for modelling.
* A **life test** is the entire test cycle from the first instance of a Data_No to the last.
* **Filter failure** is signified when the **differential pressure** across the filter **exceeds 600 Pa**.
* **Right censored data** is where “failure” has/will occur **after** the recorded time.
* **Zone of Failure** is the last 10% of RUL for that replacement part.

</details>


## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/prognosticshse/preventive-to-predicitve-maintenance) performed by Hagmeyer et al. (2021) at the Hochschule Esslingen University of Applied Sciences.

This has been shared in two files:  

* **Train_Data.csv** - 50% of the data without RUL to fit or train the model.
* **Test_Data.csv** - 50% of the data with RUL with the actual recorded values of time when the experiment exceeded the threshold.

The data is segmented into 50 life tests (data bins). The amount of observations in each bin varies depending on the input variables and a random time when the tail of the data was removed to produce a **right censored** dataset.

| Variable | Meaning | Units | Data Format | Data Type |
|---|---|---|---|---|
| **Data_no** | Test Number | Categorical Number 1 to 50 | Independent | Integer / Discrete Categorical |
 **Differential_pressure** | Pressure difference between upstream and downstream containment areas | Pascals (Pa = kg/m.s²) | Dependant | Floating point / Continuous |
| **Flow_rate** | quantity of air being moved | m<sup>3</sup>/sec | Independent | Floating point / Continuous |
| **Time** | Intervals between observations within a test as determined by sampling rate | 1/10th of a second | Independent | Floating point / Discrete (in this case) |
| **Dust_Feed** | velocity of the particle speed | mm<sup>3</sup>/s | Independent | Floating point / Continuous |
| **Dust** | 3 x types of ISO_12130 standardized dust ( A2 Fine, A3 Medium, A4 Coarse) | g/cm<sup>3</sup> | Independent | String / Discrete Number |
| **RUL** | Remaining Useful Life | Relative Units (ie 1 unit = 1 day) | Dependent | Floating point / Continuous |

A summary of the type, features and distributions of the data that is included and has been derivatively calculated can be found in a spreadsheet to help us coordinate the model creation process. This can be reviewed at the following link: [Summary Plan Cleaning & Engineering](https://docs.google.com/spreadsheets/d/1n5N7lzswrlgt83xe9iVQ6iirP7h8R0VvQ2Inr_JOKvA/edit?usp=sharing).

#### Further Information

<details>
<summary style="font-size: 1rem;"><strong>Classification measures</strong></summary>

* **Data_No** - Describes which one of the 50 ‘lifetime test’ sequences that the observations relate.
* **Dust** - Defines which type of three ISO_12130 standardized dust samples were used.
</details>

<details>
<summary style="font-size: 1rem;"><strong>Continuous Numerical observations</strong></summary>

* **Differential Pressure** - Indicates a change in air pressures before and after the filtering process.
* **Flow Rate** - Indicates the air flows following the filtering process.
</details>

<details>
<summary style="font-size: 1rem;"><strong>Ordinal Numerical Observations</strong></summary>

* **Dust Feed** - Indicates dust density fed into the system per unit of time. 
	* This Correlates to a numerical floating point number, which is a constant for each class:
		* A2_Fine - 0.900 g/m<sup>3</sup>
		* A3_Fine - 1.025 g/m<sup>3</sup>
		* A4_Fine- 1.200 g/m<sup>3</sup>
</details>

<details>
<summary style="font-size: 1rem;"><strong>Time</strong></summary>

* Indicates the discrete intervals between live test observations. 
* Time is a direct input to the RUL calculation and indicates the total number of live tests observed in the data set. i.e. 0.1s intervals:
    * Data_no 1 takes 36.6s to the end of that tests observations
        * this indicates that there has been 366 live tests in this sample set
    * Data_no 2 take 28.2s to the end of that tests observations
        * this indicates that there has been 282 live tests in this sample set
* This was confirmed by the .count() of the .unique() variables in each Data_no variable
</details>

### Considerations and Calculations:
#### **Filter medium**
The material used to filter the dust samples has been standardized across all tests. As a constant, it was not recorded as part of the datasets. Its properties were:

|Variable |Quantity | 
|---|---|
| Mean Fibre Diameter | 23 μm (micrometers) |
| Filter Area | 6131mm² |
| Filter Thickness | 20mm |
| Filter Packing Density | 0.014 - 0.0165 |
| Clean Filter Pressure Drop | 25 Pa at flow of 1080 m<sup>3</sup> / (h - m²) |

#### **Sampling rate**
Also not recorded as part of the supplied dataset, sampling rate is a constant, set at 10 Hz (or 0.1s per sample) for all tests and is reflected within the time measured at each observation.

#### **Zone of Failure**
A trade-off has to be made between discarding a replaceable part before it's entire remaining useful life is used, compared to the increased frequency of unplanned downtimes that occur in the last 10% of RUL.
* On Average, filter failure is observed to occur at the final 10% of the filter's RUL in the training data, and planned maintenance / replacement of the part would occur in this zone.
* At what point the final 10% zone commences will be a prediction based on the predicted RUL for each filer and the currently observed RUL.

#### **Important Note: Differential Pressure**

Differential pressure, the measure of the change in air pressures before and after the filtering process, seems to be a highly important variable to failure detection process, as it: 
* As a dependent variable, it seems to rely on a variety of factors (flow rate, dust size, filter type, time).
* Filter failure is considered to occur when this measure reaches **600 Pa**.
* Filter failure is positively correlated to RUL; 
	* i.e. **Differential Pressure = 600 Pa** indicates **RUL at 0 time units remaining**
* Depending on the rate of degradation toward the end of life for each type of filter, Differential Pressure may be a direct indicator of the **Zone of Failure**.

As the variable that a user would want to learn patterns, uncover relationships and predict using the rest of the dataset.

### Quantitative Calculations

| Variable | Meaning | Units | Data Format | Data Type |
|---|---|---|---|---|
| **Change in Differential Pressure** | Numerical value of change in pressure | Pascals (Pa = kg/m.s²) | Dependant | Floating point / Continuous | -->
| **Dust Density** | Numerical equivalent of dust density | g/cm<sup>3</sup> | Independent | Floating point / Continuous |
| **Dust Mass** | Mass of the dust density fed into the filter | grams | Independent | Floating point / Continuous |
| **Cumulative Dust Mass** | Cumulating dust mass fed into the filter over each test bin | grams | Independent | Floating point / Continuous |
| **Total Time of Test** | The cumulative time for the current test bin | seconds (T) | Independent | Floating point / Discrete |
| **RUL Test** | A check calculation of Remaining Useful Life from actual and calculated values in the set | Relative Units (ie 1 unit = 1 day) | Dependent | Floating point / Continuous |

#### Further Information

<details>
<summary style="font-size: 1.1rem;"><strong>Remaining Useful Life (RUL) (dropdown list)</strong></summary>

![RUL Image](https://res.cloudinary.com/yodakode/image/upload/Filter%20Maintenance/RUL_Image_pcs4v6.png)

**Remaining Useful Life** is classified as **the amount of time an asset (machine, replaceable part, filter, etc.) is likely to operate before it requires repair or replacement**. This is recorded in units relative to the test case, however units can represent Seconds, Minutes, Days, Miles, Cycles or any other quantifiable data. In practice:

* The RUL observations provided in the data are randomly right-censored. In other words, filter failure did not occur by the end of each test cycle. 
* This can be observed by the final observation of each test set in the **Differential_pressure** column. Each set does not reach the filter failure point of 600 Pa of differential pressure across the filter. This is the essence of right-censored. The data does not cover the full timeline to filter failure, however it ends somewhere before (at the right side) of this data set.

Reaming Useful Life has been chosen as the primary **Target variable** for initial investigations.

#### Calculating RUL 
* The authors idea behind the test data was, that at the point in time when the data in each set ends the remaining useful life (RUL) is to be estimated using definitive regression calculations.
	* i.e. we are interested in discovering **the rate of change in RUL**, and 
	* the subsequent shape and direction of the line(s) of best fit. 
		* This will allow us to predict RUL with some degree of certainty, based on the variables given.

For every observation, ‘RUL’ is the difference between:
* the maximum value of the ‘**time**’ in each test cycle (in this case to failure), and 
* the current observation ‘**time**’ at each test cycle

<p style="text-align: center; font-size: 1.2rem;">Remaining Useful Life (RUL) = Total time (cycles) to failure for each life test (T) - current time (t)</p>

* the RUL at the start of each test cycle was randomized to a number between the maximum time value and 3 minutes.

The resulting numerical data can then be used to observe the change in RUL and assist in producing an accurate model.

</details>

<details>
<summary style="font-size: 1.1rem;"><strong>Change in Differential Pressure</strong></summary>

Simple calculation of the change that occurs in Differential Pressure at each observation. The trend of the change in this value will indicate the trend of the change in the target variable `differential pressure` and ultimately allow us to predict when filter failure is likely to occur. 

</details>


<details>
<summary style="font-size: 1.1rem;"><strong>Calculations of Mass (g)</strong></summary>

The mass of the dust fed each life test is a factor of dust feed and dust density. These can be sourced from the data and has been calculated as:

We know:
<p style="text-align: center; font-size: 1.2rem;">
Mass Flow Rate = Volume per second × Density</br>
MFr = Q × ρ</p>


<p style="text-align: center; font-size: 1.2rem;">
Mass = Volume × Density</br>
m = V × ρ</p>

Where:
* Q = Volume flow rate (m<sup>3</sup>/s)
* V = Volume (m<sup>3</sup>)
* ρ = mass density of the dust (kg/m<sup>3</sup>)
* T = total number of seconds in each life test


Therefore: 
<p style="text-align: center; font-size: 1.2rem;">Mass = (((Q mm<sup>3</sup>/s) / 1000 ) × ρ g/cm<sup>3</sup> ) * T</p>

* =  (((1 mm<sup>3</sup>/s) / 1000 ) × 0.9 g/cm<sup>3</sup> ) * 1s
* =  0.001 cm<sup>3</sup>/s × 0.9 g/cm<sup>3</sup>
* = 0.0009 grams every test (in this example total test duration = 1s) 

</details>

<details>
<summary style="font-size: 1.1rem;"><strong>Right Censored Data</strong></summary>

By definition, right censored data is incomplete data. However, In this dataset we know that the end of life for a filter is when the differential pressure across a filter is 600 Pa. 

We can predict the remaining time at the end of life based on the trajectory of the change in differential pressure values provided.

The existence of right-censored data represents a challenge in this dataset to ensure we make the most use of the existing right-censored life data variables within the training data to predict RUL. This could be performed with conventional **determinative** data analysis, however the heuristic and versatile nature of machine learning makes it ideal in **predicting** this measure with greater statistical confidence.

The high number of observations in the supplied data (+78 K), although of relatively good quality that would minimize underfitting, it and the subsequent model, created issues with to the cloud storage limit of the deployment platform at **500MB** in total.

In consideration (which in a live project would be taken back to the business team) we decided to use the **test** dataset with +36K observations, as the primary database as this includes actual values of the target variable (Remaining Useful Life). We also included any value from the supplied **train** dataset that had observations within a bin that reached 600 Pa or more and hence RUL could be calculated.

* This resulted in a hybrid dataset of **40,112** observations that included RUL measures for us to model.

This information has then formed basis of the business requirements in this hypothetical project scenario.

#### Possible considerations to manage right censored data:
* [cross-validation : evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) (in sklearn),
* The function [cross_val_predict](https://scikit-learn.org/stable/modules/cross_validation.html#:~:text=The%20function%20cross_val_predict%20is%20appropriate%20for%3A) is appropriate for: Visualization of predictions obtained from different models.
* Model blending: When predictions of one supervised estimator are used to train another estimator in ensemble methods.

</details>

<details>
<summary style="font-size: 1.1rem;"><strong>Use of continuous vs discrete variables</strong></summary>

* Models designed for continuous variables can in many cases be used with discrete variables, however
* Models designed for discrete data used with continuous variables is rather a bad idea, as discrete variable models consider fixed, whole number values, with clear spaces between them. They consider data that cannot be measured or divided into smaller values, unlike continuous variables.

</details>

<details>
<summary style="font-size: 1.1rem;"><strong>Modification of the Data_No variable</strong></summary>

The datasets (df_test and df_train) are supplied in the following format:
* 50% df_test data (39,414 observations) have the actual RUL calculations included.
* 50% df_train data (39,420 observations) without RUL calculations.
* Both sets have had the tails of their data randomly removed. 
* This produces right-censored data (i.e. each set does not reach 600 Pa of differential pressure).

Before further dividing these datasets, we note that the categorical variable ‘Data_No’ restarts in the df_train set. To avoid confusion in later steps, or inadvertently correlate a Train set Data_No value of ‘1’ to be the same as a Test set Data_No value ‘1’, we manipulate the latter to be a **continuation** from the last value in the df_test set range with `df_test['Data_No'] + df_train_total_sets`.
</details>

<details>
<summary style="font-size: 1.1rem;"><strong>Splitting Datasets</strong></summary>

#### Test, Train, Validation Data
Recalling that the data for this analysis has been provided in a ‘pre-split’ format, we need to attend to the division prior to its use. 

Two main observations can bee seen in the distributions of these datasets 
1. The data us unevenly distributed between types of dust used in each test
2. The data has been pre-split into 50 test : 50 train with actual RUL values in the test dataset only.

#### 1. Uneven Data Distribution 

There appears to be a division of 48% Medium :  37% Coarse : 15% Fine in the **df_test** dataset and a division of 48% Medium :  28% Fine : 24% Coarse in the **df_train** dataset. This may be a refection of a typical use within the business cause, however may cause unnecessary noise to the models due to an over dominance of one or another dataset.

Strategies to account for this include:
* Assessing the central tendency (minimum | maximum | average | median | mode | skewness | kurtosis) of the bins with a greater proportion of data and choosing only the bins that represent the central tendency of the group, in a total data size close to the smaller proportion bins
* Assessing the randomness of what proportion the data has been **right censored** by comparing how far away last value of `differential_pressure` is from 600 Pa. This will give us a rudimentary idea as to the proportion of the bin that has been removed in each case. This can be seen as the `filter_balance` column in the datasets as a percentage (%) of remaining differential pressure value left to use/test.

Bins that sit further away from the measures of central tendency will be considered for removal to make the proportions of the data bins in each dataset more evenly distributed (i.e. at approximately 33% each).

#### 2. Pre-Split Data

The primary purpose of splitting the dataset into train, test and validation sets is to prevent the model(s) from overfitting. 

There is no optimal split percentage, however we would typically split the data in a way that suits the requirements and meets the model’s needs.

Ideally, the split of these subsets from a single dataset would be:
* Training Set = 60-80% (to fit the model)
* Validation Set = 10-20% (cross validation, compare models and choose hyperparameters)
* Test Set = 20-30%

![Dataset splitting Image](https://res.cloudinary.com/yodakode/image/upload/Filter%20Maintenance/Cross-validation_Performance_Evaluation_Flowchart_lnizoo.png)

As discussed, the supplied data has been split into **df_train**, **df_test** and **df_validate** by combining the supplied **test** and parts of the **train** databases into a hybrid, all with RUL values.

When selecting the best data from the **train** dataset, we noted that the distributions of tests via dust was not evenly represented in the data. 

We accounted for this by removing some of the data bins from the `dust` groups with lots more observations, that were not as normally distributed as the others. We extracted the total number of bins to create a total dataset containing roughly the same proportion of dust types as the smaller bins.

* This resulted in datasets (df_train, df_test and df_validate) with more evenly distributed proportion of data bins with more normalized data.

* Using the **df_test** as the source data, we split it into a proportion of bins closest to **Train 20%** | **Validation 20%** | **Test 20%**.
* The 60 | 20 | 20 split has been generated by the sklearn `train_test_split` function in two passes.
    * Acknowledging the hybrid dataset as the df source.
    * Splitting Test from the df by a random 20% split
	* Splitting Validate from the remaining data by 20% (calculated as 25% of the remaining data)
    * Resetting the index values.

* Now checking each data of the data sets to be in a ratio of the entire data set 40112
    * df_train - 60% (24066, 13)
	* df_validate - 20% (8023, 13)
    * df_test - 20% (8023, 13)

</details>


## Business Requirements

### Synopsis (Hypothetical)

> _As a Data Analyst, we have been requested by the Power Technique division (the client) to provide actionable insights and data-driven recommendations to a corporation that manufactures industrial tools and equipment._

> _This client has a substantial customer base in oil and gas and offshore industries, as well as power plants and surface and underground mining. They are interested in promoting the management of preventative maintenance and understanding how the industrial sales team could better interact with prospects as to the benefits of transitioning from a preventative model of maintenance to a predictive one._

### Business Case Surveys 
To determine the number and depth of ML Models required to meet the stakeholders requirements, an survey of the business needs and wants has been conducted. This will help us define stakeholders, a clear expression of their requirements from an ML solution.

This process is developed in a **business case** that can be applied to each ML model. These consider:
* The Model objective
* The Outcome 
* The Metrics 
* The Output
* The Heuristic Information (ie educated guesses, trial and error, industry rules of thumb)
* The Data
* Dashboard Design

The information for this process has been collected from the stakeholders as 
* A Survey - [Business Case Questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSfIjxD0Ki9793LTQ2szr3-qWKXUsMbQS1AhM80BCAvltxmu4A/viewform) and 
* A sample of all survey responses summarized for stakeholder **review** and **acceptance** can be seen in the attached [Business Case Understanding](static/documentation/Business_Case_Understanding.md) document.

### Business Requirements

From the above process, we confirm that stakeholders are interested in:

1. Using a predictive model to **determine the current Reaming Useful Life (RUL) of any given replaceable part** (in this case an industrial air filter).

	* _From this prediction, the client hopes to make a determination of when replaceable part is likely to reach the final 10% of its remaining useful life (known as the zone of failure)_.

2. The client is interested in determining **the primary features that correlate to Remaining Useful Life** of a replaceable part so they can confirm the most relevant variables correlated to its operational life.

	* _From this analysis, the client is interested in calculating the RUL for each type of dust as specified in the testing database_.

These requirements can now be evaluated against the dataset provided to devise the type of ML models to be delivered by this project.

---

## Hypothesis and how to validate?
* We suspect Remaining Useful Life can be projected from **the supplied data**.
    * A correlation study can help in this investigation
* We suspect the Useful Life of a Filter shortens with **Dust Feed**.
    * A predictive Regression model and correlation study can help with this investigation
* We suspect the Useful Life of a Filter is highly affected by **Flow rate**.
	* A predictive Regression model and correlation study can help with this investigation
* We suspect the Useful Life of a Filter is highly affected by **Dust type**.
	* A predictive Regression model and correlation study can help with this investigation

### What is the predominant class of variables?
The data is predominantly comprised of **continuous** data in **equal proportions**. With calculated variables included, they are classified as:

<details>
<summary style="font-size: 1.1rem;"><strong>Dependant Variables</strong></summary>

* Remaining Useful Life (target variable) = Continuous Numerical _therefore we choose_ Regression modeling
* Differential Pressure
* Cumulative Mass of Filtered Dust
</details>

<details>
<summary style="font-size: 1.1rem;"><strong>Independent variables</strong></summary>

* Flow Rate
* Time
	* Time interval (Reflecting sampling rate)
	* Total time to Filter Failure (at 600Pa Differential Pressure)
* Dust Feed
* Dust Type
	* Dust Mass
	* Grain Size
</details>

### Type of Learning Model

If the RUL target variable is: **a Continuous Number**.
* We will therefore consider a variation of Regression, Regression with PCA and/or Classification.

* A Supervised Learning Model used where the model has a target variable:
    * We will consider a **multiple regression model** which is supervised and multi-dimensional ([Business Cases 1](#business-requirements)) 
    * Should a regressor not perform well, we will run a **classification model** for each.

* We will consider a **correlation and visualization study** for this to meet the requirements of [Business Case 2](#business-requirements).

The type of regression algorithm we will evaluate are:
* Adaptive Boosting Regressor
* Decision Tree Regressor
* Extra Trees Regressor
* Gradient Boosting Regressor
* Linear Regressor
* Random Forest Regressor
* Stochastic Gradient Descent Regressor
* Extreme Gradient Boosting Regressor

## The rationale to map the business requirements to the Data Visualizations and ML tasks

* **Requirement 1** : [Predict Current RUL](https://github.com/roeszler/filter-maintenance-predictor/blob/main/jupyter_notebooks/05_ModelingAndEvaluation_RUL.ipynb) : Regression and Correlation Analysis
    * We want to **predict the RUL of a filter** and receive a binary response to indicate a filters current RUL depending on the variables that impact the prediction the most**.
    * We want to build a multiple regression model or Regression model + Principle Component Analysis or change the ML task to classification depending on the regressor performance.

* **Requirement 2** : [Correlations for Maximizing RUL](https://github.com/roeszler/filter-maintenance-predictor/blob/main/jupyter_notebooks/06_FilterFeatureStudy.ipynb)
    * We will inspect the data related to the RUL.
    * We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to RUL.
    * We will plot the main variables against RUL to visualize insights.

## Planned Pipeline to ML Model

<details>
<summary style="font-size: 1rem;">See: <strong>ML Pipeline Process</strong> for visualization of the planned process of developing an ML Model</summary>

![ML Pipeline Process](https://res.cloudinary.com/yodakode/image/upload/Filter%20Maintenance/ML_Pipeline_Process_sutrea.png)

</details>

## ML Business Case
### Business Case 1 : [Predict Current RUL](https://github.com/roeszler/filter-maintenance-predictor/blob/main/jupyter_notebooks/05_RUL_ModelingAndEvaluation.ipynb)
#### Regression Model
* Section for discussion of each Notebook on:
	* Statement of Objectives
    * Data Exploration & Manipulation
		* Create Pipeline
		* Split Data
		* Optimal Hyperparameter Search
    * Regression on Optimal Hyperparameters 
    * EDA on selected variables
    * Conclusions and Next Steps

* We want an ML model to predict the Remaining Useful Life of a replaceable part (an industrial air filter) based on controlled test data from the client, which doesn't include measures of dust mass per observation, cumulative mass, or total test time to failure.
* The target variable is a continuous numerical format and is sporadically right censored throughout a sample of test bins. We consider a supervised uni-dimensional regression model.
* The target variable units are in relative time units defined by each business scenario.
* The model success metrics are
	* At 0.7 or more for R<sup>2</sup> Score, on train, test and validation sets
	* The ML model is considered a failure if:
		* After 12 months of usage, the model's predictions are 50% off more than 30% of the time. 
		* i.e. a prediction is greater than >50% off if the model predicted 10 units and the actual value was less than <5 units or more.
* The output is defined as a continuous value for Remaining Useful Life in relative time units to the input (seconds, minutes, days, months, years etc.).
* Heuristics: Currently, there is a determinative historical data approach to determine the Remaining Useful Life for a replaceable part. This model attends to creating a predictive approach to add value to the maintenance programs within the business.
* The training data to fit the model comes is a sample that represents the same data that would be sourced from the Customer (A Power Technique division an Industrial Services Company). 
* This dataset contains about 78 thousand test observations of filter performance.
	* The training data represents:
		* Values of Remaining USeful Life (RUL) are provided or can be determinatively calculated and
		* Filter data where the cumulative change in differential pressure is greater than 0
	* Target Feature: Remaining USeful Life (RUL)
	* Features: all other variables, excluding Data Bin Number (Data_No) when modelling.


### Business Case 2 : [RUL Correlations](https://github.com/roeszler/filter-maintenance-predictor/blob/main/jupyter_notebooks/06_FilterFeatureStudy.ipynb)
#### Correlation Study
* We will inspect the data related to the RUL.
* We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to RUL.
* We will plot the main variables against RUL to visualize insights.
* We may use these factors to confirm our input variables in a machine learning model to predict RUL into the future.
* By using a correlation study to identify the most important factors, the client could improve the accuracy of their predictions and make more informed investment decisions. This could ultimately lead to decreased equipment down times, optimal use of replaceable parts, better coordination of workforce and ultimately increased profits and a competitive advantage in the market.

## Project Management

[Project Sprints](https://github.com/roeszler/filter-maintenance-predictor/milestones)
* [Sprint 1 - Collect Information & Data](https://github.com/roeszler/filter-maintenance-predictor/milestone/6?closed=1)
* [Sprint 2 - Data Visualisation, Cleaning & Preparation](https://github.com/roeszler/filter-maintenance-predictor/milestone/5?closed=1)
* [Sprint 3 - Train, Validate & Optimise Models](https://github.com/roeszler/filter-maintenance-predictor/milestone/4?closed=1)
* [Sprint 4 - Dashboard Plan, Design & Develop](https://github.com/roeszler/filter-maintenance-predictor/milestone/3?closed=1)
* [Sprint 5 - Deploy Dashboard & Release](https://github.com/roeszler/filter-maintenance-predictor/milestone/2?closed=1)
* [Sprint 6 - Bugs, Refactor & Document](https://github.com/roeszler/filter-maintenance-predictor/milestone/1?closed=1)

### User Stories
Respository Link : [github.com/users/roeszler](https://github.com/users/roeszler/projects/6)


## Dashboard Design

### Page 1: [Project Summary](https://github.com/roeszler/filter-maintenance-predictor/blob/main/app_pages/p1_summary.py)
* Project Terms & Jargon
* Describe Project Dataset
* Describe Business Requirements


### Page 2: [Project Hypothesis](https://github.com/roeszler/filter-maintenance-predictor/blob/main/app_pages/p2_project_hypothesis.py)
* Prior to the analysis, this page described each project hypotheses, the conclusions, and how to validated each. 
* Following the data analysis, we can report that:
	1. We suspect Remaining Useful Life can be projected from **the supplied data*.
		* **Correct**. The Correlation Analysis at the Filter Feature Study supports this hypothesis.
	2. We suspect the Useful Life of a Filter shortens with **Dust Feed**.
		* **Correct**. The Correlation Analysis at the Filter Feature Study supports this hypothesis.
	3. We suspect the Useful Life of a Filter is highly affected by **Flow rate**.
		* **Correct**. The Correlation Analysis at the Filter Feature Study supports this hypothesis.
	4. We suspect the Useful Life of a Filter is highly affected by **Dust type**.
		* **Not Significant**. The Correlation Analysis at the Filter Feature Study fails to support this hypothesis.

	* These insights will be referred to the business management and sales teams for further discussions, investigations required and conclusions.

* Page 3: [Predict Remaining Useful Life](https://github.com/roeszler/filter-maintenance-predictor/blob/main/app_pages/p3_rul_predictor.py)
	* This page intends to inform the associated probability of remaining useful life with a variety of input variables.
	* States business requirements
	* Set of widgets relating to the profiles of the most relevant input data.
		* Each set of inputs is related to a given ML task to predict Remaining Useful Life.
		* Sliders and select box that automatically serves the input data to our ML pipeline, and predicts the Remaining useful life of an industrial filter. 
	* It also shows a summary of the inputs selected and regression algorithm used. 


* Page 4: [ML Model: Remaining Useful Life](https://github.com/roeszler/filter-maintenance-predictor/blob/main/app_pages/p4_model_rul.py)
	* Considerations and conclusions after the pipeline is trained
	* Present ML pipeline steps
	* Feature importance
	* Pipeline performance

* Page 5: [ML Analysis: Filter Feature Study](https://github.com/roeszler/filter-maintenance-predictor/blob/main/app_pages/p5_feature_study.py)
	* State business requirement 2
	* Checkbox: data inspection on Remaining Useful Life database. 
		* This displays the last ten rows of the last observations in each bin of the data)
	* Display the top six correlated variables to RUL and the conclusions.
	* Select Box: To choose between the three types of dust to visualize 
	* Checkbox: Individual plots showing the dust types for each correlated variable
	* Checkbox: Parallel plot using RUL and correlated Dust variables


	* Considerations and conclusions after the pipeline is trained
	* Present ML pipeline steps

## Bugs
* **Resolved**: The size of the original dataset creates a subsequent `.pkl` file size that is larger than the repository likes to store or the claud application [Heroku](https://www.heroku.com/) allows to be uploaded int its slug. This has been rectified by reducing the number of observations in the model uploaded to the cloud app.
* **Resolved**: When running the custom `consolidate_df_test_dust()` the streamlit application indicates a `ValueError: Input X contains NaN` where there is none in existence. This value of reducing the balance of the dataset between the three dust variables is not indicated by the correlation analysis and / or absence of the model over / underfitting. This has been resolved by including the total  hybrid version of the data to be used without equalizing the distribution of dust within the data bin observations.  
* There are no unfixed bugs in the deployed version of the project.

## Possible Future Features
* Inclusion of training data that has **all RUL** testing bins with **all values** that extend to the end of a filters useful life.
* Facility to access a widely accessible testing procedure that indicates if a filter is currently useable or not-useable.
* Application able to alert clients the optimal time to change an air filter, in relative RUL time units.
* Capacity to confirm the industry rule of thumb to replace at a 10% RUL zone is correct or does the data indicate something else.
	* Confirm the optimal replacement time (ORT) where ORT is considered the cost benefit trade off between maximizing useful life and minimizing the risk of failure.
* A complete list of future features can be found at [Future Features](https://github.com/roeszler/filter-maintenance-predictor/issues?q=is%3Aissue+label%3A%22future+release%22+is%3Aclosed).
	* Confirm the optimal replacement time (ORT) where ORT is considered the cost benefit trade off between maximizing useful life and minimizing the risk of failure.
* A complete list of future features can be found at [Future Features](https://github.com/roeszler/filter-maintenance-predictor/issues?q=is%3Aissue+label%3A%22future+release%22+is%3Aclosed).


## Deployment

This project was deployed using the GitPod + Jupyter Environment, with a Streamlit Framework into Heroku. The steps to deploy are as follows:

* Fork or clone the [Code-Institute-Org: python-essentials-template](https://github.com/Code-Institute-Org/gitpod-full-template)
* Click the Use this template to create a clone in GitHub
* Follow Display environment settings below:

### Display Environment (GitHub / GitLab / BitBucket)

The application has been deployed to GitHub pages. 

<details>
<summary>
The steps to deploy a clone of the GitHub repository...
</summary>

* Create / open an existing repository for the project with the name of your choice on your GitHub, GitLab or Bitbucket account page.
* Navigate within the GitHub repository you chose, and then navigate to the "settings" tab, which displays the general title.
* On the left hand navigation menu, I selected the "pages" option midway down the menu.
* At the top of the pages tab, the source section drop-down menu changed to select the branch: "main" with the folder selected as `"/(root)"`
* Committed to the save and waited a few moments for the settings to coordinate with the server. 
* On refresh of the browser, the dedicated ribbon changed to the selected web address, indicating a successful deployment.

> The live application link can be found here - https://maintenance-predictor.herokuapp.com/

> The accessible GitHub repository for this application is https://github.com/roeszler/filter-maintenance-predictor
</details>

### Development Environment (GitPod)
The application has been deployed to GitPod pages during development. 

<details>
<summary >
The steps to deploy the project from GitHub to GitPod... 
</summary>

* In the GitHub, GitLab or Bitbucket account page where you created a repository for the project, navigate to the tab titled `'<> Code'`
* From here, navigate to the button on the top right of the repository navigation pane titled 'Gitpod'.
* If you press this it will create a new GitPod development environment each time.
</details>

<details>
<summary >
Alternatively, if you have already created the GitPod environment for your project...
</summary>

* In the browser’s address bar, prefix the entire URL with [gitpod.io/#](https://gitpod.io/#) or [gitpod.io/workspaces](https://gitpod.io/workspaces) and press Enter. This will take you to a list of workspaces that have been active within the past 14 days.
* Search for the workspace you wish to work on and access the link to it that lies within the pathway `https://gitpod.io/`.
* Sign in to the workspace each time with [gitpod.io/#](https://gitpod.io/#) using one of the listed providers (GitHub / GitLab / BitBucket) and let the workspace start up.
* On navigating to the workspace for the first time, it may take a little while longer than normal to initially install all it needs. Be patient.
* It is recommend that you install the GitPod browser extension to make this a one-click operation into the future.
</details>

### Heroku

The live link to the application is: [https://maintenance-predictor.herokuapp.com/](https://maintenance-predictor.herokuapp.com/)

<details>
<summary >
The project was deployed to Heroku using the following steps...
</summary>

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.

</details>

## Credits 

### Main Data Analysis and Machine Learning Libraries
* [Jupyter](https://jupyter.org/)
	* Is a development environment for notebooks, code, and data, allowing users to configure and arrange workflows in data science, scientific computing, computational journalism, and machine learning.
* [Kaggle](https://www.kaggle.com/)
	* A subsidiary of [Google](https://en.wikipedia.org/wiki/Google), Kaggle is a community of data scientists and machine learning practitioners that allows users to find and publish publicly accessible data sets to explore and build models, work with other professionals and enter competitions to solve data science challenges.
* [Matplotlib](https://matplotlib.org/)
	* A more comprehensive library for creating static, animated, and interactive visualizations in [Python](https://www.python.org/about/).
* [NumPy](https://numpy.org/)
	* A [Python](https://www.python.org/about/) programming language library that adds support for large, multi-dimensional arrays and matrices. I t also has a large collection of high-level mathematical functions to operate on these arrays.
* [Pandas](https://pandas.pydata.org/)
	* I an open source data analysis and manipulation tool, built on the [Python](https://www.python.org/about/) programming language to be a fundamental high-level building block for doing practical, real world data analysis.
* [Python](https://www.python.org/about/)
	* a high-level, general-purpose programming language emphasizing code readability with the use of indentation.
* [Seaborn](https://seaborn.pydata.org/)
	* Is a [Python](https://www.python.org/about/) data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
* [Scikit-learn](https://scikit-learn.org/stable/index.html)
	* Is a free software machine learning library based on the [Python](https://www.python.org/about/) programming language, featuring a variety of classification, regression and clustering algorithms such as support-vector machines, random forests, gradient boosting, k-means and DBSCAN. It is designed to interoperate with the [Python](https://www.python.org/about/) numerical and scientific libraries [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/).
* [Streamlit](https://streamlit.io/)
	* A [Python](https://www.python.org/about/) based environment that transforms data scripts into shareable web apps with minimal front end coding required.
* [TensorFlow](https://www.tensorflow.org/)
	* Developed by the [Google Brain Team](https://research.google.com/teams/brain/?hl=he), TensorFlow is a free and open-source software library for machine learning and artificial intelligence.
* [Xgboost](https://xgboost.readthedocs.io/en/stable/)
	* XGBoost (eXtreme Gradient Boosting) is an open-source software library which provides the gradient boosting framework for for C++, Java, [Python](https://www.python.org/about/), R, Julia, Perl and Scala.

### Main Development Environments and Editors
* [GitHub](https://github.com/)
	* Allows a variety of benefits to create, document, store, showcase and share a project in development.
* [GitPod](https://www.gitpod.io/)
	* Provides a relatively secure workspace to code and develop software projects in a remotely accessible cloud based platform.
* [Heroku Platform](https://www.heroku.com/platform)
	* Provides a platform for deploying and running [Python](https://www.python.org/about/) based apps.
* [Cloudinary](https://cloudinary.com/) Image and Video Upload, Storage, Optimization and Content Display Network.
* [Pillow](https://pypi.org/project/Pillow/) part of the [Python](https://www.python.org/about/) Imaging Library (PIL) adding image processing capabilities to the [Python](https://www.python.org/about/) interpreter.
* [Lucidchart Flowchart Diagrams](https://www.lucidchart.com/pages/)
	* A diagramming application that allows the mapping and creation of flowcharts to visualize design workflows.


### Content 

* The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/prognosticshse/preventive-to-predicitve-maintenance) performed by **Hagmeyer et al. (2021)** at the Hochschule Esslingen University of Applied Sciences.
* Primary and additional Python coding was studied and reworked from modules provided through the Code Institute's [Diploma in Full Stack Software Development](https://codeinstitute.net/se/full-stack-software-development-diploma/), [W3 Schools](https://www.w3schools.com/), [Stack overflow](https://stackoverflow.com/), [mozilla.org](https://developer.mozilla.org/en-US/docs/Web/JavaScript) and [GeeksForGeeks](https://www.geeksforgeeks.org/).
* Images were altered from original layout using [pixlr](https://pixlr.com/).
* Images have been stored for delivery using the [cloudinary](https://cloudinary.com/) content delivery network.
* Code styling and error detection by systematic code refactoring following a run of the `python3 -m flake8` command to evoke the [flake8](https://flake8.pycqa.org/en/latest/) style enforcement tool.

### Literature

* Blog: '[Considerations for the calculation of remaining useful life](https://www.ada-mode.com/blog/how-to-calculate-remaining-useful-life)', Ada Mode, 2021.
* For development of general understandings and descriptors found at [StatQuest] (https://statquest.org/), [Towards Data Science](https://towardsdatascience.com/?gi=af0c840d68e1), [Medium](https://medium.com/tag/data-science), [PlaygroundGPT](https://beta.openai.com/playground) and [Papers With Code](https://paperswithcode.com/).
* "[User Stories Applied: For Agile Software Development](https://books.google.se/books?id=SvIwuX4SVigC&lpg=PR13&ots=VrYbfbwVRQ&dq=User%20Stories%20Applied%3A%20For%20Agile%20Software%20Development%22%20by%20Mike%20Cohn&lr&pg=PR13#v=onepage&q=User%20Stories%20Applied:%20For%20Agile%20Software%20Development%22%20by%20Mike%20Cohn&f=false)" by Mike Cohn, 2004.
* "[Scrum: The Art of Doing Twice the Work in Half the Time](https://books.google.se/books?id=ikl9AwAAQBAJ&lpg=PP1&dq=Scrum%3A%20The%20Art%20of%20Doing%20Twice%20the%20Work%20in%20Half%20the%20Time%22%20by%20Jeff%20Sutherland&pg=PP1#v=onepage&q=Scrum:%20The%20Art%20of%20Doing%20Twice%20the%20Work%20in%20Half%20the%20Time%22%20by%20Jeff%20Sutherland&f=false)" by Jeff Sutherland, 2014.
* "[Agile Estimating and Planning](https://books.google.se/books?id=BuFWHffRJssC&lpg=PT29&ots=WqfgykNZJl&dq=Agile%20Estimating%20and%20Planning&lr&pg=PT29#v=onepage&q=Agile%20Estimating%20and%20Planning&f=false)" by Mike Cohn, 2005.
* "[The Lean Startup: How Today’s Entrepreneurs Use Continuous Innovation to Create Radically Successful Businesses](https://books.google.se/books?id=tvfyz-4JILwC&lpg=PA1&ots=8J6aE83msZ&dq=The%20Lean%20Startup%3A%20How%20Today%E2%80%99s%20Entrepreneurs%20Use%20Continuous%20Innovation%20to%20Create%20Radically%20Successful%20Businesses&lr&pg=PA1#v=onepage&q=The%20Lean%20Startup:%20How%20Today%E2%80%99s%20Entrepreneurs%20Use%20Continuous%20Innovation%20to%20Create%20Radically%20Successful%20Businesses&f=false)" by Eric Ries, 2011.
* The Agile Alliance (https://www.agilealliance.org/) is a non-profit organization that promotes Agile methodologies.
* Scrum.org (https://www.scrum.org/) is a website that provides information and resources on the Scrum framework.
* The Scaled Agile Framework (SAFe) (https://www.scaledagileframework.com/) is a methodology for managing and completing projects using Agile methodologies at an enterprise level.

### Media

* Remaining Useful Life image sourced from [Stratadata](https://www.stratada.com/remaining-useful-life-rul-prediction/) Nov 2022.
* Industrial filter image sourced from [Forsta Filters](https://www.forstafilters.com/wp-content/uploads/2014/06/Forsta_High_Res.jpg) Nov 2022.
* Original charts designed and developed using [Lucidchart Flowchart Diagrams](https://www.lucidchart.com/pages/).

## Acknowledgements
* Thank the people that provided support through this project.
	* My Amazing Mentor Rohit
	* My Awesome Tutor Niel
	* My Incredibly Supportive Wife K xxx

---
__COPYRIGHT NOTICE__ :
 *The Filter Maintenance predictor site is a functional program intended for educational purposes at the time of coding. Notwithstanding, it has been written as a proof of concept and invitation to treat for employers and stakeholders into the future. Copyrights for code, ideas, concepts and materials strictly lies with Stuart Roeszler © 2023. All rights reserved.*
