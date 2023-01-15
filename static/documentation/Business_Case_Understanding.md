# Business Case Sample
## What is the business objective requiring an ML solution?
We wish to change from a preventative maintenance business model to a predictive one to maximize the cost benefit of servicing equipment.

### Requirement 1
> We would like to investigate the power of a predictive ML solution to indicate the current remaining useful life of replaceable parts. In this case the replaceable part is an air filter.

### Requirement 2
> If a filter fails, or fails prematurely, which type of dust (cluster) does it belong to? 

### Requirement 3
> Which factors affect a filter's lifespan?

## Can any business objective be done with conventional data analysis?
* No. We can use conventional multiple linear regression analysis (MLRa) to discover trends over time, however;
* We wish to detect replacement part failure before it occurs. Conventional analysis does not easily indicate new patterns, accept new information or learn from the data.
* MLRa relies on a **non-collinear** set of independent variables, which is not always the case for failures of replaceable parts.

## Is data available for the training of the model, if yes which type of data?
* Yes, prospect information, which contains primarily numerical data (discrete and continuous),  with some categorical data.

### Is this data normally distributed?
Yes / No (z-distribution, gaussian distribution, continuous distribution? )

### If not, how are we going to collect the data?
* N/A. We have the data for this project supplied by the client.

## Does the customer need a dashboard or an API endpoint?
* The client has requested a Streamlit Dashboard.

## What does the client consider to be a success??
* A running dashboard meeting all business requirements, that include: 
    * A study that allows them to determine the remaining useful life of a replaceable part.
    * If a filter fails, or fails prematurely, which type of dust (cluster) does it belong to?
    * Receive an summary of the key features that the lifespan of a filter.

## Can you break down the project into Epics and User Stories?
* Yes, These can be seen mapped to the ML life cycle steps, which include:
    * Information gathering and data collection 
    * Data visualization, cleaning, and preparation
    * Model training, validation, and optimization. 
    * Dashboard planning, designing, and development
    * Dashboard deployment and release.

## Ethical or Privacy concerns?
* The client provided the data under a NDA, a non-disclosure agreement, therefore the data should be shared only with professionals that are officially involved in the project.

## Does the data suggest a particular model?
* A **Regression Analysis** where a determinative calculation of Remaining Useful Life is the target.
* A **Correlation Study** to observe the most important features that determine Remaining Useful Life.

## What are the project inputs and intended outputs?
**Inputs**:
> filter performance test information

**Outputs**:
> Predicted Remaining Useful Life.

> Summary of the primary variables that affect Remaining Useful Life.

> The type of dust does this prediction of Remaining Useful Life belong to.

## What level of prediction performance is needed?
For the regressors and correlation study, We have agreed with the stakeholder on commonly used **rule of thumb to start with**. We will try to achieve:
* **0.7** or higher for the Coefficient of Determination (R<sup>2</sup> score) as the **threshold for acceptance** into the model.

## How will the customer benefit?

Measurable Benefits
* Assess the likely current remaining useful life of a replacement part.
* Be informed when a part is likely to reach it's end of useful life based on use, not a timeline.

The client will also gain insights on 
* The variables that directly correlate to remaining useful life.
* The optimal replacement timings according to the dust types being filtered.
* Compare the performance of other filter types considered into the future.

---
_Return to the Filter Maintenance [Readme](https://github.com/roeszler/filter-maintenance-predictor/blob/main/README.md) document_.