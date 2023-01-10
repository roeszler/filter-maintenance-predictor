import streamlit as st
from PIL import Image

# streamlit run app.py
filter_image = Image.open('static/img/Forsta_High_Res-2394418179.png')

def page1_body():

    # option = st.selectbox(
    #     'How would you like to be contacted?',
    #     ('Email', 'Home phone', 'Mobile phone'))
    # st.write('You selected:', option)

    st.write(
        f'### Project Summary\n'
        # f'Page 1'
        )
    st.image(filter_image, caption='180 Series ForstaFilter', width=400)

    # text based on README file - 'Dataset Content' section
    st.info(
        f'**General Project Terms & Jargon**\n'
        f'* **The client** is the business we are providing a Machine Learning (ML) solution for.\n'
        f'* *The stakeholder** is a team, business or entity involved with the development of the machine '
        f'learning model.\n'
        f'* **A user** is a person or business looking to use the model to inform business decisions.\n '
        f'* **A prospect** is a potential new customer\n '
        f'* **The project** is the plan and delivery of a ML solution to meet a variety of requirements' 
        f'to Predict Maintenance of a replaceable part.\n\n'

        f'**Industry Specific Terminology**\n'
        f'* **A replaceable part** for this project, is considered a filter mat made out of ' 
        f'randomly oriented, non-woven fibre material.\n '
        f'* A **Life test** is the entire test cycle from the first instance of a Data_No to the last.\n '
        f'* The **Filter degradation process** is the gradual performance decline over time, '
        f'which can be quantified and used by statistical models.\n '
        f'* **Right censored data** is where “failure” has/will occur after the recorded time.\n '
        f'* **Filter failure** is when the **differential pressure** across the filter **exceeds 600 Pa**.\n '
        f'* **Zone of Failure** is the last 10% of RUL for that replacement part.\n'
        f'* **The Threshold** is the actual time when the experiment exceeded the threshold,'
        f'(used to define when the observations pass into the zone of failure).\n'
        
        f'**Project Dataset**\n\n'
        f'The dataset represents a collection of test measures obtained from the **Power Technique division** '
        f'of an **Industrial Solutions company**. \n''It contains testing data that relates to:\n'
        f'* The **change in performance of a replaceable filter part** '
        f'(like difference in pressure and estimated remaining useful life)\n'
        f'* **Feeding rates** (like dust feed, flow rate and sampling rate in seconds) and\n'
        f'* **Materials filtered** (type of dust fed into the testing system).'
        )
    
    # copied from README file - 'Business Requirements' section
    st.success(
        f'**Project Business Requirements**\n'
        f'1. The client is interested in understanding the patterns from the customer base '
        f'so that the client can learn the most relevant variables that are correlated to a '
        f'churned customer.\n\n'

        f'2. The client is interested in determining what are the primary features relating to RUL. '
        f'If so, the client is interested to know when the RUL reaches the last 10% of its remaining useful life. In addition, the client is '
        f'interested in learning the RUL for each type of dust in the testing database. '
        f'Based on that, present potential factors that could maintain and/or bring  '
        f'the prospect to a maximize RUL.'
        )
    
    # Link to README file, so the users can have access to full project documentation
    st.write(
        f'**For additional information, please visit the ML projects '
        f'[README](https://github.com/roeszler/filter-maintenance-predictor/blob/main/README.md) file.**')
    
    # st.snow()
    # st.balloons()
