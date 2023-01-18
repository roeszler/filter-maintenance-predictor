""" Project Summary Page """
#  flake8: noqa
import streamlit as st
from PIL import Image

filter_image = Image.open('static/img/Forsta_High_Res-2394418179.png')

def page1_body():
    """ Defines the p1_summary page """

    st.subheader(
        f'Project Summary\n'
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
    st.success('**Project Business Requirements**')
    st.success(
        f'1. The client is interested in understanding the patterns from a controlled testing '
        f'procedure so that they can **predict the current Reaming Useful Life (RUL) of any given '
        f'replaceable part** (in this case an industrial air filter).\n\n'
        )
    st.write(
    f'_From this prediction, the client hopes to make a determination of when replaceable '
    f'part is likely to reach the final 10% of its remaining useful life (known as the zone '
    f'of failure)_.\n\n'
    )
    st.success(
        f'2. The client is interested in determining what **the primary features that correlate to RUL**. ')
    st.write(
        f'_From this analysis, the client is interested in calculating the RUL for each type of dust '
        f'as specified in the testing database_.\n\n'
        )
    # Link to README file, so the users can have access to full project documentation
    st.warning(
        f'**For additional information, please visit the ML projects '
        f'[README](https://github.com/roeszler/filter-maintenance-predictor/blob/main/README.md) file.**')
