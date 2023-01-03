import streamlit as st
from PIL import Image

# streamlit run app.py
filter_image = Image.open('static/img/Forsta_High_Res-2394418179.png')

def page1_body():

    st.write(
        f'### Project Summary\n'
        f'Page 1'
        )
    st.image(filter_image, caption='180 Series ForstaFilter')

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **customer** is a person who consumes your service or product.\n"
        f"* A **prospect** is a potential customer.\n"
        f"* A **churned** customer is a user who has stopped using your product or service.\n "
        f"* This customer has a **tenure** level, the number of months this person " 
        f"has used our product/service.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset represents a **customer base from a Telco company** "
        f"containing individual customer data on the products and services "
        f"(like internet type, online security, online backup, tech support), "
        f"account information (like contract type, payment method, monthly charges) "
        f"and profile (like gender, partner, dependents).")

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/roeszler/filter-maintenance-predictor/blob/main/README.md).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in understanding the patterns from the customer base "
        f"so that the client can learn the most relevant variables that are correlated to a "
        f"churned customer.\n"
        f"* 2 - The client is interested in determining whether or not a given prospect will churn. "
        f"If so, the client is interested to know when. In addition, the client is "
        f"interested in learning from which cluster this prospect will belong in the customer base. "
        f"Based on that, present potential factors that could maintain and/or bring  "
        f"the prospect to a non-churnable cluster."
        )

        