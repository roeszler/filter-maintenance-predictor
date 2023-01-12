import streamlit as st
from app_pages.multi_page import MultiPage

# load pages scripts
from app_pages.p1_summary import page1_body
from app_pages.p2_project_hypothesis import page2_body
from app_pages.p3_interface_rul import page3_body
from app_pages.p4_predict_rul import page4_body
from app_pages.p5_feature_study import page5_body
# from app_pages.p7_ import page7_body

app = MultiPage(app_name= 'Filter Maintenance Predictor') # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page('Project Summary', page1_body)
app.add_page('Project Hypothesis', page2_body)
app.add_page('Remaining Useful Life Interface', page3_body)
app.add_page('ML Model: Remaining Useful Life', page4_body)
app.add_page('ML Analysis: Filter Feature Study', page5_body)

app.run() # Run the  app