import streamlit as st
from app_pages.multi_page import MultiPage

# load pages scripts
from app_pages.p1_summary import page1_body
from app_pages.p2_project_hypothesis import page2_body
from app_pages.p3_interface_rul import page3_body
# from app_pages.p4_ import page4_body
from app_pages.p5_predict_rul import page5_body
from app_pages.p6_feature_study import page6_body
# from app_pages.p7_ import page7_body

app = MultiPage(app_name= 'Filter Maintenance Predictor') # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page('Project Summary', page1_body)
app.add_page('Project Hypothesis', page2_body)
app.add_page('Remaining Useful Life Interface', page3_body)
# app.add_page('Page 4', page4_body)
app.add_page('ML Prediction Model: Remaining Useful Life', page5_body)
app.add_page('Filter Feature Study', page6_body)
# app.add_page('ML Model 3: Page 7', page7_body)

app.run() # Run the  app