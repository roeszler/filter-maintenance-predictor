import streamlit as st
from app_pages.multi_page import MultiPage

# load pages scripts
from app_pages.p1_summary import page1_body
from app_pages.p2_ import page2_body
from app_pages.p3_ import page3_body
from app_pages.p4_ import page4_body
from app_pages.p5_ import page5_body
from app_pages.p6_ import page6_body
from app_pages.p7_ import page7_body

app = MultiPage(app_name= "Filter Maintenance Predictor") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Project Summary", page1_body)
app.add_page("Page 2", page2_body)
app.add_page("Page 3", page3_body)
app.add_page("Page 4", page4_body)
app.add_page("ML Model 1: Page 5", page5_body)
app.add_page("ML Model 2: Page 6", page6_body)
app.add_page("ML Model 3: Page 7", page7_body)

app.run() # Run the  app