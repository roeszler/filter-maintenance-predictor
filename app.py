from app_pages.multi_page import MultiPage

# load pages scripts
from app_pages.p1_summary import page1_body
from app_pages.p2_project_hypothesis import page2_body
from app_pages.p3_rul_predictor import page3_body
from app_pages.p4_model_rul import page4_body
from app_pages.p5_feature_study import page5_body

app = MultiPage(app_name='Filter Maintenance Predictor')


# Pages
app.add_page('Project Summary', page1_body)
app.add_page('Project Hypothesis', page2_body)
app.add_page('Predict Remaining Useful Life', page3_body)
app.add_page('ML Model: Remaining Useful Life', page4_body)
app.add_page('ML Analysis: Filter Feature Study', page5_body)

app.run()
