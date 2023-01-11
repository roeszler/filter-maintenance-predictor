import streamlit as st

def page2_body():
    st.write('This is page 2')
    st.write('### Project Hypothesis and Validation')

    # conclusions taken from '05 Modelling & Evaluation' and '06 Filter Feature' notebooks
    st.success(
        f'* We suspect the Useful Life of a Filter shortens with Dust Feed: **Correct**. \n\n'
        f'The correlation analysis at the Filter Feature Study supports this hypothesis. \n\n'
        
        f'* We suspect the Useful Life of a Filter is highly affected by Flow rate: **Correct**. '
        f'The correlation analysis at the Filter Feature Study supports this hypothesis. \n\n'

        f'* We suspect the Useful Life of a Filter is highly affected by Dust type: **Moderate**. '
        f'The correlation analysis at the Filter Feature Study supports this hypothesis. \n\n'

        f'* A correlation and PCA analysis showed that the rate of change in differential pressure '
        f'is a strong indicator of a filters current Remaining Useful Life. \n\n'

        f'A short Useful Filter Life typically has A4 Coarse Dust with a high air flow rate, as '
        f'demonstrated by a Filter Feature Study. \n\n'
        
        f'These insights will be used by the business management and sales teams for further '
        f'discussions and investigations.'
    )
