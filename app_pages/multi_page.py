""" Page structures the streamlit defaults """
import streamlit as st


class MultiPage:

    def __init__(self, app_name) -> None:
        """ Initiate app """
        self.pages = []
        self.app_name = app_name
        st.set_page_config(
            page_title=self.app_name,
            page_icon="💻",
        )

    # Create class method to add pages to object
    # Output is none
    def add_page(self, title, func) -> None:

        self.pages.append({"title": title, "function": func})

    # Method to run the object
    def run(self):

        st.title(self.app_name)
        # add a sidebar menu, consisting of radio buttons
        # Situated on the left-hand side of the page.
        # With a radio button for each of the pages.
        # Each radio button will be labeled with the
        # title of the page.
        page = st.sidebar.radio('Menu', self.pages,
                        format_func=lambda page: page['title'])
        page['function']()
