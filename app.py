import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import featselection

from minitoolboxVB import FeatureSelection


st.title("Feature Selection App")

def main():
    createlayout()
    #side_bar_homepage()

# homepage side bar
def side_bar_homepage():
    st.sidebar.title("About Author")
    st.sidebar.image("images/profile_pic.jpg", use_column_width=False,width=200)
    st.sidebar.info("""My name is Valmir Bucaj. I am an Assistant Professor of Mathematics at the U.S. Military Academy, West Point. 
    I am passionate about Data Science, Machine Learning, and AI.
    You can connect with me via [LinkedIn](https://www.linkedin.com/in/valmir-bucaj-phd-7731a093/),
     [Github](https://github.com/vbucaj), or [Website](https://vbucaj.github.io/MA477-course/)""")

# app layout
def createlayout():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Please select a page", ["Homepage", "Feature Selection"])
    if app_mode == 'Homepage':
        homepage()
        side_bar_homepage()
    elif app_mode == "Feature Selection":
        featselection.load_page()

# content of homepage
def homepage():
    #st.image("images/logo1.png",use_column_width=True)
    st.markdown("> Selecting the most important features is a vital part of Data Analysis")
    st.write("""
            This app will enable the user to perform feature selection for both Classification and Regression models.
            The output will be eye pleasing boxplots or ROC and AUC curves.
             """)





if __name__ == "__main__":
    main()
