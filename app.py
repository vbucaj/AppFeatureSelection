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
                This app will enable the user to perform feature selection for both Classification and Regression problems.
                The output will be interactive Boxplots or ROC curves, respectively.
                 """)

    st.write(""" 
        <h3> How to use the App</h3>

        Below we provide a brief description of some of how to use the App and a few of the tasks that this app can perform.

        All of the widgets for this app will appear on the left side-bar.
        <ul>
        <li> Start by selecting the page <b> Feature Selection </b> under <b> Menu</b>
        <li> Check the box <b> Feature Selection</b>
        <li> Upload a file. Currently only the following formats are supported <b>.csv, .xlsx, .xls</b></li>
        <li> You can provide a <b> Sheet Name</b>. If you do not provide one, it will be assumed that it is the first one.
        <li>Once the file is uploaded it will be converted to a Pandas DataFrame. 
        For the time being, the last column of the DataFrame <b> MUST</b> be the response variable. In the future we will make
        this feature smarter and allow the user to specify which column is the response variable.</li>
        <li> You will be given the option to select a subset of the columns of the DataFrame</li>
        <li> Scenario 1: If you select <b> YES:</b>
        <ul>
        <li> <b>Select Columns:</b> You will be prompted to select multiple columns from your DataFrame. Remember, you <b> MUST</b> select
        the column of the target/response variable last.</li>
        <li><b>Drop & Imputation:</b> You can specify a percentage limt for dropping the columns that miss a higher percentage of data. 
        The rest of the missing data will be imputed by by first building an empirical distribution for each feature and then 
        imputing the missing values by values randomly sampled from the corresponding empirical distribution.</li>
        <li> Next you will be prompted to selecting a Problem Type</li>
        <li> Next you will be required to select the desired ML Method/Model that will be used by the algorithm to perform the analysi. </li>
        <li> After picking the ML Model, you will be prompted to specify a few parameters for that particular Method.</li>
        <li> Finally, to display the results check the <b> Run Feature Selection </b> button on the main body.
        <ul>
            <li>A brief description of what the algorithm is doing: First, it will randomly split the data in a training and test set.
            Then, it will train the model on the training set. Next, it will shuffle the values of each feature in the test set 
            and measure one of the pre-specified metrics to observe the decrease in the predictive performance of the model</li>
        </ul></li>
        <li> Depending on whether it is a Regression or Classification problem, a Boxplot or ROC curve plot will be built</li>
        </ul></li>
        <li> Scenario 2: If you select <b> NO:</b>
        <ul>
        <li> Only the <b> Select Columns</b> step from above will be skipped, the rest is identical.</li>
        </ul>
        </li>


        </ul>




if __name__ == "__main__":
    main()
