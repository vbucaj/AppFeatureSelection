import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from minitoolboxVB import FeatureSelection, DropImpute,Outliers
#from tkinter.filedialog import askopenfilename

#MLE Methods
from sklearn.model_selection import train_test_split
#from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error,mean_squared_log_error, r2_score, auc, roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import defaultdict
from scipy.stats import gaussian_kde




# controller
def load_page():
    sidebar_info()
    body()

# the side bar information for the feature selection page
def sidebar_info():
    st.sidebar.subheader('Feature Selection')
    st.sidebar.markdown("""
                  This is a simple module I wrote designed
                  to perform feature selection for both Regression
                  and Classification models, and display the results
                  visually in the form of a Boxplot (for Classification Problems) or
                  ROC/AUC Curve (for Regression Problems).
                   """)

# main content of the app
def body():
    if st.sidebar.checkbox('Feature Selection'):

        def file_selector(folder_path='.'):
            filenames = os.listdir(folder_path)
            selected_filename = st.sidebar.selectbox('Select a file', filenames)
            return os.path.join(folder_path, selected_filename)


        filename = file_selector()
        st.write('You selected `%s`' % filename)
        sheet_name = st.sidebar.text_input(label='Sheet Name')

        def create_dataframe(filename,sheetname=None):


            if filename[-4:]=='.csv':
                return pd.read_csv(filename,sheet_name=sheetname)
            elif filename[-4:]=='.xls':
                return pd.read_excel(filename,sheet_name=sheetname)
            elif filename[-5:]=='.xlsx':
                return pd.read_excel(filename,sheet_name=sheetname)
            else:
                raise ValueError('Your file is not CSV or Excel')

        #data = create_dataframe(filename, 'Edited Data')

        if st.checkbox('Display Data:'):

            data = create_dataframe(filename, sheet_name)
            st.dataframe(data)

            select_subset=st.sidebar.selectbox("Select a Subset of DataFrame",['Yes','No'])
            st.write('You selected:',select_subset)

            if select_subset=='Yes':

                if st.checkbox("Select a Subset of Columns"):

                    selected_columns = st.multiselect(
                        'Select the columns you want to keep:',
                        data.columns)
                    data = data[selected_columns]
                    st.write('You selected:', selected_columns)

                    if st.checkbox("Display the Selected DataFrame:"):

                        st.dataframe(data)

                    if st.checkbox('Display a Heat Map of Missing Data Before Drop and Imputation'):
                        sns.heatmap(data.isnull(),cbar=False,yticklabels=False)
                        plt.xticks(fontsize=8, rotation=30)
                        st.pyplot()

                    st.sidebar.markdown("Preprocessing: Drop & Impute")

                    missing_col_perc=st.sidebar.slider("Limit for % of missing column values",1,90,5)
                    st.write("All columns missing more than {}% of values will be dropped. The rest will be imputed.".format(missing_col_perc))

                    # missing_row_num=st.sidebar.slider('Limit # of missing row values',1,len(data.index))
                    # st.write("All rows missing more than {} values will be dropped. The rest will be imputed".format(missing_row_num))

                    drop=DropImpute(data)

                    drop.impute_values(feat_min_perc=missing_col_perc,inplace=True)

                    if st.checkbox("Display Data After Drop and Imputation"):
                        st.dataframe(data)

                    if st.checkbox('Display a Heat Map of Data after Drop and Imputation'):
                        sns.heatmap(data.isnull(),cbar=False,yticklabels=False)
                        plt.xticks(fontsize=8,rotation=30)
                        st.pyplot()

                    problem_type = st.sidebar.selectbox('Select the Problem Type', ['Regression', 'Classification'])
                    st.write('You selected:', problem_type)

                    if problem_type == 'Regression':

                        model_type=st.sidebar.selectbox('Select ML Model:',
                                                        ['LinearRegression','RandomForestRegressor'])
                        st.write('You selected:',model_type)

                        if model_type=='RandomForestRegressor':
                            num_trees=st.sidebar.slider('Number of Trees:',1,100,50)
                            st.write('You selected {} trees'.format(num_trees))

                            depth=st.sidebar.slider('Max Depth:',1,100,10)
                            st.write('Max Depth is:',depth)

                            selected_regressor=RandomForestRegressor(n_estimators=num_trees,max_depth=depth)
                        elif model_type=='LinearRegression':
                            selected_regressor=LinearRegression()

                        metric_type = st.sidebar.selectbox('Select Metric:', ['R2_Score', 'MSE'])
                        st.write("You selected:", metric_type)

                        iterations = st.sidebar.slider('Iterations', 1, 1000, 100)
                        st.write('You selected {} iterations:'.format(iterations))

                        frame_width=st.sidebar.slider('Frame Width:', 200,1000,800)
                        frame_height=st.sidebar.slider('Frame Height:',200,1000,400)


                        if metric_type=='R2_Score':
                            if st.checkbox('Run Feature Selection'):

                                feat_sel=FeatureSelection(data=data,r2_score=True,mse=False, regressor=selected_regressor)
                                output=feat_sel.feature_selection(regModel=True,
                                                             classModel=False,
                                                             boxPlot=True,
                                                             length=frame_width,
                                                             height=frame_height,
                                                             x_fontsize=18,
                                                             y_fontsize=16,
                                                             xticks_size=14,
                                                             yticks_size=14,
                                                             title='Feature Selection For XXX',
                                                             title_fontsize=18,
                                                             iterations=iterations)
                                #if st.checkbox("Display BoxPlot of Feature Importance"):
                                output
                                # st.pyplot()

                        elif metric_type=='MSE':
                            if st.checkbox('Run Feature Selection'):

                                feat_sel = FeatureSelection(data=data, r2_score=False, mse=True, regressor=LinearRegression())
                                output = feat_sel.feature_selection(regModel=True,
                                                                    classModel=False,
                                                                    boxPlot=True,
                                                                    length=frame_width,
                                                                    height=frame_height,
                                                                    x_fontsize=18,
                                                                    y_fontsize=16,
                                                                    xticks_size=14,
                                                                    yticks_size=14,
                                                                    title='Feature Selection For XXX',
                                                                    title_fontsize=18,
                                                                    iterations=iterations)
                                output
                                #st.pyplot()

                    elif problem_type=='Classification':

                        model_type = st.sidebar.selectbox('Select ML Model:',
                                                          ['LogisticRegression', 'RandomForest','SVC'])
                        st.write('You selected:', model_type)

                        if model_type == 'RandomForest':
                            num_trees = st.sidebar.slider('Number of Trees:', 1, 1000, 50)
                            st.write('You selected {} trees'.format(num_trees))

                            depth = st.sidebar.slider('Max Depth:', 1, 100, 10)
                            st.write('Max Depth is:', depth)

                            selected_classifier = RandomForestClassifier(n_estimators=num_trees, max_depth=depth)
                        elif model_type == 'LogisticRegression':

                            selected_penalty=st.sidebar.selectbox('Select Regularization:',['l1','l2'])
                            selected_solver=st.sidebar.selectbox('Select Solver',['liblinear','warn'])
                            selected_C=st.sidebar.slider('C:',0.0,10.0,1.0)

                            selected_classifier = LogisticRegression(penalty=selected_penalty,solver=selected_solver, C=selected_C)

                        elif model_type=='SVC':
                            selected_gamma=st.sidebar.slider('Gamma:',0.00001,0.1,0.001)
                            selected_C=st.sidebar.slider('C:',0.0,10.0,1.0)
                            selected_kernel=st.sidebar.selectbox('Kernel:',['rbf','poly'])


                            if selected_kernel=='rbf':
                                selected_classifier=SVC(C=selected_C,gamma=selected_gamma,kernel=selected_kernel,probability=True)
                            elif selected_kernel=='poly':
                                deg = st.sidebar.slider('Degree:', 1, 10, 3)
                                selected_classifier = SVC(C=selected_C, gamma=selected_gamma, kernel=selected_kernel,degree=deg,probability=True)


                        if st.checkbox("Run Feature Selection"):
                            frame_width = st.sidebar.slider('Frame Width:', 4, 30, 12)
                            frame_height = st.sidebar.slider('Frame Height:', 4, 30, 6)

                            feat_sel = FeatureSelection(data, classifier=selected_classifier)

                            output=feat_sel.feature_selection(classModel=True,regModel=False,
                                                          roc=True,
                                                          boxPlot=False,
                                                          split=False,
                                                          iterations=1,
                                                          length=frame_width,
                                                          height=frame_height,
                                                          title='Feature Importance: ROC Curves After Shuffling',
                                                          title_fontsize=22,
                                                          x_fontsize=16,
                                                          y_fontsize=16)
                            #output
                            st.pyplot()





            elif select_subset=='No':
                st.sidebar.markdown("Preprocessing: Drop & Impute")

                missing_col_perc = st.sidebar.slider("Limit for % of missing column values", 1, 90, 5)
                st.write("All columns missing more than {}% of values will be dropped. The rest will be imputed.".format(
                    missing_col_perc))

                # missing_row_num = st.sidebar.slider('Limit # of missing row values', 1, len(data.index))
                # st.write("All rows missing more than {} values will be dropped. The rest will be imputed".format(
                #     missing_row_num))

                drop = DropImpute(data)
                drop.impute_values(feat_min_perc=missing_col_perc,inplace=True)

                if st.checkbox("Display Data After Drop and Imputation"):
                    st.dataframe(data)




                problem_type=st.sidebar.selectbox('Select the Model Type', ['Regression', 'Classification'])
                st.write('You selected:',problem_type)

                if problem_type=='Regression':

                    metric_type = st.sidebar.selectbox('Select Metric:', ['R2_Score', 'MSE'])
                    st.write("You selected:", metric_type)

                    iterations=st.sidebar.slider('Iterations',1,1000,100)
                    st.write('You selected {} iterations:'.format(iterations))

                    if metric_type=='R2_Score':
                        feat_sel=FeatureSelection(data=data,r2_score=True,mse=False, regressor=LinearRegression())
                        output=feat_sel.feature_selection(regModel=True,
                                                     classModel=False,
                                                     boxPlot=True,
                                                     length=12,
                                                     x_fontsize=18,
                                                     y_fontsize=16,
                                                     xticks_size=14,
                                                     yticks_size=14,
                                                     title='Feature Selection For XXX',
                                                     title_fontsize=18,
                                                     iterations=50)
                        output
                        st.plotly_chart()
                    elif metric_type=='MSE':
                        feat_sel = FeatureSelection(data=data, r2_score=False, mse=True, regressor=LinearRegression())


                elif problem_type == 'Classification':

                    model_type = st.sidebar.selectbox('Select ML Model:',
                                                      ['LogisticRegression', 'RandomForest', 'SVC'])
                    st.write('You selected:', model_type)

                    if model_type == 'RandomForest':
                        num_trees = st.sidebar.slider('Number of Trees:', 1, 1000, 50)
                        st.write('You selected {} trees'.format(num_trees))

                        depth = st.sidebar.slider('Max Depth:', 1, 100, 10)
                        st.write('Max Depth is:', depth)

                        selected_classifier = RandomForestClassifier(n_estimators=num_trees, max_depth=depth)
                    elif model_type == 'LogisticRegression':

                        selected_penalty = st.sidebar.selectbox('Select Regularization:', ['l1', 'l2'])
                        selected_solver = st.sidebar.selectbox('Select Solver', ['liblinear', 'warn'])
                        selected_C = st.sidebar.slider('C:', 0.0, 10.0, 1.0)

                        selected_classifier = LogisticRegression(penalty=selected_penalty, solver=selected_solver,
                                                                 C=selected_C)

                    elif model_type == 'SVC':
                        selected_gamma = st.sidebar.slider('Gamma:', 0.00001, 0.1, 0.001)
                        selected_C = st.sidebar.slider('C:', 0.0, 10.0, 1.0)
                        selected_kernel = st.sidebar.selectbox('Kernel:', ['rbf', 'poly'])

                        if selected_kernel == 'rbf':
                            selected_classifier = SVC(C=selected_C, gamma=selected_gamma, kernel=selected_kernel,
                                                      probability=True)
                        elif selected_kernel == 'poly':
                            deg = st.sidebar.slider('Degree:', 1, 10, 3)
                            selected_classifier = SVC(C=selected_C, gamma=selected_gamma, kernel=selected_kernel,
                                                      degree=deg, probability=True)

                    if st.checkbox("Run Feature Selection"):
                        frame_width = st.sidebar.slider('Frame Width:', 200, 1000, 800)
                        frame_height = st.sidebar.slider('Frame Height:', 100, 100, 400)

                        feat_sel = FeatureSelection(data, classifier=selected_classifier)

                        output = feat_sel.feature_selection(classModel=True, regModel=False,
                                                            roc=True,
                                                            boxPlot=False,
                                                            split=False,
                                                            iterations=1,
                                                            length=frame_width,
                                                            height=frame_height,
                                                            title='Feature Importance: ROC Curves After Shuffling',
                                                            title_fontsize=22,
                                                            x_fontsize=16,
                                                            y_fontsize=16)
                        output
                        st.pyplot()



