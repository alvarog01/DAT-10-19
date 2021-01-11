import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import xgboost as xgb
from matplotlib.pyplot import style
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline
from pdpbox import pdp
import plotly.express as px
   

@st.cache
def load_data():
    df = pd.read_excel(r"C:\Users\alvar\Downloads\beerApp\beerhw4.xlsx")
    return df

df = load_data()

st.title("What makes a strong beer")
st.write(df)


page = st.sidebar.radio('Section', ['Data Explorer', 'Model Explorer'])

@st.cache
def print_page_title(page_val):
    print(f"The value of the page is: {page_val}")

print_page_title(page)

if page == 'Data Explorer':
    st.title("Correlation between beer attributes")
    data0= df.corr()
    corr1= px.imshow(data0)
    st.write(corr1)
        
    
    x_axis = st.sidebar.selectbox('Choose A Value for the X-Axis', df.columns.tolist(), index=1)
    y_axis = st.sidebar.selectbox('Choose A Value for the Y-Axis', df.select_dtypes(include=np.number).columns.tolist(), index=3)
    chart_type = st.sidebar.selectbox(
            'What Type of Chart Would You Like to Create?', ['line', 'bar', 'box'])
    
    st.header(f"Chart For: {x_axis} vs {y_axis}")
    
    if chart_type == 'line':
        grouping = df.groupby(x_axis)[y_axis].mean()
        st.line_chart(grouping)
        
    elif chart_type == 'box':
        chart = px.box(df, x=x_axis, y=y_axis)
        st.plotly_chart(chart)
    elif chart_type == 'bar':
        data = df.groupby(x_axis)[y_axis].mean()
        st.bar_chart(data)
          
        
if page == 'Model Explorer':
    num_rounds      = st.sidebar.number_input('Number of Boosting Rounds',
                                 min_value=100, max_value=5000, step=100)
    
    tree_depth      = st.sidebar.number_input('Tree Depth',
                                 min_value=2, max_value=8, step=1, value=3)
    
    learning_rate   = st.sidebar.number_input('Learning Rate',
                                    min_value=.001, max_value=1.0, step=.05, value=0.1)
    
    validation_size = st.sidebar.number_input('Validation Size',
                                      min_value=.1, max_value=.5, step=.1, value=0.2)
    
    random_state    = st.sidebar.number_input('Random State', value=1985)
    
    pipe = make_pipeline(OneHotEncoder(use_cat_names=True), xgb.XGBRegressor())
        
    pipe[1].set_params(n_estimators=num_rounds, max_depth=tree_depth, learning_rate=learning_rate)
    
    X_train, X_val, y_train, y_val = train_test_split(df.drop('abv', axis=1), df['abv'], test_size=validation_size, random_state=random_state)
    
    pipe.fit(X_train, y_train)
    
    mod_results = pd.DataFrame({
            'Train Size': X_train.shape[0],
            'Validation Size': X_val.shape[0],
            'Boosting Rounds': num_rounds,
            'Tree Depth': tree_depth,
            'Learning Rate': learning_rate,
            'Training Score': pipe.score(X_train, y_train),
            'Validation Score': pipe.score(X_val, y_val)
            }, index=['Values'])
    pred_results = pd.DataFrame()
    pred_results['true']= y_val
    pred_results['predicted']= pipe.predict(X_val)
    
    plotly_chart = px.scatter(pred_results, x='true', y='predicted', trendline = 'ols')
    
    
    st.subheader("Model Results")
    st.table(mod_results)
    

    st.subheader("Real vs Predicted Validation Values")

 
    st.plotly_chart(plotly_chart)
    
    pipe.fit(X_train, y_train)
    

    
    
