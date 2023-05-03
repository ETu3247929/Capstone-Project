import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def app():
    # Set page title and icon
    st.set_page_config(page_title='Tips App', page_icon=':money_with_wings:')

    # Load the tips dataset
    tips_df = pd.read_csv('data/tips.csv')
    tips_df["sex"] = tips_df["sex"].apply(lambda sex: 1 if sex == "Male" else 0)
    tips_df['day'] = 0
    tips_df['time']=0
    tips_df["smoker"] = tips_df["smoker"].apply(lambda smoker: 1 if smoker == "Yes" else 0)



    # Add a title and subtitle
    st.title('Tips Prediction App')
    st.markdown('This app predicts the tip amount based on various features.')

    # EDA
    st.subheader('Exploratory Data Analysis')

    # Show the original dataset
    if st.checkbox('Show first few rows:'):
        st.write(tips_df.head())

    # Show the summary statistics
    if st.checkbox('Show summary statistics'):
        st.write(tips_df.describe())

    # Show the correlation heatmap
    if st.checkbox('Show correlation heatmap'):
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(tips_df.corr(), annot=True, cmap='coolwarm', linewidths=1)
        st.pyplot(fig)



    # Show the regression plot
    if st.checkbox('Show regression plot'):
        fig = plt.figure(figsize=(8, 6))
        sns.regplot(x='total_bill', y='tip', data=tips_df)
        st.pyplot(fig)

   # Show the boxplot
    if st.checkbox('Show boxplot'):
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(x='day', y='tip', data=tips_df)
        st.pyplot(fig)
        
   # Show Distribution of total bills
    if st.checkbox("Show Distribution of total bills"):
        fig = plt.figure(figsize=(8,6))
        sns.histplot(data=tips_df, x='total_bill', kde='density')
        plt.xlabel('Total Bill')
        plt.title('Distribution of Total Bill Amounts')
        st.pyplot(fig)
        
    # PDA
    st.subheader('Predictive Data Analysis')


    # Create input features
    regressor = st.selectbox('Select regressor:', ('Linear Regression', 'Random Forest Regressor'))
    total_bill = st.number_input('Enter total bill:', min_value=0.00, max_value=100.00, value=0.00)
    sex = st.selectbox('Select sex:', ('Male', 'Female'))
    smoker = st.selectbox('Select smoker:', ('Yes', 'No'))
    size = st.slider('Select size:', 1, 6)

    # Encode categorical variables
    sex_encoded = 1 if sex == 'Male' else 0
    smoker_encoded = 1 if smoker == 'Yes' else 0

    # Create input dataframe
    input_df = pd.DataFrame({
        'total_bill': [0],
        'sex': [sex_encoded],
        'smoker': [smoker_encoded],
        'size': [size]
    })

    # Train a linear regression model
    X = tips_df.drop(['tip', 'day', 'time'], axis=1)
    y = tips_df['tip']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # train a random forest regressor model
    rfa_model = RandomForestRegressor()
    rfa_model.fit(X_train, y_train)


    if regressor == 'Linear Regression':
        tip = model.predict(input_df)
        # Show the model metrics
        st.subheader('Model metrics')
        st.write(f'R^2 score: {model.score(X_test, y_test):.2f}')
        st.write(f'Feature importances: {model.coef_}')
        st.write(f'MSE: {model.score(X_test, y_test):.2f}')
    else:
        tip = rfa_model.predict(input_df)
        # Show the model metrics
        st.subheader('Model metrics')
        st.write(f'Feature importances: {rfa_model.feature_importances_}')
        st.write(f'MSE: {rfa_model.score(X_test, y_test):.2f}')

    # Show the predicted tip amount in boxed
    st.divider()
    st.success('Predicted tip amount')
    st.success(f'${tip[0]:.2f}')








if __name__ == '__main__':
    app()
