import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    st.title("Regression Model using Neural Network Approach")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Dataset:")
        st.write(df.head())
        st.write("### Dataset Shape:", df.shape)
        st.write("### Column Names:", df.columns.tolist())
        
        # Handling missing values
        if st.checkbox("Show missing values summary"):
            st.write(df.isnull().sum())
        
        # Convert non-numeric columns to NaN before filling missing values with mean
        df_numeric = df.select_dtypes(include=[np.number])
        df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())
        
        # Selecting features and target
        target_column = "Final Grade"
        feature_columns = ["Study Hours per Week", "Attendance Rate"]
        
        if target_column in df.columns and all(col in df.columns for col in feature_columns):
            X = df[feature_columns]
            y = df[target_column]
            
            st.write("### Selected Features:")
            st.write(X.head())
            st.write("### Target Variable:")
            st.write(y.head())
            
            if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                st.write("### Error: Dataset contains missing values after preprocessing. Please check your data.")
                return
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_choice = st.radio("Select Regression Model", ["XGBoost"])
            
            if model_choice == "XGBoost":
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"### Model Performance (MSE): {mse:.4f}")
            
            st.write("### Predictions vs Actual:")
            result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.write(result_df.head())
        else:
            st.write("### Error: Selected columns not found in dataset. Please check column names.")

if __name__ == "__main__":
    main()
