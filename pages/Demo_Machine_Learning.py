import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def preprocess_data(df):
    # Drop irrelevant columns if exists
    if 'Student ID' in df.columns:
        df.drop(columns=['Student ID'], inplace=True)
    
    # Fill missing values for numerical columns with mean
    num_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    return df

def train_model(df):
    X = df.drop(columns=['Health Condition'])
    y = df['Health Condition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, report, cm, X.columns

st.title("Health & Nutrition Random Forest")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Raw Data")
    st.write(df.head())
    
    st.write("### Exploratory Data Analysis")
    st.write("#### Missing Values")
    st.write(df.isnull().sum())
    
    st.write("#### Data Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Preprocessing
    df = preprocess_data(df)
    
    # Train model
    model, accuracy, report, cm, feature_names = train_model(df)
    
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("#### Classification Report")
    st.write(pd.DataFrame(report).transpose())
    
    # Confusion Matrix
    st.write("#### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(df['Health Condition']), yticklabels=np.unique(df['Health Condition']))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
    
    # Feature Importance
    st.write("#### Feature Importance")
    feature_importance = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, ax=ax)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in Random Forest")
    st.pyplot(fig)

# Step 1: Load Data
st.title("Predict BMI With Column Weight and Height")
st.title("Logistic Regression Model")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

st.write("### BMI Classification Table")
st.write("""
| BMI Range        | Classification       |
|-----------------|---------------------|
| BMI < 18.5      | Underweight         |
| 18.5 ≤ BMI < 24.9 | Normal weight      |
| BMI ≥ 25        | Overweight/Obese    |
""")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Select Features
    feature_cols = st.multiselect("Select Feature Columns", df.columns, default=["Weight", "Height"])

    if "Weight" in feature_cols and "Height" in feature_cols:
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

        # Classify BMI
        df["BMI_Class"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 24.9, np.inf],
            labels=["Underweight", "Normal weight", "Overweight/Obese"]
        )

        # Drop missing values before training
        df_cleaned = df.dropna(subset=["Weight", "Height", "BMI_Class"])

        X = df_cleaned[feature_cols].copy()
        y = df_cleaned["BMI_Class"]

        # Step 2: Preprocessing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fill missing values in X_train and X_test
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Step 3: Train Model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Step 4: Evaluate Model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write("### Model Performance")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write("#### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Step 5: Hyperparameter Tuning
        st.write("### Hyperparameter Tuning")
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        st.write(f"Best Parameters: {grid_search.best_params_}")

        # Step 6: Deployment (Simple Prediction)
        st.write("### Make a Prediction")
        weight = st.number_input("Enter Weight (kg)", value=55.0)
        height = st.number_input("Enter Height (cm)", value=170.0)

        if st.button("Predict"):
            user_bmi = weight / ((height / 100) ** 2)
            scaled_input = scaler.transform([[weight, height]])
            prediction = model.predict(scaled_input)

            st.write(f"Predicted BMI: {user_bmi:.2f}")
            st.write(f"Predicted Class: {prediction[0]}")

