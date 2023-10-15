import gradio as gr
import joblib
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Encoder, Scaler, Model
def load_ml_components(directory_path):
    encoder_path = os.path.join(directory_path, 'lp2encoder.joblib')
    scaler_path = os.path.join(directory_path, 'lp2scaler.joblib')
    model_path = os.path.join(directory_path, 'lp2model.joblib')

    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    return encoder, scaler, model

# Specify the directory where your joblib files are stored
ml_directory = r'C:\Users\marth\OneDrive - Azubi Africa\Sprints\LP4\Azubi--LP4-Sprint-Embedding-a-Machine-Learning-model-into-a-GUI\gradio_project\ML directory'

# Load the components
encoder, scaler, model = load_ml_components(ml_directory)

# Create a dictionary with default values for all input features
default_inputs = {
    'CustomerID': ["YourValue"],
    'Gender': ["Male", "Female"],
    'SeniorCitizen': ["Yes", "No"],
    'Partner': ["Yes", "No"],
    'Dependents': ["Yes", "No"],
    'Tenure': [10],  # Set your default value
    'PhoneService': ["Yes", "No"],
    'MultipleLines': ["Unknown", "No", "Yes"],
    'InternetService': ["DSL", "Fiber optic", "No"],
    'OnlineSecurity': ["No", "Yes", "Unknown"],
    'OnlineBackup': ["No", "Yes", "Unknown"],
    'DeviceProtection': ["No", "Yes", "Unknown"],
    'TechSupport': ["No", "Yes", "Unknown"],
    'StreamingTV': ["No", "Yes", "Unknown"],
    'StreamingMovies': ["No", "Yes", "Unknown"],
    'Contract': ["Month-to-month", "One year", "Two year"],
    'PaperlessBilling': ["Yes", "No"],
    'PaymentMethod': ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    'MonthlyCharges': [50.0],  # Set your default value
    'TotalCharges': [500.0]  # Set your default value
}

# Function to fill in missing inputs with default values
def fill_missing_inputs(inputs):
    for feature, values in default_inputs.items():
        if feature not in inputs:
            inputs[feature] = values
    return inputs


# Define customer data
def predict_customer_churn(
    CustomerID, Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService,
    MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
    PaymentMethod, MonthlyCharges, TotalCharges
):
    # Create a dictionary with the provided inputs
    inputs = {
        'CustomerID': [CustomerID],
        'Gender': [Gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'Tenure': [Tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }
    # Fill in missing inputs with default values
    inputs = fill_missing_inputs(inputs)

    # Create a DataFrame from the input data
    user_data = pd.DataFrame(inputs)

    # Create a new feature 'ChargesperYear'
    user_data['ChargesperYear'] = user_data['MonthlyCharges'] * 12

    # Reorder the columns
    columns = ['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'Tenure', 'PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
               'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'ChargesperYear']

    user_data = user_data.reindex(columns=columns)

    # Encoding Categorical columns
    # Selecting only numerical columns and setting a named variable
    num_data = user_data.select_dtypes(exclude="object")
    # Select columns as objects
    cat_user_data = user_data.select_dtypes(include="object")
    # Let's drop the 'CustomerID' column from the trainset
    ct_user_data = cat_user_data.drop(columns=['CustomerID'])

    # Fit encoder
    encoder.fit(ct_user_data)
    
    # Transform the data
    tct_user_data = pd.DataFrame(encoder.transform(ct_user_data).toarray(), index=ct_user_data.index, columns=encoder.get_feature_names_out(input_features=ct_user_data.columns))

    # Combining Categorical and number transformed user data
    comb_user_data= pd.merge(left=num_data,right=tct_user_data, how='outer',left_index =True,right_index=True)
    
    # Fill the dataframe with default values for unselected features
    for feature, values in default_inputs.items():
        if feature not in comb_user_data.columns:
            comb_user_data[feature] = [values[0]] * len(user_data)

    # Scaling user data
    # Apply the scaler to the datasets
    final_user_data = pd.DataFrame(scaler.transform(comb_user_data), columns=comb_user_data.columns,
                                   index=comb_user_data.index)
    #Readd Customer ID dropped in the dataframe
    final_user_data["CustomerID"] = ct_user_data["CustomerID"]

    prediction = model.predict(final_user_data)

    return "Customer will churn." if prediction[0] == 1 else "Customer will not churn."

# Define Gradio interface
iface = gr.Interface(
    fn=predict_customer_churn,
    inputs=[
    # Subheader for "Customer Details"
    gr.Label("Customer Details"),
    gr.Textbox(label="CustomerID"),
    gr.Radio(label="Gender", choices=["Male", "Female"]),
    gr.Radio(label="SeniorCitizens", choices=["Yes", "No"]),
    gr.Radio(label="Partner", choices=["Yes", "No"]),
    gr.Radio(label="Dependents", choices=["Yes", "No"]),
    gr.Number(label="Tenure"),
    gr.Radio(label="PhoneService", choices=["Yes", "No"]),
    gr.Dropdown(
        label="MultipleLines",
        choices=["Unknown", "No", "Yes"]
    ),

    # Subheader for "Vodafone Product Data"
    gr.Label("Vodafone Product Data"),
    gr.Dropdown(label="InternetService", choices=["DSL", "Fiber optic", "No"]),
    gr.Dropdown(label="OnlineSecurity", choices=["No", "Yes", "Unknown"]),
    gr.Dropdown(label="OnlineBackup", choices=["No", "Yes", "Unknown"]),
    gr.Dropdown(label="DeviceProtection", choices=["No", "Yes", "Unknown"]),
    gr.Dropdown(label="TechSupport", choices=["No", "Yes", "Unknown"]),
    gr.Dropdown(label="StreamingTV", choices=["No", "Yes", "Unknown"]),
    gr.Dropdown(label="StreamingMovies", choices=["No", "Yes", "Unknown"]),

    # Subheader for "Payment/Billing Plans"
    gr.Label("Payment/Billing Plans"),
    gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"]),
    gr.Radio(label="PaperlessBilling", choices=["Yes", "No"]),
    gr.Dropdown(label="PaymentMethod",
                choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    gr.Number(label="MonthlyCharges"),
    gr.Number(label="TotalCharges"),
   ],

outputs=[
        gr.Textbox("Prediction")
    ],
title="Vodafone Customer Attrition Prediction"
)

# Launch the interface
iface.launch()
