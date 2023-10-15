#import necessary Libraries
import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import zipfile
import os
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import sklearn
#Load your data, encoder, scaler, and model
train_data= pd.read_csv(r'C:\Users\marth\OneDrive - Azubi Africa\Sprints\LP4\Azubi--LP4-Sprint-Embedding-a-Machine-Learning-model-into-a-GUI\streamlit_project\Train data\train.csv')

#Load model,encoder and scaler
def load_ml_components(directory_path):
    """
    Load the encoder, scaler, and model from joblib files in the specified directory.

    Args:
        directory_path (str): The directory where joblib files are stored.

    Returns:
        tuple: A tuple containing the loaded encoder, scaler, and model.
    """
    encoder_path = os.path.join(directory_path, 'lp3encoder.joblib')
    scaler_path = os.path.join(directory_path, 'lp3scaler.joblib')
    model_path = os.path.join(directory_path, 'lp3model.joblib')

    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    return encoder, scaler, model

# Specify the directory where your joblib files are stored
ml_directory = r'C:\Users\marth\OneDrive - Azubi Africa\Sprints\LP4\Azubi--LP4-Sprint-Embedding-a-Machine-Learning-model-into-a-GUI\streamlit_project\ML directory'

# Load the components
encoder, scaler, model = load_ml_components(ml_directory)

st.title('FAVORITA STORES SALES FORCASTING APP 🛒🥕🥦🛍️📈')

# Preview the first few rows to see the columns
columns_preview = train_data.head(0)
st.write("Preview of CSV File Columns:")
st.write(columns_preview)

# Create an empty list to store user data
user_data_list = []
#Inputs interface
Date = st.date_input("Date")
ID = st.number_input("Transaction ID", min_value=0)
Store_nbr = st.number_input("Store Number", min_value=0, max_value=54)
Onpromotion = st.selectbox("On Promotion", [0, 1])
Daily_Oil_Price = st.number_input("Daily Oil Price")

#Retrieve Store Metadata Based on Store Number
# Check if the user input store number exists in train_data
if Store_nbr in train_data['Store_nbr'].values:
    store_info = train_data[train_data['Store_nbr'] == Store_nbr].iloc[0]

    # Extract metadata fields
    Family = store_info['Family']
    City = store_info['City']
    State = store_info['State']
    Type = store_info['Type']
    Cluster = store_info['Cluster']
    
else:
    # Handle the case where the store number is not found
    st.warning("Store Number not found.")

#displacing the store metadata
#Display Metadata Fields:
st.subheader("Store Metadata:")

if Store_nbr in train_data['Store_nbr'].values:
    st.write(f"Store Family: {Family}")
    st.write(f"City: {City}")
    st.write(f"State: {State}")
    st.write(f"Type: {Type}")
    st.write(f"Cluster: {Cluster}")
    

if st.button("Predict Sales"):
# Append user input to the list
 user_data_list.append([Date, ID,Store_nbr,Family,Onpromotion,Daily_Oil_Price,City,State,Type,Cluster])
user_data = pd.DataFrame(user_data_list, columns=["Date", "ID","Store_nbr","Family","Onpromotion","Daily Oil Price","City","State","Type","Cluster"])

# Display input data in a table format
st.subheader("User Input Data:")
st.dataframe(user_data)

    # Categorize the 'Family' column
category_mapping = {
'AUTOMOTIVE': 'Others',
'BABY CARE': 'Personal Care',
'BEAUTY': 'Personal Care',
'BEVERAGES': 'Beverages',
'BOOKS': 'Others',
'BREAD/BAKERY': 'Food',
'CELEBRATION': 'Food',
'CLEANING': 'Others',
'DAIRY': 'Food',
'DELI': 'Food',
'EGGS': 'Food',
'FROZEN FOODS': 'Food',
'GROCERY I': 'Food',
'GROCERY II': 'Food',
'HARDWARE': 'Others',
'HOME AND KITCHEN I': 'Home and Kitchen',
'HOME AND KITCHEN II': 'Home and Kitchen',
'HOME APPLIANCES': 'Home and Kitchen',
'HOME CARE': 'Home and Kitchen',
'LADIESWEAR': 'Clothing',
'LAWN AND GARDEN': 'Others',
'LINGERIE': 'Clothing',
'LIQUOR,WINE,BEER': 'Beverages',
'MAGAZINES': 'Others',
'MEATS': 'Food',
'PERSONAL CARE': 'Personal Care',
'PET SUPPLIES': 'Others',
'PLAYERS AND ELECTRONICS': 'Others',
'POULTRY': 'Food',
'PREPARED FOODS': 'Food',
'PRODUCE': 'Food',
'SCHOOL AND OFFICE SUPPLIES': 'Others',
'SEAFOOD': 'Food'
}

user_data['Family'] = user_data['Family'].apply(lambda x: next((category for category, items in category_mapping.items() if x in items), 'UNKNOWN'))

    # Get date features
    # Define a fucnction to get date features
def getDatefeatures(df,date):
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype('int32')  # 5 and 6 represent Saturday and Sunday
    return df

user_data = getDatefeatures(user_data, 'Date')

    # Encode categorical columns

#Selecting only numerical columns and setting a named variable 
num_user_data = user_data.select_dtypes(exclude="object")
#selecting categorical columns
cat_user_data = user_data.select_dtypes(include="object")

 # model to learn from Categorical  train data
encoder.fit(cat_user_data)

#transform Cat data
# Transform the categorical data to one-hot encoded format
cat_encoded = encoder.transform(cat_user_data)

tct_user_data = pd.DataFrame(cat_encoded.toarray(), index=cat_user_data.index, columns=encoder.get_feature_names_out(input_features=cat_user_data.columns))

#Combined cat and numeric data
comb_user_data= pd.merge(left=num_user_data,right=tct_user_data, how='outer',left_index =True,right_index=True)

user_data =comb_user_data
# Scale numerical columns    
for column in ['ID','Store_nbr','Onpromotion','Daily Oil Price','Cluster','year','month','day','weekday','is_weekend','Family_Beverages',
'Family_Clothing','Family_Food', 'Family_Home and Kitchen', 'Family_Others', 'Family_Personal Care',
'City_Ambato', 'City_Babahoyo', 'City_Cayambe', 'City_Cuenca',
'City_Daule', 'City_El Carmen', 'City_Esmeraldas', 'City_Guaranda','City_Guayaquil', 'City_Ibarra', 
'City_Latacunga', 'City_Libertad','City_Loja', 'City_Machala', 'City_Manta', 'City_Playas', 'City_Puyo',
'City_Quevedo', 'City_Quito', 'City_Riobamba', 'City_Salinas',
'City_Santo Domingo', 'State_Azuay', 'State_Bolivar','State_Chimborazo', 'State_Cotopaxi', 'State_El Oro',
'State_Esmeraldas', 'State_Guayas', 'State_Imbabura', 'State_Loja','State_Los Rios', 'State_Manabi',
'State_Pastaza', 'State_Pichincha','State_Santa Elena', 'State_Santo Domingo de los Tsachilas',
'State_Tungurahua', 'Type_A', 'Type_B', 'Type_C', 'Type_D', 'Type_E']:
        user_data[column] = scaler.transform(user_data[[column]])

    # Predict sales using the model
predictions = model.predict(user_data)

    # Display the results
st.subheader("Sales Prediction:")
st.write("Predicted Sales Amount:", predictions[0])
