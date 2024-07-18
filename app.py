import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open('models/model.pkl', 'rb'))




# Function to predict sales
def predict_sales(features):
    # Convert input into dataframe
    input_df = pd.DataFrame([features])

    # Apply label encoding to categorical features
    label_encoder = LabelEncoder()
    input_df['Item_Fat_Content'] = label_encoder.fit_transform(input_df['Item_Fat_Content'])
    input_df['category'] = input_df['category'].astype('category')


    # Make predictions
    prediction = model.predict(input_df)

    return prediction[0]

    return prediction[0]
X = ["Dairy", "Fruits", "Vegetables"]
y = [10, 20, 30]


import numpy as np

X = np.array([0, 1, 2])
y = np.array([1, 3, 7]).reshape(-1, 1)

le = LinearRegression()
X = X.reshape(-1, 1)
y = y.ravel()
model.fit(X, y)


# assume you have a new input with value "Dairy"
new_input = ["Dairy"]


le = LabelEncoder()

# Fit and transform the data using the LabelEncoder
new_input_encoded = le.fit_transform(np.array(new_input).reshape(-1, 1))

# and then convert it into a 2D array
new_input_df = pd.DataFrame([new_input_encoded], columns=["category"])


# predict the sales for the new input
prediction = model.predict(new_input_df)



# Create a web app
def main():
    st.title('Big Mart Sales Prediction')

    # Get input features from user
    item_weight = st.number_input('Item Weight')
    item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular', 'Non-Edible'])
    item_visibility = st.number_input('Item Visibility')
    item_type = st.selectbox('Item Type', ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
                                           'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
                                           'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
                                           'Breads', 'Starchy Foods', 'Others'])
    item_mrp = st.number_input('Item MRP')
    outlet_establishment_year = st.number_input('Outlet Establishment Year')
    outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
    outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2',
                                                'Supermarket Type3', 'Grocery Store'])

    # Map input features to a dictionary
    # Map input features to a dictionary
    input_dict = {'Item_Weight': item_weight,
              'Item_Fat_Content': item_fat_content,
              'Item_Visibility': item_visibility,
              'Item_Type': item_type,
              'Item_MRP': item_mrp,
              'Outlet_Establishment_Year': outlet_establishment_year,
              'Outlet_Size': outlet_size,
              'Outlet_Location_Type': outlet_location_type,
              'Outlet_Type': outlet_type}

# Apply label encoding to categorical features
    label_encoder = LabelEncoder()
    input_dict['Item_Fat_Content'] = label_encoder.fit_transform([input_dict['Item_Fat_Content']])[0]
    def predict_sales(input_dict):
        input_df = pd.DataFrame(input_dict, index=[0])
        for column in input_df.columns:
            if input_df[column].dtype == 'object':
                input_df[column] = input_df[column].replace(' ', np.nan)
                input_df[column] = input_df[column].fillna('Unknown')
            elif input_df[column].dtype == 'int64':
                input_df[column] = input_df[column].astype('float64')
        prediction = model.predict(input_df)
        return prediction
    # Pass the cleaned input_df to the model for prediction
    
    
    


# Predict sales using the trained model
    if st.button('Predict Sales'):
       sales = predict_sales(input_dict)
       st.success(f'Predicted Sales: {sales:.2f}')


if __name__ == '__main__':
    main()
