import streamlit as st
import pandas as pd
import os

# Function to check if the CSV file exists, and create it if it doesn't
def initialize_csv(filename='attendees.csv'):
    if not os.path.isfile(filename):
        df = pd.DataFrame(columns=['Name', 'Country', 'City'])
        df.to_csv(filename, index=False)

# Function to append data to the CSV file and remove duplicates and blanks
def append_to_csv(data, filename='attendees.csv'):
    df = pd.read_csv(filename)
    new_data = pd.DataFrame([data])
    df = pd.concat([df, new_data], ignore_index=True)
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    df.dropna(how='all', inplace=True)  # Remove rows that are completely blank
    df.to_csv(filename, index=False)

# Initialize the CSV file
initialize_csv()

# Streamlit app
st.title('Attendees Registration')

# Streamlit form
with st.form(key='attendees_form'):
    name = st.text_input('Name')
    country = st.selectbox('Country', ['USA', 'Canada', 'UK', 'Germany', 'France', 'India', 'China', 'Japan'])
    city = st.text_input('City')
    submit_button = st.form_submit_button(label='Submit')

# Handle form submission
if submit_button:
    if name and country and city:
        data = {'Name': name, 'Country': country, 'City': city}
        append_to_csv(data)
        st.success('Submitted successfully!')
    else:
        st.error('Please fill out all fields.')

# Display the CSV file content
if os.path.isfile('attendees.csv'):
    st.subheader('Current Attendees')
    df = pd.read_csv('attendees.csv')
    st.write(df)
