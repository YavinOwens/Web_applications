import streamlit as st
import pandas as pd
import os
import plotly.express as px

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

# Load world cities data
def load_data(file_path):
    return pd.read_csv(file_path)

# Initialize the CSV file
initialize_csv()

# Load world cities data
world = load_data("data/worldcities.csv")

# Get unique countries and cities from the world cities data
countries = world['country'].unique()
cities = world['city'].unique()

# Streamlit app
st.title('Attendees Registration')

# Streamlit form
with st.form(key='attendees_form'):
    name = st.text_input('Name')
    country = st.selectbox('Country', countries)
    city = st.selectbox('City', cities)
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

# Merge with attendees data to get count of attendees per country
attendee_counts = df['Country'].value_counts().reset_index()
attendee_counts.columns = ['country', 'attendee_count']
world = world.merge(attendee_counts, left_on='country', right_on='country', how='left')
world['attendee_count'].fillna(0, inplace=True)

# Display world cities data with an expander
with st.expander("Data View"):
    st.dataframe(world)

# Plotting the map
fig = px.scatter_mapbox(world,
                        lat="lat",
                        lon="lng",
                        size="attendee_count",  # Size of the points based on attendee count
                        hover_name="city",
                        hover_data=["country", "population"],
                        color_discrete_sequence=["blue"],
                        zoom=1,
                        height=700)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)
