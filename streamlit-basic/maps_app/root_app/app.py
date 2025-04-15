import streamlit as st
import pandas as pd
import os
import plotly.express as px
import json

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAI

openai_api_key = Y_Key
# Function to check if the CSV file exists, and create it if it doesn't

st.set_page_config(layout="wide")
def initialize_csv(filename='attendees.csv'):
    if not os.path.isfile(filename):
        df = pd.DataFrame(columns=['Country', 'Count'])
        df.to_csv(filename, index=False)

# Function to append data to the CSV file and aggregate counts
def append_to_csv(country, filename='attendees.csv'):
    df = pd.read_csv(filename)
    if country in df['Country'].values:
        df.loc[df['Country'] == country, 'Count'] += 1
    else:
        new_data = pd.DataFrame([[country, 1]], columns=['Country', 'Count'])
        df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(filename, index=False)

# Load world cities data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except pd.errors.ParserError as e:
        st.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# Initialize the CSV file
initialize_csv()

# Load world cities data
world = load_data("data/worldcities.csv")

# Check if data is loaded successfully
if world.empty:
    st.stop()

# Get unique countries from the world cities data
countries = world['country'].unique()

# Streamlit app
st.title('Attendees Registration')

# Streamlit form
with st.form(key='attendees_form'):
    country = st.selectbox('Country', countries)
    submit_button = st.form_submit_button(label='Submit')

# Handle form submission
if submit_button:
    if country:
        append_to_csv(country)
        st.success('Submitted successfully!')
    else:
        st.error('Please select a country.')

# Display the CSV file content
if os.path.isfile('attendees.csv'):
    st.subheader('Current Attendees')
    df = pd.read_csv('attendees.csv')
    st.write(df)
else:
    df = pd.DataFrame(columns=['Country', 'Count'])

# Merge with world cities data to get latitude and longitude of countries
world_filtered = world.merge(df, left_on='country', right_on='Country', how='inner').drop_duplicates('country')

# Aggregate coordinates for countries (average coordinates of cities within each country)
country_coords = world_filtered.groupby('country').agg({'lat': 'mean', 'lng': 'mean', 'Count': 'sum'}).reset_index()

# Plotting the map
if not country_coords.empty and 'Count' in country_coords:
    fig = px.scatter_mapbox(country_coords,
                            lat="lat",
                            lon="lng",
                            size="Count",  # Size of the points based on attendee count
                            hover_name="country",
                            hover_data=["Count"],
                            color_discrete_sequence=["blue"],
                            zoom=1.70,
                            height=700)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No attendee data available to plot.")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-4"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True
)

user_input_ = st.text_input(label="Please enter your prompt", value="What is the country with the highest count?")

if user_input_:
    try:
        response = agent.invoke(json.dumps({"prompt": user_input_}))
        st.write(response)
    except ValueError as e:
        st.error(f"An error occurred: {e}")
