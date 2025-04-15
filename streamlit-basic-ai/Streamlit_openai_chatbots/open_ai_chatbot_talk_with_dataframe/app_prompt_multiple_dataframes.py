import streamlit as st

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


import pandas as pd
from langchain_openai import OpenAI


st.set_page_config( page_title="Pandas Prompt Mulitple Data frames", layout="wide")
st.title("Pandas Dataframe Agent")
st.write(
    """
This agent is a Pandas Dataframe agent that uses OpenAI's GPT-3.5-Turbo model.

"""
)

st.markdown("Similar to the previous app you will be able to prompt 2 dataframes at a time :) ")
df1 = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)

df2 = pd.read_csv("https://raw.githubusercontent.com/toddwschneider/nyc-taxi-data/master/data/central_park_weather.csv")
openai_api_key = Y_key

col1, col2 = st.columns(2)

with col1:
    st.write("Titanic Dataset")
    st.write(df1)

with col2:
    st.write("Central Park Weather Dataset")
    st.write(df2, )

agent = create_pandas_dataframe_agent(
    ChatOpenAI(api_key=openai_api_key,temperature=0, model="gpt-3.5-turbo-0613"),
    [df1, df2],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

user_input_ = st.text_input(label="Please enter your prompt",value="what are the features of both datasets ?")


if user_input_:
    response = agent.invoke(user_input_)  # Ensure the agent has a method `ask` or similar
    st.write(response)
