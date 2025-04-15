import streamlit as st

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


import pandas as pd
from langchain_openai import OpenAI

st.title("Pandas Dataframe Agent")

st.write(
    """
This agent is a Pandas Dataframe agent that uses OpenAI's GPT-3.5-Turbo model.

"""
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)

openai_api_key = Y_key

st.write(df)
agent = create_pandas_dataframe_agent(
    ChatOpenAI(api_key=openai_api_key,temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

user_input_ = st.text_input(label="Please enter your your prompt",value="What is the average age of the passengers?")


if user_input_:
    response = agent.invoke(user_input_)
    st.write(response)
