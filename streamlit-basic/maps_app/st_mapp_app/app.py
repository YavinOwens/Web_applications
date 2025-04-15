import streamlit as st 
import pandas as pd
import plotly.express as px


st.cache_data()
def load_data(data):
    return pd.read_csv(data)

def main():

    st.title("maps app")
    menu = ["Home", "Country", "City", "Attendees", "About"]

    choice = st.sidebar.selectbox("Menu", menu)
    color = st.sidebar.color_picker("color",value="#9E1FA2")
    if choice == "Home":
        world = load_data("data/worldcities.csv")
        with st.expander("Data View"):
            st.dataframe(world)
        fig = px.scatter_mapbox
        fig = px.scatter_mapbox(world,
                                lat="lat"
                                ,lon="lng"
                                ,hover_name="city"
                                ,hover_data=["country","population"]
                                ,color_discrete_sequence=[color]
                                ,zoom=1
                                ,height=700)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)

    if choice == "Country":
        world = load_data("data/worldcities.csv")
        countries_list = world["country"].unique().tolist()
        
        selected_country = st.sidebar.selectbox("country", countries_list)

        with st.expander("Data View"):
                df = world[world["country"] == selected_country]
                st.dataframe(df)
                fig = px.scatter_mapbox
                fig = px.scatter_mapbox(df,
                                lat="lat"
                                ,lon="lng"
                                ,hover_name="city"
                                ,hover_data=["country","population"]
                                ,color_discrete_sequence=[color]
                                ,zoom=1
                                ,height=700)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)

    if choice == "City":
        world = load_data("data/worldcities.csv")
        countries_list = world["city"].unique().tolist()
        
        selected_city = st.multiselect("city", countries_list, default="Jamaica")

        with st.expander("Data View"):
                df = world[world["city"].isin(selected_city)]
                st.dataframe(df)
                fig = px.scatter_mapbox
                fig = px.scatter_mapbox(df,
                                lat="lat"
                                ,lon="lng"
                                ,hover_name="city"
                                ,hover_data=["country", "city", "population"]
                                ,color_discrete_sequence=[color]
                                ,zoom=1
                                ,height=700)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)

    if choice == "Attendees":
        pass
    if choice == "About":
        pass

if __name__ == "__main__":
    main()