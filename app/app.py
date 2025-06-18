import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json


def set_page_config():
    st.set_page_config(
        page_title="Forecasting App",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_data():
    # read the json data file
    path = "/Users/D065623/Documents/Amna/Time_series_project/Forcasting_app/forecasts/stores_forecasts_15_updated.json"
    with open(path, "r") as file:
        data = json.load(file)

    return data


def get_stores(data):
    stores = list(data.keys())
    # remove 'date' from the list
    if "date" in stores:
        stores.remove("date")
    return stores


def main():
    set_page_config()
    # create two columns for a side by side view in the app

    data = read_data()

    # create a sidebar for selection of store_nbr and item_nbr
    with st.sidebar:
        st.title("Select Store and Item for Forecasting")
        st.markdown("---")

        store_nbr = st.selectbox(
            "Select Store Number:", get_stores(data), index=0, key="store_nbr"
        )

        # Get the items for the selected store
        items = list(data[store_nbr]["top_items"])

        item_nbr = st.selectbox("Select Item Number:", items, index=0, key="item_nbr")
        st.markdown("---")

        st.write("*How it works:*")
        st.write(
            "Please select a store and item to forecast the sales for the selected store and item."
        )
        st.write(
            "*Note:* All Stores in Gurayas Region are included in the dataset to provide forecast for top-10 items of selected store."
        )
        st.markdown("---")

    st.markdown("## Corporación Favorita Sales Forecasting")
    st.markdown("---")

    # Create two columns structure
    col1, col2 = st.columns(2)

    with col1:

        # using matplotlib to plot the actual and forecasted data for the selected store and item
        st.markdown(
            "#### Forecasted Sales for Store: **{}** and Item: **{}**".format(
                store_nbr, item_nbr
            )
        )

        # Make data for plotting
        store_nbr_key = str(store_nbr)
        item_nbr_key = str(item_nbr)
        x1 = data[store_nbr_key][item_nbr_key]["auto_arima"]["test_data"]
        x2 = data[store_nbr_key][item_nbr_key]["auto_arima"]["forecast"]
        x3 = data[store_nbr_key][item_nbr_key]["xgboost"]["predicted_sales"]
        date = data["date"]["test_data_dates"]

        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(date, x1, label="Actual Sales", color="blue")
        ax.plot(
            date, x2, label="arima-Forecasted Sales", color="orange", linestyle="-."
        )
        ax.plot(
            date, x3, label="xgboost-Predicted Sales", color="green", linestyle="-."
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title(
            "Forecasted Sales for Store: {} and Item: {}".format(store_nbr, item_nbr)
        )
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid(True)
        # Display the plot in the Streamlit app
        st.pyplot(fig)
        st.markdown("---")

        st.write(
            "The above graph shows the actual sales and the forecasted sales for the selected store and item."
        )

    with col2:
        # create a table to show the performance metrics
        # st.title("Performance Metrics")

        # st.markdown("## ")
        left, center, right = st.columns([1, 3, 1])
        with center:
            st.markdown("#### Performance Metrics:")

            arima_ = data[store_nbr_key][item_nbr_key]["auto_arima"]
            xgboost_ = data[store_nbr_key][item_nbr_key]["xgboost"]

            data_1 = {
                "Metric": ["Bias", "MAD", "rMAD", "R2", "MAE"],
                "Auto-Arima": [
                    arima_["Bias"],
                    arima_["MAD"],
                    arima_["rMAD"],
                    arima_["r2"],
                    arima_["mae"],
                ],
                "XGBoost": [
                    xgboost_["Bias"],
                    xgboost_["MAD"],
                    xgboost_["rMAD"],
                    xgboost_["r2"],
                    xgboost_["mae"],
                ],
            }
            df = pd.DataFrame(data_1)

            # Show as a static table
            # st.table(df)

            # Or as an interactive table (sortable, scrollable)
            st.dataframe(df, height=250, width=500)

            st.markdown("---")


if __name__ == "__main__":
    main()
