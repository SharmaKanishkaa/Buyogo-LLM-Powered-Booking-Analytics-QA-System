import streamlit as st
from HotelBookingPipeline import HotelBookingPipeline
from HotelBookingRAG import HotelBookingRAG
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime

# Initialize and cache the analytics and RAG systems
@st.cache_resource
def initialize_systems():
    pipeline = HotelBookingPipeline("C:/Users/Kanishka/Documents/Buyogo/hotel_bookings.csv")
    analytics = pipeline.run_pipeline()  # Run full pipeline for analytics
    rag = HotelBookingRAG(analytics)     # Initialize RAG system with analytics
    return pipeline, rag

# Load the systems
pipeline, rag = initialize_systems()

# App Title
st.title("Hotel Booking Analytics Dashboard")

# Sidebar Navigation Options
sidebar_options = ["Home", "Analytics", "Ask a Question", "Visualizations"]
selected_option = st.sidebar.selectbox("Select Option", sidebar_options)

# Home Page Layout
if selected_option == "Home":
    st.header("Welcome to the Hotel Booking Analytics Dashboard!")
    st.write("""
        This interactive dashboard allows you to explore hotel booking data.
        - **Analytics**: View key metrics and statistics
        - **Ask a Question**: Get natural language answers about the data
        - **Visualizations**: Explore data through charts and graphs
    """)

# Analytics Page Layout
elif selected_option == "Analytics":
    st.header("Booking Analytics Overview")

    # Display overall summary statistics
    st.subheader("Summary Statistics")
    summary = pipeline.analytics['summary_stats']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bookings", f"{summary['total_bookings']:,}")
    with col2:
        st.metric("Cancellation Rate", f"{summary['cancellation_rate']:.1%}")
    with col3:
        st.metric("Avg Lead Time", f"{summary['avg_lead_time']:.1f} days")

    # Display monthly performance metrics
    st.subheader("Monthly Performance")
    monthly = pipeline.analytics['monthly_metrics']
    st.dataframe({
        "Month": list(monthly['monthly_adr'].keys()),
        "Average Daily Rate": [f"${v:.2f}" for v in monthly['monthly_adr'].values()],
        "Total Revenue": [f"${v:,.0f}" for v in monthly['monthly_revenue'].values()]
    })

# Natural Language Question Page
elif selected_option == "Ask a Question":
    st.header("Ask About the Data")
    question = st.text_input("Enter your question about the hotel bookings:")

    if st.button("Get Answer") and question:
        with st.spinner("Analyzing your question..."):
            try:
                response = rag.query(question)  # Get RAG response to the query
                st.subheader("Answer")
                st.write(response['answer'])

                # Optionally show source data for transparency
                if st.checkbox("Show sources"):
                    st.subheader("Supporting Data")
                    for i, source in enumerate(response['sources'], 1):
                        st.write(f"{i}. {source}")
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

# Data Visualizations Page
elif selected_option == "Visualizations":
    st.header("Data Visualizations")

    # Visualization selector dropdown
    viz_option = st.selectbox("Choose a visualization", 
                            ["Monthly ADR", "Yearly ADR Comparison", "ADR: Canceled vs Not", "Top 10 Booking Countries", "Lead Time Distribution", "Total Guests Distribution", "Meal Type Distribution"])

    df = pipeline.analytics['raw_data']
  # Use processed raw data stored from pipeline

    # Maintain proper month order for visual clarity
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    # Visualization 1: Monthly ADR
    if viz_option == "Monthly ADR":
        monthly_adr = df.groupby('arrival_date_month')['adr'].mean().reset_index()
        monthly_adr['arrival_date_month'] = pd.Categorical(monthly_adr['arrival_date_month'], categories=month_order, ordered=True)
        monthly_adr = monthly_adr.sort_values('arrival_date_month')

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=monthly_adr, x='arrival_date_month', y='adr', ax=ax, palette='Set2')
        ax.set_title("Average Daily Rate (ADR) by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("ADR ($)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    # Visualization 2: Yearly ADR Comparison
    elif viz_option == "Yearly ADR Comparison":
        yearly_monthly_adr = df.groupby(['arrival_date_year', 'arrival_date_month'])['adr'].mean().reset_index()
        yearly_monthly_adr['arrival_date_month'] = pd.Categorical(yearly_monthly_adr['arrival_date_month'], categories=month_order, ordered=True)
        yearly_monthly_adr = yearly_monthly_adr.sort_values(['arrival_date_year', 'arrival_date_month'])

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=yearly_monthly_adr, x='arrival_date_month', y='adr', hue='arrival_date_year', palette='Set2', ax=ax)
        ax.set_title("Year ADR Comparison")
        ax.set_xlabel("Month")
        ax.set_ylabel("ADR ($)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    # Visualization 3: Canceled vs Not Canceled ADR Trends
    elif viz_option == "ADR: Canceled vs Not":
        cancellation_adr = df.groupby(['is_canceled', 'arrival_date_month'])['adr'].mean().reset_index()
        cancellation_adr['is_canceled'] = cancellation_adr['is_canceled'].map({0: 'Not Canceled', 1: 'Canceled'})
        cancellation_adr['arrival_date_month'] = pd.Categorical(cancellation_adr['arrival_date_month'], categories=month_order, ordered=True)
        cancellation_adr = cancellation_adr.sort_values('arrival_date_month')

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=cancellation_adr, x='arrival_date_month', y='adr', hue='is_canceled', palette='Set2', ax=ax)
        ax.set_title("ADR Trends: Canceled vs Non-Canceled Bookings")
        ax.set_xlabel("Month")
        ax.set_ylabel("ADR ($)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    # Visualization 4: Top 10 Booking Countries
    elif viz_option == "Top 10 Booking Countries":
        top_countries = df['country'].value_counts().head(10)
        country_map = {'PRT': 'Portugal', 'GBR': 'UK', 'FRA': 'France', 'ESP': 'Spain', 'DEU': 'Germany',
                       'ITA': 'Italy', 'IRL': 'Ireland', 'BEL': 'Belgium', 'BRA': 'Brazil', 'NLD': 'Netherlands'}
        top_countries.index = top_countries.index.map(lambda x: country_map.get(x, x))

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_countries.values, y=top_countries.index, palette='Set2', ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title('Top 10 Booking Countries')
        ax.set_xlabel('Number of Bookings')
        ax.set_ylabel('Country')

        # Annotate values
        for i, value in enumerate(top_countries.values):
            ax.text(value + 100, i, f'{value:,}', va='center', fontsize=10)
        st.pyplot(fig)

    # Visualization 5: Lead Time Distribution
    elif viz_option == "Lead Time Distribution":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['lead_time'], bins=50, kde=True, ax=ax, color='teal')
        ax.axvline(df['lead_time'].median(), color='red', linestyle='--', label=f"Median: {df['lead_time'].median():.0f} days")
        ax.set_title('Lead Time Distribution (Days Before Arrival)')
        ax.set_xlabel('Lead Time (Days)')
        ax.set_ylabel('Booking Count')
        ax.legend()
        st.pyplot(fig)

    # Visualization 6: Total Guests Distribution
    elif viz_option == "Total Guests Distribution":
        dataNoCancel = df[df['is_canceled'] == 0]  # Filter only non-canceled bookings
        dataNoCancel['Total Guests'] = dataNoCancel['adults'] + dataNoCancel['children']
        NumberOfGuests_Daily = dataNoCancel['Total Guests'].groupby(dataNoCancel['arrival_date']).sum()
        NumberOfGuests_Daily = NumberOfGuests_Daily.resample('d').sum().to_frame()

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(NumberOfGuests_Daily['Total Guests'], kde=True, stat="density", bins=30, color='blue', edgecolor='black', ax=ax)
        ax.set_title("Distribution of Total Guests")
        ax.set_xlabel("Total Guests")
        ax.set_ylabel("Density")
        st.pyplot(fig)

    # Visualization 7: Meal Type Distribution
    elif viz_option == "Meal Type Distribution":
        main_meal = df['meal'].value_counts()
        cmap = plt.get_cmap("Set1")
        colors = cmap(np.arange(len(main_meal)) * 1)
        my_circle = plt.Circle((0, 0), 0.7, color='white')  # Donut chart center
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(main_meal, labels=main_meal.index, colors=colors,
                                          wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
                                          autopct='%1.1f%%')
        ax.add_artist(my_circle)
        ax.set_title("Meal Type Distribution")
        st.pyplot(fig)
