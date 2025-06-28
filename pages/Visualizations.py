import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="📊 Sales Visualizations", layout="wide")
st.title("📊 Super Store Sales Dashboard")
st.markdown("Visual insights from **Vrinda Store** data")

# --- 1. Cached Data Loading ---
@st.cache_data
def load_data():
    file_path = "data/SUPER_STORE_raw.xlsx"
    if not os.path.exists(file_path):
        return None
    df = pd.read_excel(file_path, sheet_name="Vrinda Store")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["day"] = df["Date"].dt.day
    df["weekday"] = df["Date"].dt.day_name()
    return df

df = load_data()

if df is None:
    st.error("❌ Data file not found. Please ensure 'SUPER_STORE_raw.xlsx' is inside the 'data/' folder.")
    st.stop()

# Layout: 2 columns
col1, col2 = st.columns(2)

# --- 2. Monthly Sales Trend ---
@st.cache_data
def get_monthly_sales(df):
    return df.groupby("Month")["Amount"].sum().sort_index()

with col1:
    st.subheader("📅 Monthly Sales Trend")
    st.line_chart(get_monthly_sales(df))

# --- 3. Sales by Category ---
@st.cache_data
def get_category_sales(df):
    return df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

with col2:
    st.subheader("🛒 Sales by Category")
    st.bar_chart(get_category_sales(df))

# --- 4. Gender Pie Chart ---
@st.cache_resource
def gender_pie_chart(df):
    gender_count = df["Gender"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(gender_count, labels=gender_count.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    return fig

st.subheader("🚻 Orders by Gender")
st.pyplot(gender_pie_chart(df))

# --- 5. Age Group Distribution ---
@st.cache_data
def get_age_distribution(df):
    return df["Age Group"].value_counts()

st.subheader("👶 Age Group Distribution")
st.bar_chart(get_age_distribution(df))

# --- 6. Sales by Channel ---
@st.cache_resource
def channel_bar_chart(df):
    channel_sales = df.groupby("Channel")["Amount"].sum()
    fig, ax = plt.subplots()
    channel_sales.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Sales Amount")
    return fig

st.subheader("📦 Sales by Channel")
st.pyplot(channel_bar_chart(df))

# --- 7. Heatmap of Daily Sales ---
@st.cache_resource
def daily_sales_heatmap(df):
    pivot_table = df.pivot_table(index="weekday", columns="day", values="Amount", aggfunc="sum")
    weekdays_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_table = pivot_table.reindex(weekdays_order)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot_table, cmap="YlOrBr", ax=ax)
    return fig

st.subheader("🔥 Heatmap of Daily Sales")
st.pyplot(daily_sales_heatmap(df))

# --- 8. Top Cities by Sales ---
@st.cache_data
def get_top_cities(df):
    return df.groupby("ship-city")["Amount"].sum().sort_values(ascending=False).head(10)

st.subheader("🏙️ Top 10 Cities by Sales")
st.bar_chart(get_top_cities(df))
