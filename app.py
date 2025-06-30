import streamlit as st
import pandas as pd

# Load CSV file
DATA_FILE = "test_all.csv"
df = pd.read_csv(DATA_FILE)

# Normalize column names
df.columns = df.columns.str.strip()

# Header interface
st.title("Russian Router Ranking (RRR) Leaderboard")
st.markdown("""
[GitHub Repository](https://github.com/EvilFreelancer/rrr-benchmark)  
The table shows the accuracy and performance of the models on the 
[rrr-benchmark](https://huggingface.co/datasets/evilfreelancer/rrr-benchmark) dataset.
""")

# Sidebar filtering
with st.sidebar:
    st.header("Filters")
    model_name = st.multiselect("Select model:", options=sorted(df["model_name"].dropna().unique()))
    model_size = st.multiselect("Select size:", options=sorted(df["model_size"].dropna().unique()))
    model_quant = st.multiselect("Select quantization:", options=sorted(df["model_quant"].dropna().unique()))

# Apply filters
filtered_df = df.copy()
if model_name:
    filtered_df = filtered_df[filtered_df["model_name"].isin(model_name)]
if model_size:
    filtered_df = filtered_df[filtered_df["model_size"].isin(model_size)]
if model_quant:
    filtered_df = filtered_df[filtered_df["model_quant"].isin(model_quant)]

# Column formatting for display
format_dict = {
    "accuracy":          "{:.2%}".format,
    "avg_response_time": "{:.3f}".format,
    "avg_token_count":   "{:.1f}".format
}

# Display the table sorted by accuracy in descending order
st.dataframe(
    filtered_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True).style.format(format_dict)
)
