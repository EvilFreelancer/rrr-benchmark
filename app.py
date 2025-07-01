import re
import streamlit as st
import pandas as pd

# Load CSV file
DATA_FILE = "test_all.csv"
df = pd.read_csv(DATA_FILE)

# Normalize column names
df.columns = df.columns.str.strip()

# Page header
st.title("Russian Router Ranking (RRR) Leaderboard")
st.markdown("""
[GitHub Repository](https://github.com/EvilFreelancer/rrr-benchmark)  
The table shows the accuracy and performance of the models on the 
[rrr-benchmark](https://huggingface.co/datasets/evilfreelancer/rrr-benchmark) dataset.
""")
st.markdown("""
<style>
.scrollable-table {
    max-height: 600px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #ddd;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)


# Function to sort model sizes numerically (e.g., 7b < 13b < 32b, etc.)
def model_size_sort_key(size: str):
    if not isinstance(size, str):
        return float('inf')
    match = re.match(r"(\d+(?:\.\d+)?)([mb])", size.lower())
    if not match:
        return float('inf')
    num, unit = match.groups()
    multiplier = 1e6 if unit == 'm' else 1e9
    return float(num) * multiplier


# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Model name filter (case-insensitive sort)
    model_name_options = sorted(df["model_name"].dropna().unique(), key=str.lower)
    model_name = st.multiselect("Select model:", options=model_name_options)

    # Model size filter (numerical sort)
    model_size_options = sorted(df["model_size"].dropna().unique(), key=model_size_sort_key)
    model_size = st.multiselect("Select size:", options=model_size_options)

    # Quantization level filter (default alphabetical sort)
    model_quant = st.multiselect("Select quantization:", options=sorted(df["model_quant"].dropna().unique()))


# Function to render model_name as a clickable link with a tooltip (title)
def make_clickable_label(row):
    model_field = row["model"]
    name = row["model_name"]
    if model_field.startswith("hf.co/"):
        url = f"https://{model_field}"
    else:
        url = f"https://ollama.com/library/{model_field}"
    return f'<a href="{url}" title="{model_field}" target="_blank">{name}</a>'


def render_split_table(split_data, split_name):
    """Render table for a specific dataset split."""
    if split_data.empty:
        st.info(
            f"No data available for {split_name} split yet. Results will appear here after running benchmarks on this "
            f"split.")
        return

    # Apply filters to the split data
    filtered_df = split_data.copy()
    if model_name:
        filtered_df = filtered_df[filtered_df["model_name"].isin(model_name)]
    if model_size:
        filtered_df = filtered_df[filtered_df["model_size"].isin(model_size)]
    if model_quant:
        filtered_df = filtered_df[filtered_df["model_quant"].isin(model_quant)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Format specification for numerical columns
    format_dict = {
        "accuracy":          "{:.2%}".format,
        "avg_response_time": "{:.3f}".format,
        "avg_token_count":   "{:.1f}".format
    }

    # Create new column with HTML links for model_name
    display_df = filtered_df.copy()
    display_df["model_name"] = display_df.apply(make_clickable_label, axis=1)

    # Drop 'model' column from display (but keep it for link rendering)
    display_df = (
        display_df
        .drop(columns=["model"], errors="ignore")
        .drop(columns=["dataset_name"], errors="ignore")
        .drop(columns=["dataset_split"], errors="ignore")
        .drop(columns=["total_tests"], errors="ignore")
        .drop(columns=["valid_responses"], errors="ignore")
        .drop(columns=["correct_responses"], errors="ignore")
    )

    # Apply sorting, formatting, and styling
    styled = (
        display_df.sort_values(by="accuracy", ascending=False)
        .reset_index(drop=True)
        .style
        .format(format_dict)
        .set_sticky(axis="index")  # Keep first column visible on scroll
        .hide(axis="index")  # Hide row index
        .set_properties(subset=["model_name"], **{"text-align": "left"})  # Align left
    )

    st.markdown(
        f'<div class="scrollable-table">{styled.to_html(escape=False)}</div>',
        unsafe_allow_html=True
    )


# Define dataset splits and their descriptions
splits_config = {
    "generic":  {
        "name":        "Generic",
        "description": "Original dataset with variable number of routes per item (2-9 routes)"
    },
    "routes_3": {
        "name":        "3 Routes",
        "description": "Synthetic dataset with exactly 3 route options per item (simple complexity)"
    },
    "routes_5": {
        "name":        "5 Routes",
        "description": "Synthetic dataset with exactly 5 route options per item (medium complexity)"
    },
    "routes_7": {
        "name":        "7 Routes",
        "description": "Synthetic dataset with exactly 7 route options per item (high complexity)"
    },
    "routes_9": {
        "name":        "9 Routes",
        "description": "Synthetic dataset with exactly 9 route options per item (maximum complexity)"
    }
}

# Create tabs for different dataset splits
tab_names = [splits_config[split]["name"] for split in splits_config.keys()]
tabs = st.tabs(tab_names)

# Render each tab
for i, (split_key, split_config) in enumerate(splits_config.items()):
    with tabs[i]:
        st.markdown(f"**{split_config['description']}**")

        # Filter data for this specific split
        split_data = df[df["dataset_split"] == split_key] if "dataset_split" in df.columns else pd.DataFrame()

        render_split_table(split_data, split_config["name"])
