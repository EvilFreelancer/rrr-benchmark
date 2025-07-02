import re
import streamlit as st
import pandas as pd
import altair as alt

# Load CSV file
DATA_FILE = "test_all.csv"
df = pd.read_csv(DATA_FILE)

# Normalize column names
df.columns = df.columns.str.strip()

# Page header
st.title("🇷🇺 Russian Router Ranking (RRR)")
st.markdown("""
This leaderboard evaluates Large Language Models (LLMs) on their ability to perform **text routing and classification 
tasks in Russian**. Models are assessed based on their capability to return answers in a **structured output** format 
(JSON), which is essential for automation and system integration in real-world applications.

The dataset used is [rrr-benchmark](https://huggingface.co/datasets/evilfreelancer/rrr-benchmark), which focuses on 
practical routing tasks across various domains.

Source code and details: [GitHub Repository](https://github.com/EvilFreelancer/rrr-benchmark)
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
.sortable-header {
    cursor: pointer;
    background-color: #f0f2f6 !important;
    color: #262730 !important;
    padding: 8px 12px !important;
    border: 1px solid #ddd !important;
    user-select: none;
    position: relative;
}
.sortable-header:hover {
    background-color: #e6e9f0 !important;
    color: #262730 !important;
}
.sort-indicator {
    margin-left: 5px;
    font-size: 12px;
    color: #666;
}
.tooltip-icon {
    margin-left: 5px;
    color: #666;
    cursor: help;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# Utility function to numerically sort model sizes (e.g., 7b < 13b < 65b)
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


# Function to create model URL from model field
def get_model_url(model_field, model_name):
    # Create URL with model name embedded for regex extraction
    if model_field.startswith("hf.co/"):
        # Remove tag after colon if present (e.g., hf.co/model:tag -> hf.co/model)
        if ":" in model_field:
            model_field = model_field.split(":")[0]
        base_url = f"https://{model_field}"
        # Add model name as URL fragment for regex extraction
        return f"{base_url}#{model_name}"
    else:
        base_url = f"https://ollama.com/library/{model_field}"
        # Add model name as URL fragment for regex extraction  
        return f"{base_url}#{model_name}"


# Function to render interactive table
def render_interactive_table(data, split_name):
    if data.empty:
        st.info(f"No data available for {split_name} split yet.")
        return

    # Apply sidebar filters
    filtered_df = data.copy()
    if model_name:
        filtered_df = filtered_df[filtered_df["model_name"].isin(model_name)]
    if model_size:
        filtered_df = filtered_df[filtered_df["model_size"].isin(model_size)]
    if model_quant:
        filtered_df = filtered_df[filtered_df["model_quant"].isin(model_quant)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Prepare display dataframe
    display_df = filtered_df.copy()

    # Convert accuracy to percentage (multiply by 100)
    display_df["accuracy"] = display_df["accuracy"] * 100

    # Create numerical size for proper sorting (hidden column)
    display_df["size_numeric"] = display_df["model_size"].apply(model_size_sort_key)

    # Create model URLs with embedded model names
    display_df["Model_URL"] = display_df.apply(lambda row: get_model_url(row["model"], row["model_name"]), axis=1)

    # Clean up and select needed columns
    display_df = display_df[[
        "Model_URL", "model_size", "size_numeric", "model_quant",
        "accuracy", "avg_response_time", "avg_token_count"
    ]].copy()

    # Rename columns
    display_df = display_df.rename(columns={
        "Model_URL":         "Model",
        "model_size":        "Size",  # Use original size format (1b, 7b, 16b)
        "model_quant":       "Quant",
        "accuracy":          "Accuracy",
        "avg_response_time": "Avg Time",
        "avg_token_count":   "Avg Tokens"
    })

    # Sort by accuracy by default (descending)
    display_df = display_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    # Column configuration
    column_config = {
        "Model":        st.column_config.LinkColumn(
            "Model",
            help="Click to open model page",
            width="medium",
            display_text=r".*#(.*)"  # Extract model name after # symbol
        ),
        "Size":         st.column_config.TextColumn(
            "Size",
            help="Model size (parameters count)",
            width="small"
        ),
        "size_numeric": None,  # Hide this column but keep it for sorting
        "Quant":        st.column_config.TextColumn(
            "Quant",
            help="Quantization level",
            width="small"
        ),
        "Accuracy":     st.column_config.NumberColumn(
            "Accuracy (%)",
            help="Accuracy score (higher is better)",
            format="%.2f",
            width="small"
        ),
        "Avg Time":     st.column_config.NumberColumn(
            "Avg Time (s)",
            help="Average response time in seconds (lower is better)",
            format="%.3f",
            width="small"
        ),
        "Avg Tokens":   st.column_config.NumberColumn(
            "Avg Tokens",
            help="Average number of tokens in response",
            format="%.1f",
            width="small"
        )
    }

    # Display the table
    st.data_editor(
        display_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        disabled=True
    )


# Function to render averaged scores table
def render_averaged_table():
    if "dataset_split" not in df.columns:
        st.info("Dataset does not contain 'dataset_split' column.")
        return

    # Filter out generic split for averaging
    non_generic_df = df[df["dataset_split"] != "generic"]

    if non_generic_df.empty:
        st.info("No non-generic data available for averaging.")
        return

    # Apply sidebar filters first
    filtered_df = non_generic_df.copy()
    if model_name:
        filtered_df = filtered_df[filtered_df["model_name"].isin(model_name)]
    if model_size:
        filtered_df = filtered_df[filtered_df["model_size"].isin(model_size)]
    if model_quant:
        filtered_df = filtered_df[filtered_df["model_quant"].isin(model_quant)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Calculate averages grouped by model
    avg_df = (
        filtered_df
        .groupby(["model_name", "model", "model_size", "model_quant"], as_index=False)
        .agg({
            "accuracy":          "mean",
            "avg_response_time": "mean",
            "avg_token_count":   "mean"
        })
    )

    render_interactive_table(avg_df, "Average Scores")

    # Add accuracy chart by model and split
    st.markdown("### 📊 Accuracy by Model and Number of Routes")
    st.markdown("*Shows accuracy performance across different number of routes*")

    # Prepare data for chart - group by model_name AND model_size for unique variations
    chart_data = (
        filtered_df
        .groupby(["model_name", "model_size", "dataset_split"], as_index=False)
        .agg({"accuracy": "mean"})
    )

    # Create unique model identifier combining name and size
    chart_data["model_variant"] = chart_data["model_name"] + " (" + chart_data["model_size"] + ")"

    # Convert accuracy to percentage for display
    chart_data["accuracy"] = chart_data["accuracy"] * 100

    # Ensure accuracy is within 0-100 range
    chart_data["accuracy"] = chart_data["accuracy"].clip(0, 100)

    if not chart_data.empty:
        # Create pivot table for chart using model_variant as columns
        pivot_data = chart_data.pivot(index="dataset_split", columns="model_variant", values="accuracy")

        # Reorder index to show logical progression of route complexity
        route_order = ["routes_3", "routes_5", "routes_7", "routes_9"]
        pivot_data = pivot_data.reindex([split for split in route_order if split in pivot_data.index])

        # Rename index to be more readable (X-axis labels)
        index_rename = {
            "routes_3": "3",
            "routes_5": "5",
            "routes_7": "7",
            "routes_9": "9"
        }
        pivot_data = pivot_data.rename(index=index_rename)

        # Display line chart with fixed Y-axis
        # Prepare data for Altair
        chart_df = pivot_data.reset_index().melt(id_vars="dataset_split", var_name="model_variant",
                                                 value_name="accuracy")

        # Create Altair line chart with fixed Y-axis
        chart = alt.Chart(chart_df).mark_line(point=True).add_selection(
            alt.selection_multi(fields=['model_variant'])
        ).encode(
            x=alt.X('dataset_split:O', title='Number of Routes', sort=['3', '5', '7', '9']),
            y=alt.Y('accuracy:Q', title='Accuracy (%)', scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('model_variant:N', title='Model (Size)'),
            tooltip=['dataset_split:O', 'model_variant:N', 'accuracy:Q']
        ).properties(
            height=400,
            title="Accuracy Performance Across Route Complexity"
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data available for chart display.")


# Dataset splits configuration
splits_config = {
    "average":  {
        "name":        "Average Scores",
        "description": "Average metrics for each model across all route datasets (excluding Generic)"
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
    },
    "generic":  {
        "name":        "Generic",
        "description": "Original dataset with variable number of routes per item (2-9 routes)"
    }
}

# Build tab names
tab_names = [splits_config[split]["name"] for split in splits_config.keys()]
tabs = st.tabs(tab_names)

# Render each dataset split
for i, (split_key, split_config) in enumerate(splits_config.items()):
    with tabs[i]:
        st.markdown(f"**{split_config['description']}**")
        st.markdown("*Click on column headers to sort the table*")

        if split_key == "average":
            render_averaged_table()
        else:
            split_data = df[df["dataset_split"] == split_key] if "dataset_split" in df.columns else pd.DataFrame()
            render_interactive_table(split_data, split_config["name"])
