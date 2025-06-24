import os
import pandas as pd
import gradio as gr
from huggingface_hub import upload_file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

CSV_PATH = "test_all.csv"
HF_REPO_ID = "evilfreelancer/rrr-leaderboard"
HF_LEADERBOARD_FILE = "leaderboard_data.csv"

# Read leaderboard data
def load_leaderboard():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH)

# Upload leaderboard to Hugging Face Hub
def upload_leaderboard_to_hf():
    if not HF_TOKEN:
        return "HF_TOKEN not set in .env"
    if not os.path.exists(CSV_PATH):
        return f"File {CSV_PATH} not found."
    upload_file(
        path_or_fileobj=CSV_PATH,
        path_in_repo=HF_LEADERBOARD_FILE,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Update leaderboard_data.csv"
    )
    return f"Leaderboard uploaded to https://huggingface.co/datasets/{HF_REPO_ID}/blob/main/{HF_LEADERBOARD_FILE}"

# Gradio UI
def leaderboard_app():
    df = load_leaderboard()
    if df.empty:
        return gr.Dataframe(value=[], headers=[], label="No data found in test_all.csv")
    return gr.Dataframe(value=df, label="Model Leaderboard")

def on_upload_click():
    msg = upload_leaderboard_to_hf()
    return gr.Textbox.update(value=msg)

with gr.Blocks() as demo:
    gr.Markdown("# LLM Model Leaderboard")
    leaderboard = leaderboard_app()
    upload_btn = gr.Button("Upload leaderboard to Hugging Face")
    upload_status = gr.Textbox(label="Upload status", interactive=False)
    upload_btn.click(fn=on_upload_click, outputs=upload_status)

if __name__ == "__main__":
    demo.launch() 