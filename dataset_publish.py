import os
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please add it to your .env file.")

REPO_ID = "evilfreelancer/rrr-benchmark"


def load_json_dataset(file_path):
    """Load dataset from JSON file."""
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: {file_path} not found")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {file_path}")
    return data


def upload_readme_to_hub(repo_id, readme_content, token):
    """Upload README.md to Hugging Face Hub using HfApi."""
    try:
        # Save README temporarily
        with open("README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Initialize HF API
        api = HfApi()
        
        # Upload README file
        print("📝 Uploading README.md to Hub...")
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Update dataset README"
        )
        print("✅ README.md uploaded successfully!")
        
        # Clean up local README
        if os.path.exists("README.md"):
            os.remove("README.md")
            
    except Exception as e:
        print(f"❌ Failed to upload README: {e}")
        # Clean up on error
        if os.path.exists("README.md"):
            os.remove("README.md")


def main():
    print("🚀 Starting dataset upload to Hugging Face Hub with splits...")

    # Define dataset splits and their files
    splits_config = {
        "generic":  {
            "file":        "dataset_output.json",
            "description": "Original processed dataset from dataset_input.json with variable routes per item"
        },
        "routes_3": {
            "file":        "synthetic_dataset_3_routes.json",
            "description": "Synthetic dataset with exactly 3 route options per item (simple complexity)"
        },
        "routes_5": {
            "file":        "synthetic_dataset_5_routes.json",
            "description": "Synthetic dataset with exactly 5 route options per item (medium complexity)"
        },
        "routes_7": {
            "file":        "synthetic_dataset_7_routes.json",
            "description": "Synthetic dataset with exactly 7 route options per item (high complexity)"
        },
        "routes_9": {
            "file":        "synthetic_dataset_9_routes.json",
            "description": "Synthetic dataset with exactly 9 route options per item (maximum complexity)"
        }
    }

    # Load all datasets and create DatasetDict
    dataset_dict = {}

    for split_name, config in splits_config.items():
        print(f"\n📦 Loading split: {split_name}")
        data = load_json_dataset(config["file"])

        if data is not None:
            # Convert to Hugging Face Dataset
            dataset = Dataset.from_list(data)
            dataset_dict[split_name] = dataset
            print(f"✅ Added split '{split_name}' with {len(data)} items")
        else:
            print(f"❌ Skipping split '{split_name}' - file not found")

    if not dataset_dict:
        print("❌ No datasets loaded. Exiting.")
        return

    # Create DatasetDict
    dataset_collection = DatasetDict(dataset_dict)

    print(f"\n📊 Dataset summary:")
    for split_name, dataset in dataset_collection.items():
        print(f"  - {split_name}: {len(dataset)} items")

    # Create main README with YAML frontmatter
    readme_content = f"""---
dataset_info:
  config_name: default
  splits:"""

    # Add splits info to YAML
    for split_name in dataset_dict.keys():
        item_count = len(dataset_dict[split_name])
        readme_content += f"""
  - name: {split_name}
    num_examples: {item_count}"""

    readme_content += f"""
task_categories:
- text-classification
- question-answering
language:
- ru
tags:
- dialogue
- routing
- benchmark
- russian
- customer-service
pretty_name: RRR Benchmark Datasets
size_categories:
- n<1K
license: mit
---

# RRR Benchmark Datasets

Russian Router Ranking (RRR) benchmark datasets for testing dialogue routing models.

## Dataset Splits

This dataset contains {len(dataset_dict)} splits organized by complexity level:

"""

    for split_name, config in splits_config.items():
        if split_name in dataset_dict:
            item_count = len(dataset_dict[split_name])
            readme_content += f"### `{split_name}` ({item_count} items)\n{config['description']}\n\n"

    readme_content += """## Usage

```python
from datasets import load_dataset

# Load specific split
dataset = load_dataset("evilfreelancer/rrr-benchmark", split="routes_5")

# Load all splits
dataset_dict = load_dataset("evilfreelancer/rrr-benchmark")

# Access specific split
generic_data = dataset_dict["generic"]
routes_3_data = dataset_dict["routes_3"]
```

## Data Format

Each dataset item contains:
- **`messages`**: List of dialogue messages with role ("assistant"/"user") and content
- **`routes`**: List of available routing options with route_id and description
- **`answer_id`**: Correct route ID for the given dialogue context

### Example:

```json
{
  "messages": [
    {"role": "assistant", "content": "Здравствуйте! Как дела?"},
    {"role": "user", "content": "Где ваш офис?"}
  ],
  "routes": [
    {"route_id": 2198, "description": "Информация об адресе организации"},
    {"route_id": 3519, "description": "Прекращение диалога в виду неадекватности абонента"},
    {"route_id": 9821, "description": "Прощание с абонентом после успешного диалога"}
  ],
  "answer_id": 2198
}
```

## Benchmark Goal

Test model ability to select the correct route based on dialogue context and available options.

## Generation Details

- **Original data** (`generic`): Processed from customer service dialogues
- **Synthetic data** (`routes_*`): Generated with unique route descriptions, no duplicates
- **Quality assurance**: All synthetic dialogues maintain natural conversation flow
- **Route selection**: Smart selection from different semantic categories for better evaluation

## License

This dataset is available under the MIT license.
"""

    # Upload dataset to Hugging Face Hub
    print(f"\n🚀 Uploading dataset to {REPO_ID}...")
    try:
        dataset_collection.push_to_hub(
            REPO_ID,
            token=HF_TOKEN,
            commit_message="Update RRR benchmark datasets with splits"
        )
        print(f"✅ Successfully uploaded dataset with {len(dataset_dict)} splits!")
        
        # Upload README separately to ensure it's updated
        upload_readme_to_hub(REPO_ID, readme_content, HF_TOKEN)
        
        print(f"🌐 View at: https://huggingface.co/datasets/{REPO_ID}")

        # Print usage examples
        print(f"\n📖 Usage examples:")
        print(f"```python")
        print(f"from datasets import load_dataset")
        print(f"")
        print(f"# Load specific split")
        print(f'dataset = load_dataset("{REPO_ID}", split="routes_5")')
        print(f"")
        print(f"# Load all splits")
        print(f'dataset_dict = load_dataset("{REPO_ID}")')
        print(f"```")

    except Exception as e:
        print(f"❌ Failed to upload dataset: {e}")


if __name__ == "__main__":
    main()
