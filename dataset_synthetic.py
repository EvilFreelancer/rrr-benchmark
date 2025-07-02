import json
import random
import csv
from collections import defaultdict
from typing import List, Dict, Tuple


class SyntheticDatasetGenerator:
    def __init__(self, input_file: str):
        """
        Initialize generator with existing dataset.
        """
        self.input_file = input_file
        self.original_data = []
        self.all_route_pairs = set()  # Set of (route_id, description) tuples

        self.load_data()
        self.analyze_routes()

    def load_data(self):
        """Load original dataset."""
        print(f"Loading data from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        print(f"Loaded {len(self.original_data)} items")

    def analyze_routes(self):
        """Extract all unique route_id + description pairs."""
        for item in self.original_data:
            for step in item["steps"]:
                route_id = step["id"]
                description = step["sense"]
                self.all_route_pairs.add((route_id, description))

        print(f"Found {len(self.all_route_pairs)} unique (route_id, description) pairs")

    def select_routes_with_unique_descriptions(self, original_routes: List[Dict], correct_route_id: int, num_routes: int) -> List[Dict]:
        """Select routes ensuring all descriptions are unique."""
        
        # Find the correct route from original routes
        correct_route = None
        for route in original_routes:
            if route["id"] == correct_route_id:
                correct_route = {"route_id": route["id"], "description": route["sense"]}
                break
        
        if not correct_route:
            print(f"ERROR: Could not find correct route {correct_route_id} in original routes")
            return []

        # Start with the correct route
        routes = [correct_route]
        used_descriptions = {correct_route["description"]}
        
        # Get candidate routes from the original item (excluding the correct one)
        candidate_routes = []
        for route in original_routes:
            if route["id"] != correct_route_id and route["sense"] not in used_descriptions:
                candidate_routes.append({"route_id": route["id"], "description": route["sense"]})
                used_descriptions.add(route["sense"])

        # If we need more routes, get them from all available route pairs
        if len(candidate_routes) < num_routes - 1:
            # Convert all_route_pairs to list and filter out already used descriptions
            available_pairs = [
                {"route_id": route_id, "description": description} 
                for route_id, description in self.all_route_pairs
                if description not in used_descriptions
            ]
            
            # Add more candidate routes
            random.shuffle(available_pairs)
            needed = num_routes - 1 - len(candidate_routes)
            for pair in available_pairs[:needed]:
                candidate_routes.append(pair)
                used_descriptions.add(pair["description"])

        # Select the needed number of additional routes
        needed = min(num_routes - 1, len(candidate_routes))
        if needed > 0:
            selected_additional = random.sample(candidate_routes, needed)
            routes.extend(selected_additional)

        if len(routes) < num_routes:
            print(f"Warning: Only found {len(routes)} routes with unique descriptions out of {num_routes} requested")

        # Shuffle to randomize positions
        random.shuffle(routes)
        return routes

    def generate_synthetic_dataset(self, num_items: int, route_counts: List[int]) -> Dict:
        """Generate synthetic datasets with different route counts."""
        synthetic_datasets = {}

        for route_count in route_counts:
            print(f"Generating dataset with {route_count} routes...")
            dataset = []

            for i in range(num_items):
                # Select a random item from original dataset
                original_item = random.choice(self.original_data)
                
                # Extract dialogue text and convert to messages
                dialogue_text = original_item["text"]
                messages = self.convert_dialog(dialogue_text)
                
                # Get the correct answer and original routes
                correct_route_id = original_item["rightStepId"]
                original_routes = original_item["steps"]

                # Select routes for this item
                routes = self.select_routes_with_unique_descriptions(original_routes, correct_route_id, route_count)

                # Only add if we have the required number of routes
                if len(routes) == route_count:
                    # Verify answer_id is in routes (safety check)
                    route_ids = [route["route_id"] for route in routes]
                    if correct_route_id not in route_ids:
                        print(f"ERROR: answer_id {correct_route_id} not in route_ids {route_ids}")
                        continue

                    dataset.append({
                        "messages": messages,
                        "routes": routes,
                        "answer_id": correct_route_id
                    })

            synthetic_datasets[route_count] = dataset
            print(f"Generated {len(dataset)} items with {route_count} routes")

        return synthetic_datasets

    def convert_dialog(self, text: str) -> List[Dict]:
        """Convert dialogue text to messages format."""
        import re

        pattern = r"(Робот:|Абонент:)"
        parts = re.split(pattern, text)

        messages = []
        role_map = {"Робот:": "assistant", "Абонент:": "user"}

        current_role = None
        for part in parts:
            part = part.strip()
            if part in role_map:
                current_role = role_map[part]
            elif part and current_role:
                messages.append({"role": current_role, "content": part})
        return messages

    def save_datasets(self, datasets: Dict, output_prefix: str = "synthetic_dataset"):
        """Save generated datasets to files."""
        for route_count, dataset in datasets.items():
            filename = f"{output_prefix}_{route_count}_routes.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(dataset)} items to {filename}")

    def validate_datasets(self, datasets: Dict):
        """Validate that all datasets are correct."""
        print("\n=== VALIDATION ===")
        for route_count, dataset in datasets.items():
            errors = 0
            duplicate_descriptions = 0
            semantic_issues = 0
            
            for i, item in enumerate(dataset):
                answer_id = item["answer_id"]
                routes = item["routes"]
                route_ids = [route["route_id"] for route in routes]
                descriptions = [route["description"] for route in routes]
                
                # Check if answer_id is in routes
                if answer_id not in route_ids:
                    print(f"ERROR in {route_count}-routes dataset, item {i}: "
                          f"answer_id {answer_id} not in routes {route_ids}")
                    errors += 1
                
                # Check for duplicate descriptions
                if len(descriptions) != len(set(descriptions)):
                    print(f"ERROR in {route_count}-routes dataset, item {i}: "
                          f"duplicate descriptions found")
                    duplicate_descriptions += 1
                    
                # Check semantic matching (basic)
                user_message = None
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        user_message = msg["content"]
                        break
                
                correct_description = None
                for route in routes:
                    if route["route_id"] == answer_id:
                        correct_description = route["description"]
                        break
                        
                if user_message and correct_description:
                    # Basic semantic check - can be improved
                    if ("пароль" in user_message.lower() and "пароль" not in correct_description.lower()) or \
                       ("тариф" in user_message.lower() and "тариф" not in correct_description.lower() and "информац" not in correct_description.lower()) or \
                       ("адрес" in user_message.lower() and "адрес" not in correct_description.lower() and "график" not in correct_description.lower()):
                        semantic_issues += 1
            
            print(f"{route_count} routes dataset: {errors} missing answer_id errors, "
                  f"{duplicate_descriptions} duplicate description errors, "
                  f"{semantic_issues} potential semantic issues")

    def save_statistics(self, datasets: Dict, output_file: str = "synthetic_stats.csv"):
        """Save dataset statistics."""
        stats = []
        for route_count, dataset in datasets.items():
            # Count unique correct answers
            unique_answers = set(item["answer_id"] for item in dataset)
            
            # Count unique descriptions across all items
            all_descriptions = set()
            for item in dataset:
                for route in item["routes"]:
                    all_descriptions.add(route["description"])
            
            stats.append({
                "route_count": route_count,
                "dataset_size": len(dataset),
                "unique_correct_answers": len(unique_answers),
                "unique_descriptions_used": len(all_descriptions)
            })

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
        print(f"Statistics saved to {output_file}")


def main():
    # Initialize generator
    generator = SyntheticDatasetGenerator('dataset_input.json')

    # Define route counts to generate
    route_counts = [2, 3, 5, 7, 9]

    # Generate synthetic datasets
    num_items_per_dataset = 1000
    datasets = generator.generate_synthetic_dataset(num_items_per_dataset, route_counts)

    # Validate datasets
    generator.validate_datasets(datasets)

    # Save datasets
    generator.save_datasets(datasets)

    # Save statistics
    generator.save_statistics(datasets)

    print("\n=== GENERATION COMPLETE ===")
    print("Generated synthetic datasets with IMPROVED logic:")
    for route_count in route_counts:
        if route_count in datasets:
            print(f"  - {route_count} routes: {len(datasets[route_count])} items")

    # Print some examples
    print(f"\nExamples from generated datasets:")
    for route_count in [3, 5]:
        if route_count in datasets and datasets[route_count]:
            example = datasets[route_count][0]
            user_msg = next(msg for msg in example['messages'] if msg['role'] == 'user')
            correct_route = next(r for r in example['routes'] if r['route_id'] == example['answer_id'])
            print(f"\n{route_count}-routes example:")
            print(f"  Question: {user_msg['content']}")
            print(f"  Answer ID: {example['answer_id']}")
            print(f"  Correct description: {correct_route['description']}")


if __name__ == "__main__":
    main()
