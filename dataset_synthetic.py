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
        self.unique_routes = {}  # route_id -> description
        self.dialogue_templates = []
        self.route_categories = defaultdict(list)

        self.load_data()
        self.analyze_routes()
        self.extract_dialogue_templates()

    def load_data(self):
        """Load original dataset."""
        print(f"Loading data from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        print(f"Loaded {len(self.original_data)} items")

    def analyze_routes(self):
        """Extract all unique routes and categorize them."""
        route_descriptions = {}

        for item in self.original_data:
            for step in item["steps"]:
                route_id = step["id"]
                description = step["sense"]

                # Use first encountered description for each route_id
                if route_id not in route_descriptions:
                    route_descriptions[route_id] = description

        self.unique_routes = route_descriptions

        # Create description -> route mapping (pick one route per unique description)
        self.description_to_route = {}
        for route_id, description in route_descriptions.items():
            if description not in self.description_to_route:
                self.description_to_route[description] = route_id

        # Analyze unique descriptions
        descriptions_set = set(route_descriptions.values())
        print(f"Found {len(descriptions_set)} unique descriptions from {len(route_descriptions)} route IDs")
        print(f"Using {len(self.description_to_route)} unique route representatives")

        # Categorize routes by type using unique representative routes
        for description, route_id in self.description_to_route.items():
            if "акци" in description.lower() or "предложени" in description.lower():
                self.route_categories["offers"].append(route_id)
            elif "прощани" in description.lower():
                self.route_categories["goodbye"].append(route_id)
            elif "прекращени" in description.lower() or "неадекватност" in description.lower():
                self.route_categories["terminate"].append(route_id)
            elif "график" in description.lower() or "работы" in description.lower():
                self.route_categories["schedule"].append(route_id)
            elif "информаци" in description.lower():
                self.route_categories["info"].append(route_id)
            elif "решени" in description.lower() or "проблем" in description.lower():
                self.route_categories["support"].append(route_id)
            elif "восстанов" in description.lower():
                self.route_categories["recovery"].append(route_id)
            elif "отмена" in description.lower() or "подписк" in description.lower():
                self.route_categories["subscription"].append(route_id)
            else:
                self.route_categories["other"].append(route_id)

        print(f"Found {len(self.unique_routes)} unique routes")
        for category, routes in self.route_categories.items():
            print(f"  {category}: {len(routes)} routes")

    def extract_dialogue_templates(self):
        """Extract dialogue patterns for generation."""
        for item in self.original_data:
            text = item["text"]
            correct_answer = item["rightStepId"]

            # Extract robot and user parts
            parts = text.split("Абонент:")
            if len(parts) == 2:
                robot_part = parts[0].replace("Робот:", "").strip()
                user_part = parts[1].strip()

                template = {
                    "robot_greeting":    robot_part,
                    "user_request":      user_part,
                    "correct_route":     correct_answer,
                    "route_description": self.unique_routes.get(correct_answer, "Unknown")
                }
                self.dialogue_templates.append(template)

        print(f"Extracted {len(self.dialogue_templates)} dialogue templates")

    def generate_dialogue_variations(self, template: Dict) -> List[str]:
        """Generate variations of a dialogue."""
        robot_greetings = [
            "Здравствуйте! Как я могу вам помочь?",
            "Добрый день! Вы обратились в службу поддержки. Чем могу помочь?",
            "Привет! Я помогу вам с настройкой интернета. Опишите проблему.",
            "Здравствуйте! Вам нужна информация о тарифах?",
            "Приветствую! Чем могу помочь?",
            "Добрый день! Хотите узнать о наших акциях?",
            "Здравствуйте! С чем обратились?",
            "Добро пожаловать! Как дела, чем помочь?"
        ]

        # Use original greeting or pick a random one
        robot_text = random.choice([template["robot_greeting"]] + robot_greetings)
        user_text = template["user_request"]

        return f"Робот: {robot_text} Абонент: {user_text}"

    def select_routes_for_item(self, correct_route_id: int, num_routes: int) -> List[Dict]:
        """Select routes including the correct one, avoiding duplicate descriptions."""
        routes = []
        used_descriptions = set()

        # Add correct route
        correct_description = self.unique_routes[correct_route_id]
        routes.append({
            "route_id":    correct_route_id,
            "description": correct_description
        })
        used_descriptions.add(correct_description)

        # Get all available unique descriptions excluding the correct one
        available_descriptions = [desc for desc in self.description_to_route.keys()
                                  if desc != correct_description]

        # Find category of correct route
        correct_category = None
        for category, route_ids in self.route_categories.items():
            if correct_route_id in route_ids:
                correct_category = category
                break

        # Try to add diverse distractors from different categories first
        selected_descriptions = []
        other_categories = [cat for cat in self.route_categories.keys() if cat != correct_category]

        for category in other_categories:
            # Find descriptions for this category
            category_descriptions = []
            for desc in available_descriptions:
                route_id = self.description_to_route[desc]
                if route_id in self.route_categories[category]:
                    category_descriptions.append(desc)

            if category_descriptions:
                # Add up to 2 descriptions from each other category
                selected = random.sample(category_descriptions, min(2, len(category_descriptions)))
                for desc in selected:
                    if len(selected_descriptions) < num_routes - 1:
                        selected_descriptions.append(desc)
                        available_descriptions.remove(desc)

        # Fill remaining slots with any available descriptions
        remaining_slots = num_routes - 1 - len(selected_descriptions)
        if remaining_slots > 0 and available_descriptions:
            additional_descriptions = random.sample(available_descriptions,
                                                    min(remaining_slots, len(available_descriptions)))
            selected_descriptions.extend(additional_descriptions)

        # Convert selected descriptions to routes
        for description in selected_descriptions:
            route_id = self.description_to_route[description]
            routes.append({
                "route_id":    route_id,
                "description": description
            })

        # If we still don't have enough routes, pad with available ones
        # (this can happen if there aren't enough unique descriptions)
        if len(routes) < num_routes:
            print(f"Warning: Only found {len(routes)} unique descriptions out of {num_routes} requested")

        # Shuffle routes so correct answer isn't always first
        random.shuffle(routes)

        return routes

    def generate_synthetic_dataset(self, num_items: int, route_counts: List[int]) -> Dict:
        """Generate synthetic datasets with different route counts."""
        synthetic_datasets = {}

        for route_count in route_counts:
            print(f"Generating dataset with {route_count} routes...")
            dataset = []

            for i in range(num_items):
                # Select a random template
                template = random.choice(self.dialogue_templates)

                # Generate dialogue variation
                dialogue_text = self.generate_dialogue_variations(template)

                # Parse dialogue to messages format
                messages = self.convert_dialog(dialogue_text)

                # Select routes for this item
                routes = self.select_routes_for_item(template["correct_route"], route_count)

                if len(routes) == route_count:  # Only add if we have enough routes
                    dataset.append({
                        "messages":  messages,
                        "routes":    routes,
                        "answer_id": template["correct_route"]
                    })

            synthetic_datasets[route_count] = dataset
            print(f"Generated {len(dataset)} items with {route_count} routes")

        return synthetic_datasets

    def convert_dialog(self, text: str) -> List[Dict]:
        """Convert dialogue text to messages format (from original dataset_prepare.py)."""
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

    def save_statistics(self, datasets: Dict, output_file: str = "synthetic_stats.csv"):
        """Save dataset statistics."""
        stats = []
        for route_count, dataset in datasets.items():
            stats.append({
                "route_count":            route_count,
                "dataset_size":           len(dataset),
                "avg_routes_per_item":    route_count,
                "unique_correct_answers": len(set(item["answer_id"] for item in dataset))
            })

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
        print(f"Statistics saved to {output_file}")


def main():
    # Initialize generator
    generator = SyntheticDatasetGenerator('dataset_input.json')

    # Define route counts to generate (based on available unique descriptions)
    max_unique_descriptions = len(generator.description_to_route)
    print(f"Maximum possible routes without duplicates: {max_unique_descriptions}")

    # Generate practical route counts
    route_counts = [3, 5, 7, 9]  # Only generate what's actually possible

    # Generate synthetic datasets
    num_items_per_dataset = 100  # Generate more items for smaller datasets
    datasets = generator.generate_synthetic_dataset(num_items_per_dataset, route_counts)

    # Save datasets
    generator.save_datasets(datasets)

    # Save statistics
    generator.save_statistics(datasets)

    print("\n=== GENERATION COMPLETE ===")
    print("Generated synthetic datasets:")
    for route_count in route_counts:
        print(f"  - {route_count} routes: {len(datasets[route_count])} items")

    # Print unique descriptions summary
    print(f"\nAvailable unique descriptions ({max_unique_descriptions}):")
    for i, (desc, route_id) in enumerate(generator.description_to_route.items(), 1):
        print(f"  {i}. Route {route_id}: {desc}")


if __name__ == "__main__":
    main()
