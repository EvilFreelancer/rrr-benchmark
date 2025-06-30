import json
import pandas as pd
from collections import Counter


def analyze_routes(input_file):
    """
    Analyzes routes from dataset_input.json and counts unique ones.
    """
    print(f"Loading data from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Total dataset items: {len(dataset)}")

    # Collect all routes
    all_routes = []
    route_descriptions = {}

    for item in dataset:
        for step in item["steps"]:
            route_id = step["id"]
            route_sense = step["sense"]

            all_routes.append(route_id)

            # Store description for each route_id (in case there are duplicates)
            if route_id in route_descriptions:
                if route_descriptions[route_id] != route_sense:
                    print(f"Warning: Route {route_id} has different descriptions:")
                    print(f"  Previous: {route_descriptions[route_id]}")
                    print(f"  Current:  {route_sense}")
            else:
                route_descriptions[route_id] = route_sense

    # Count unique routes
    unique_routes = set(all_routes)
    route_counts = Counter(all_routes)

    print(f"\n=== ROUTE ANALYSIS ===")
    print(f"Total route mentions: {len(all_routes)}")
    print(f"Unique routes: {len(unique_routes)}")

    # Show most common routes
    print(f"\n=== TOP 10 MOST COMMON ROUTES ===")
    for route_id, count in route_counts.most_common(10):
        description = route_descriptions.get(route_id, "Unknown")
        print(f"Route {route_id}: {count} times - {description}")

    # Show least common routes
    print(f"\n=== TOP 10 LEAST COMMON ROUTES ===")
    for route_id, count in route_counts.most_common()[-10:]:
        description = route_descriptions.get(route_id, "Unknown")
        print(f"Route {route_id}: {count} times - {description}")

    # Create DataFrame for analysis
    route_data = []
    for route_id in unique_routes:
        route_data.append({
            'route_id':    route_id,
            'description': route_descriptions[route_id],
            'frequency':   route_counts[route_id]
        })

    df = pd.DataFrame(route_data)
    df = df.sort_values('frequency', ascending=False)

    # Save detailed analysis
    output_file = 'route_analysis.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nDetailed analysis saved to {output_file}")

    # Analyze rightStepId distribution
    print(f"\n=== CORRECT ANSWER ANALYSIS ===")
    correct_answers = [item["rightStepId"] for item in dataset]
    correct_answer_counts = Counter(correct_answers)

    print(f"Total correct answers: {len(correct_answers)}")
    print(f"Unique correct answers: {len(set(correct_answers))}")

    print(f"\n=== TOP 10 MOST COMMON CORRECT ANSWERS ===")
    for route_id, count in correct_answer_counts.most_common(10):
        description = route_descriptions.get(route_id, "Unknown")
        print(f"Route {route_id}: {count} times - {description}")

    return df, route_descriptions, route_counts


if __name__ == "__main__":
    df, descriptions, counts = analyze_routes('dataset_input.json')
