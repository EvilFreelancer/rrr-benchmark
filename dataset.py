import json
import re


def convert_dialog(text):
    """Парсит однострочный текст с маркерами 'Робот:' и 'Абонент:'."""
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


def process_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    processed = []
    for item in dataset:
        messages = convert_dialog(item["text"])
        routes = [{"id": step["id"], "sense": step["sense"]} for step in item["steps"]]
        right_step_id = item["rightStepId"]

        processed.append({
            "messages":    messages,
            "routes":      routes,
            "rightStepId": right_step_id
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Конвертация завершена! Результат записан в {output_file}")


process_dataset('dataset_input.json', 'dataset_output.json')
