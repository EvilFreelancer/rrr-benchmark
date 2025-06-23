import json
import asyncio
import logging
import time
import csv
from tqdm import tqdm

from structured_router import StructuredRouter  # путь подкорректируй под свой проект

logging.basicConfig(level=logging.INFO)
logging.getLogger('pydantic_core').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# Список моделей для тестирования
MODEL_LIST = [
    'llama3.2:1b-instruct-q4_K_M',
    'llama3.2:1b-instruct-q8_0',
    'llama3.2:3b-instruct-q4_K_M',
    'llama3.2:3b-instruct-q8_0',
    'qwen3:8b-q4_K_M',
    'qwen3:8b-q8_0',
    'llama3.1:8b-instruct-q4_K_M',
    'llama3.1:8b-instruct-q8_0',
    'deepseek-r1:7b-qwen-distill-q4_K_M',
    'deepseek-r1:7b-qwen-distill-q8_0'
]


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_report_to_csv(report: dict, filename: str):
    fieldnames = list(report.keys())

    write_header = not os.path.exists(filename)

    with open(filename, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(report)

    logger.info(f"Отчёт добавлен в {filename}")


async def test_router(agent, dataset):
    total_tests = len(dataset)
    correct = 0
    valid_responses = 0
    total_time = 0.0
    total_tokens = 0
    responses_with_tokens = 0

    logger.info(f"Начало тестирования модели {agent.model} на {total_tests} примерах")

    for item in tqdm(dataset, desc=f"Тестирование ({agent.model})"):
        messages = item["messages"]
        routes = item["routes"]
        expected_id = item["rightStepId"]

        result = None
        attempt = 0
        start_time = None

        while attempt < MAX_RETRIES:
            attempt += 1
            logger.debug(f"Попытка {attempt} для {messages}")

            try:
                start_time = time.time()
                result = await agent.query(routes=routes, messages=messages)
                elapsed = time.time() - start_time

                if result is not None:
                    valid_responses += 1
                    total_time += elapsed

                    if hasattr(agent.client, "last_response_data"):
                        response_data = agent.client.last_response_data
                        eval_count = getattr(response_data, "eval_count", None)
                        if eval_count:
                            total_tokens += eval_count
                            responses_with_tokens += 1

                    break
                else:
                    logger.warning(f"Ответ None на попытке {attempt}")
            except Exception as e:
                logger.exception(f"Ошибка на попытке {attempt}: {e}")

        if result is None:
            logger.error(f"Не удалось получить валидный ответ после {MAX_RETRIES} попыток")
            predicted_id = None
        else:
            predicted_id = result.route_id

        if predicted_id == expected_id:
            correct += 1

    accuracy = correct / total_tests if total_tests else 0
    avg_time = total_time / valid_responses if valid_responses else 0
    avg_tokens = total_tokens / responses_with_tokens if responses_with_tokens else 0

    logger.info("\n=== ОТЧЁТ ПО МОДЕЛИ ===")
    logger.info(f"Модель: {agent.model}")
    logger.info(f"Всего тестов: {total_tests}")
    logger.info(f"Валидных ответов: {valid_responses}")
    logger.info(f"Правильных ответов: {correct}")
    logger.info(f"Точность: {accuracy:.4f}")
    logger.info(f"Среднее время генерации: {avg_time:.3f} секунд")
    logger.info(f"Среднее количество токенов: {avg_tokens:.1f}")

    report = {
        "model":             agent.model,
        "total_tests":       total_tests,
        "valid_responses":   valid_responses,
        "correct_responses": correct,
        "accuracy":          round(accuracy, 4),
        "avg_response_time": round(avg_time, 3),
        "avg_token_count":   round(avg_tokens, 1),
    }

    return report


def main():
    dataset = load_dataset("dataset_output.json")
    logger.info(f"Загружено {len(dataset)} тестовых примеров")

    summary_filename = "summary.csv"
    all_reports = []

    for model_name in MODEL_LIST:
        logger.info(f"\n=== ТЕСТИРОВАНИЕ МОДЕЛИ: {model_name} ===")
        agent = StructuredRouter(model=model_name)

        # monkey-patch client для сбора raw response
        orig_chat = agent.client.chat

        def patched_chat(*args, **kwargs):
            result = orig_chat(*args, **kwargs)
            agent.client.last_response_data = result
            return result

        agent.client.chat = patched_chat

        report = asyncio.run(test_router(agent, dataset))

        # Сохраняем отчёт для этой модели
        save_report_to_csv(report, f"test_{model_name.replace(':', '_')}.csv")

        # Добавляем в сводный отчёт
        save_report_to_csv(report, summary_filename)

        all_reports.append(report)

    logger.info("\n=== ОБЩАЯ СВОДКА ===")
    for r in all_reports:
        logger.info(f"{r['model']} | точность: {r['accuracy']:.4f} | правильных: {r['correct_responses']}/{r['total_tests']} | avg_time: {r['avg_response_time']:.3f}s | avg_tokens: {r['avg_token_count']}")

    logger.info(f"Сводный отчёт сохранён в {summary_filename}")


if __name__ == "__main__":
    main()
