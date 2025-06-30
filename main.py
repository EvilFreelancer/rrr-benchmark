import asyncio
import logging
import time
import csv
import os
from tqdm import tqdm
from datasets import load_dataset
from structured_router import StructuredRouter
from parse_model_name import parse_model_name

logging.basicConfig(level=logging.INFO)
logging.getLogger('pydantic_core').setLevel(logging.WARNING)
logger = logging.getLogger("RRR")

MAX_RETRIES = 3

# List of models for testing
MODEL_LIST = [
    # 'llama3.1:8b-instruct-q4_K_M',
    # 'llama3.1:8b-instruct-q8_0',
    # 'llama3.1:8b-instruct-fp16',

    # 'llama3.2:1b-instruct-q4_K_M',
    # 'llama3.2:1b-instruct-q8_0',
    # 'llama3.2:1b-instruct-fp16',

    # 'llama3.2:3b-instruct-q4_K_M',
    # 'llama3.2:3b-instruct-q8_0',
    # 'llama3.2:3b-instruct-fp16',

    'qwen3:8b-q4_K_M',
    'qwen3:8b-q8_0',
    'qwen3:8b-fp16',

    # 'deepseek-r1:7b-qwen-distill-q4_K_M',
    # 'deepseek-r1:7b-qwen-distill-q8_0',
    # 'deepseek-r1:7b-qwen-distill-fp16',
    # 'deepseek-r1:8b-llama-distill-q4_K_M',
    # 'deepseek-r1:8b-llama-distill-q8_0',
    # 'deepseek-r1:8b-llama-distill-fp16',

    # 'hf.co/t-tech/T-pro-it-1.0-Q4_K_M-GGUF',
    # 'hf.co/t-tech/T-pro-it-1.0-Q8_0-GGUF',

    # 'deepseek-v2:16b-lite-chat-q4_K_M',
    # 'deepseek-v2:16b-lite-chat-q8_0',
    # 'deepseek-v2:16b-lite-chat-fp16',
]


def load_dataset_from_hf():
    """
    Loads the dataset from Hugging Face Hub (evilfreelancer/rrr-benchmark).
    Returns a list of dicts with keys: messages, routes, answer_id.
    """
    ds = load_dataset("evilfreelancer/rrr-benchmark", split="train")
    return [
        {
            "messages":  item["messages"],
            "routes":    item["routes"],
            "answer_id": item["answer_id"]
        }
        for item in ds
    ]


def save_report_to_csv(report: dict, filename: str):
    """
    Appends a dictionary-based report to a CSV file.

    If the file doesn't exist yet, it creates one and writes headers first.
    """
    fieldnames = list(report.keys())

    write_header = not os.path.exists(filename)

    with open(filename, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(report)

    logger.info(f"Report added to {filename}")


async def test_router(agent, dataset):
    """
    Tests an AI router on a given dataset by querying its predictions
    and comparing them against ground truth labels.
    """
    total_tests = len(dataset)
    correct = 0
    valid_responses = 0
    total_time = 0.0
    total_tokens = 0
    responses_with_tokens = 0

    logger.info(f"Starting tests for model {agent.model} on {total_tests} examples.")

    for item in tqdm(dataset, desc=f"Testing ({agent.model})"):
        messages = item["messages"]
        routes = item["routes"]
        expected_id = item["answer_id"]

        result = None
        attempt = 0

        while attempt < MAX_RETRIES:
            attempt += 1
            logger.debug(f"Attempt {attempt} for {messages}")

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
                    logger.warning(f"Got None response at attempt {attempt}")
            except Exception as e:
                logger.exception(f"Error during attempt {attempt}: {e}")

        if result is None:
            logger.error(f"Failed to get valid response after {MAX_RETRIES} attempts")
            predicted_id = None
        else:
            predicted_id = result.route_id

        if predicted_id == expected_id:
            correct += 1

    accuracy = correct / total_tests if total_tests else 0
    avg_time = total_time / valid_responses if valid_responses else 0
    avg_tokens = total_tokens / responses_with_tokens if responses_with_tokens else 0

    model_name, model_size, model_quant = parse_model_name(agent.model)

    logger.info("\n=== REPORT FOR MODEL ===")
    logger.info(f"Model: {agent.model}")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Valid responses: {valid_responses}")
    logger.info(f"Correct answers: {correct}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Average generation time: {avg_time:.3f} seconds")
    logger.info(f"Average token count: {avg_tokens:.1f}")

    report = {
        "model":             agent.model,
        "model_name":        model_name,
        "model_size":        model_size,
        "model_quant":       model_quant,
        "total_tests":       total_tests,
        "valid_responses":   valid_responses,
        "correct_responses": correct,
        "accuracy":          round(accuracy, 4),
        "avg_response_time": round(avg_time, 3),
        "avg_token_count":   round(avg_tokens, 1),
    }

    return report


def main():
    dataset = load_dataset_from_hf()
    logger.info(f"Loaded {len(dataset)} test cases from Hugging Face Hub")

    all_reports = []
    for model_name in MODEL_LIST:
        logger.info(f"\n=== TESTING MODEL: {model_name} ===")
        agent = StructuredRouter(model=model_name)

        # Monkey patch client to collect raw response data
        orig_chat = agent.client.chat

        def patched_chat(*args, **kwargs):
            result = orig_chat(*args, **kwargs)
            agent.client.last_response_data = result
            return result

        agent.client.chat = patched_chat

        report = asyncio.run(test_router(agent, dataset))

        # Save report for this model
        save_report_to_csv(report, f"test_{model_name.replace(':', '_').replace('/', '_')}.csv")

        all_reports.append(report)

    logger.info("\n=== OVERALL SUMMARY ===")
    for r in all_reports:
        logger.info(
            f"{r['model']} | "
            f"accuracy: {r['accuracy']:.4f} | "
            f"correctness: {r['correct_responses']}/{r['total_tests']} | "
            f"avg_time: {r['avg_response_time']:.3f}s | "
            f"avg_tokens: {r['avg_token_count']}"
        )


if __name__ == "__main__":
    main()
