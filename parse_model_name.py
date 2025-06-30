import re


def parse_model_name(model: str):
    """
    Parses a model string into base_model, size, and quantization.
    Handles colon- and slash-based names (e.g., 'llama3.1:8b-instruct-q4_K_M', 'hf.co/t-tech/T-pro-it-1.0-Q4_K_M-GGUF')
    """

    original = model.strip()

    # Clean suffixes like -GGUF
    model = re.sub(r'-gguf$', '', original, flags=re.IGNORECASE)

    # Lowercase copy for search
    lowered = model.lower()

    # Match quantization at the end
    quant_match = re.search(r'(q\d+[_a-z0-9]*)$', lowered)
    quant = quant_match.group(1) if quant_match else ""

    # Match size (like 7b, 13b, 32b)
    size_match = re.search(r'(\d{1,3}b)', lowered)
    size = size_match.group(1) if size_match else ""

    # Cut off known suffixes from base model
    base = model
    if quant:
        base = re.sub(f'[-_/]{quant}$', '', base, flags=re.IGNORECASE)
    if size:
        base = re.sub(f'[-_/]{size}', '', base, flags=re.IGNORECASE)
    base = re.sub(r'[-_/]?instruct$', '', base, flags=re.IGNORECASE)
    base = base.strip(":/-")

    return base.lower(), size, quant.lower()


if __name__ == '__main__':
    MODEL_LIST = [
        'llama3.1:8b-instruct-q4_K_M',
        'llama3.1:8b-instruct-q8_0',
        'llama3.2:1b-instruct-q4_K_M',
        'llama3.2:1b-instruct-q8_0',
        'llama3.2:3b-instruct-q4_K_M',
        'llama3.2:3b-instruct-q8_0',
        'qwen3:8b-q4_K_M',
        'qwen3:8b-q8_0',
        'deepseek-r1:7b-qwen-distill-q4_K_M',
        'deepseek-r1:7b-qwen-distill-q8_0',
        'deepseek-r1:8b-llama-distill-q4_K_M',
        'deepseek-r1:8b-llama-distill-q8_0',
        'hf.co/t-tech/T-pro-it-1.0-Q4_K_M-GGUF',
        'hf.co/t-tech/T-pro-it-1.0-Q8_0-GGUF',
    ]

    for model in MODEL_LIST:
        parsed = parse_model_name(model)
        print(parsed)
