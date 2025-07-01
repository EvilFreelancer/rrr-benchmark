import re


def parse_model_name(model: str):
    """
    Parses a model string into base_model, size, and quantization.
    Handles colon- and slash-based names (e.g., 'llama3.1:8b-instruct-q4_K_M', 'hf.co/t-tech/T-pro-it-1.0-Q4_K_M-GGUF')
    """

    original = model.strip()
    model = original

    # Lowercase copy for search
    lowered = model.lower()

    # Match quantization anywhere in the string (including fp16)
    quant_match = re.search(r'(q\d+[_a-z0-9]*|fp16)', lowered)
    quant = quant_match.group(1) if quant_match else ""

    # Match size (like 7b, 13b, 32b)
    size_match = re.search(r'(\d{1,3}b)', lowered)
    size = size_match.group(1) if size_match else ""

    # Cut off known suffixes from base model
    base = model
    if quant:
        base = re.sub(f'[-_/]?{re.escape(quant)}(?=[-_/]|$)', '', base, flags=re.IGNORECASE)
    if size:
        base = re.sub(f'[-_/]?{re.escape(size)}(?=[-_/]|$)', '', base, flags=re.IGNORECASE)
    base = re.sub(r'[-_/]?instruct(?=[-_/]|$)', '', base, flags=re.IGNORECASE)
    
    # Remove everything after the first colon
    if '/' in base:
        base = base.split('/')[-1]
    if ':' in base:
        base = base.split(':')[0]
    
    # Clean suffixes like -GGUF at the end
    base = re.sub(r'-gguf$', '', base, flags=re.IGNORECASE)
    base = base.strip(":/-")

    return base, size, quant.lower()


if __name__ == '__main__':
    MODEL_LIST = [
        'llama3.1:8b-instruct-q4_K_M',
        'llama3.1:8b-instruct-q8_0',
        'llama3.1:8b-instruct-fp16',

        'llama3.2:1b-instruct-q4_K_M',
        'llama3.2:1b-instruct-q8_0',
        'llama3.2:1b-instruct-fp16',

        'llama3.2:3b-instruct-q4_K_M',
        'llama3.2:3b-instruct-q8_0',
        'llama3.2:3b-instruct-fp16',

        'qwen3:8b-q4_K_M',
        'qwen3:8b-q8_0',
        'qwen3:8b-fp16',

        'deepseek-r1:7b-qwen-distill-q4_K_M',
        'deepseek-r1:7b-qwen-distill-q8_0',
        'deepseek-r1:7b-qwen-distill-fp16',
        'deepseek-r1:8b-llama-distill-q4_K_M',
        'deepseek-r1:8b-llama-distill-q8_0',
        'deepseek-r1:8b-llama-distill-fp16',

        'deepseek-v2:16b-lite-chat-q4_K_M',
        'deepseek-v2:16b-lite-chat-q8_0',
        'deepseek-v2:16b-lite-chat-fp16',

        'hf.co/NikolayKozloff/T-pro-it-1.0-Q2_K-GGUF',
        'hf.co/t-tech/T-pro-it-1.0-Q4_K_M-GGUF',
        'hf.co/t-tech/T-pro-it-1.0-Q8_0-GGUF',

        'hf.co/mradermacher/T-lite-it-1.0-GGUF:Q4_K_M',

        'hf.co/ai-sage/GigaChat-20B-A3B-instruct-v1.5-GGUF:Q4_K_M',
    ]

    for model in MODEL_LIST:
        parsed = parse_model_name(model)
        print(parsed)
