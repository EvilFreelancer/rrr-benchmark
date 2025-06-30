import json
import logging
import os
from typing import Any
import re

from pydantic_core import from_json
from pydantic import BaseModel, Field
import pydantic
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').level = logging.ERROR
logger = logging.getLogger(__name__)

# Environment variables
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'llama3.1:8b')

SYSTEM_PROMPT = f"""\
Вы интеллектуальный ассистент для маршрутизации запросов. Ваша задача - анализировать историю диалога и входной запрос, чтобы определить наиболее подходящий следующий route_id из предоставленного списка доступных маршрутов.
Инструкции:
1. Учти контекст предыдущих сообщений в диалоге
2. Проанализируйте диалог и его скрытые интенции
3. Сопоставьте диалог с доступными условиями которые описаны в senses
4. Выберите ТОЛЬКО ОДИН sense по условиям и соответствующий id маршрута для формирования ответа
5. Объясните логику выбора коротким сообщением, учитывая как текущий запрос, так и историю взаимодействия
6. В случае неоднозначности, выберите наиболее вероятный маршрут
7. Требования к ответу:
- reasoning: короткое обоснование выбора именно этого id с учётом контекста (обязательно)
- route_id: Целочисленный идентификатор следующего маршрута из предоставленного списка (обязательно)

Формат ответа: Только валидный JSON без пояснений или текста.

Пример:

{{"reasoning": "...", "route_id": N}}

Помните:
- Выбирайте маршрут максимально точно соответствующий условиям которые описаны в поле sense
- Учитывайте последовательность запросов в диалоге
- Всегда возвращайте ОБА поля в ответе
"""


class Route(BaseModel):
    """Detecting which router to use"""
    reasoning: str = Field(
        description="Детальное обоснование выбора именно этого route_id с учётом контекста (обязательно)")
    route_id: int = Field(
        description="Целочисленный идентификатор следующего роута из предоставленного списка (обязательно)")


def simplify_schema(schema: dict, defs: dict = None) -> Any:
    """
    Recursively build a simplified dictionary describing the schema.
    """
    # On first call, store the top-level "$defs" so we can resolve refs.
    if defs is None:
        defs = schema.get('$defs', {})

    # If this schema references something by "$ref", resolve that first.
    if '$ref' in schema:
        ref_path = schema['$ref']
        # Typical ref is like "#/$defs/SomeModel"
        ref_name = ref_path.split('/')[-1]
        ref_schema = defs.get(ref_name, {})
        return simplify_schema(ref_schema, defs)

    schema_type = schema.get('type')

    if schema_type == 'object':
        # For an object, recursively simplify each property
        properties = schema.get('properties', {})
        result = {}
        for prop_name, prop_schema in properties.items():
            result[prop_name] = simplify_schema(prop_schema, defs)
        return result

    elif schema_type == 'array':
        # For an array, look at its "items" and recurse
        items_schema = schema.get('items', {})
        return [simplify_schema(items_schema, defs)]

    else:
        # Assume a primitive type
        field_type = schema.get('type', 'string')  # default to string if missing
        field_desc = schema.get('description', '')
        return f"({field_type}) {field_desc}"


class StructuredOutput:
    def __init__(
        self,
        system_prompt=None,
        schema=None,
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL_NAME,
        num_ctx=1000,
        enable_validation=True,
        temperature=0.1,
        **kwargs
    ):
        self.schema = None
        if schema is not None:
            self.schema = schema

            # Generate the schema for the Route model
            schema_json = self.schema.model_json_schema()
            field_descriptions = simplify_schema(schema_json)
            # print(field_descriptions)

            # Merge JSON schema into the system prompt
            self.system_prompt = (
                system_prompt
                + "\n\nOUTPUT JSON SCHEMA:\n"
                + json.dumps(field_descriptions, indent=2, ensure_ascii=False)
            )

        else:
            # Use default system prompt if schema is not provided
            self.system_prompt = system_prompt

        self.enable_validation = enable_validation
        self.num_ctx = num_ctx or 2048
        self.temperature = temperature or 0.1

        self.base_url = base_url
        if base_url is None:
            self.base_url = OLLAMA_BASE_URL

        self.model = model
        if model is None:
            self.model = OLLAMA_MODEL_NAME
        logger.debug(f"Using model: {self.model}")

        self.client = Client(host=self.base_url)

    async def query(self, routes: list[dict], messages: list[dict]) -> dict | None:
        try:

            # Add the system message if it's not already present
            if 'system' not in messages[0]:
                # Add the routes to the system prompt
                routes_text = "\n".join([f"{route['id']} - {route['sense']}" for route in routes])
                full_system_prompt = self.system_prompt + f"\n\nМаршруты:\n" + routes_text

                # Add the system message to the beginning of the conversation history
                messages = [{"role": "system", "content": full_system_prompt}] + messages
                # print(json.dumps(messages, ensure_ascii=False, indent=2))

            response_data = self.client.chat(
                model=self.model,
                messages=messages,
                format="json",
                options={
                    "temperature": self.temperature,
                    "num_ctx":     self.num_ctx,
                    "timeout":     60,
                },
            )
            # print(response_data)

            # Print API usage
            tok_in = response_data.prompt_eval_count
            tok_out = response_data.eval_count
            logger.debug(f"> USAGE / {self.model} / IN:{tok_in} / OUT:{tok_out}")

            # Extract generated content
            if "message" not in response_data or "content" not in response_data["message"]:
                logger.error(f"Invalid response format: {response_data}")
                return None

            # Extract text from the assistant's message
            generated_text = response_data.message.content
            logger.debug(f"Generated text:\n{generated_text}")
            cleaned_text = re.sub(r"^<think>.*?</think>\s*", "", generated_text, flags=re.DOTALL)

            # Now parse that text as JSON
            try:
                if self.schema is None:
                    result_json = {"response": cleaned_text}
                else:
                    if self.enable_validation:
                        # Validate the response against the schema
                        result_json = self.schema.model_validate(from_json(cleaned_text, allow_partial=True))
                    else:
                        # Just parse the response
                        result_json = from_json(cleaned_text, allow_partial=True)

            except json.JSONDecodeError as e:
                logger.exception("Invalid JSON in LLM response", exc_info=e)
                return None

            return result_json

        # except Exception as e:
        #     logger.exception("Network or other error", exc_info=e)
        #     return None
        except pydantic.ValidationError as e:
            print(messages, routes)
            logger.warning(f"ValidationError (невалидный JSON по схеме): {e}")
            return None  # не делать retry, сразу выход


class StructuredRouter(StructuredOutput):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            schema=Route,
            **kwargs
        )
