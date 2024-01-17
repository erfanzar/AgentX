import warnings
from typing import List, Any, Optional, Tuple
from datetime import datetime
from ..inference import InferenceSession


class BaseInteract:
    def __init__(
            self,
            prompter_type: str,
            user_message_token: str,
            assistant_message_token: str,
            end_of_turn_token: Optional[str] = None,
            user_name: Optional[str] = None,
            assistant_name: Optional[str] = None
    ):
        self.prompter_type = prompter_type
        self.user_message_token = user_message_token
        self.assistant_message_token = assistant_message_token
        self.end_of_turn_token = end_of_turn_token
        user_name = user_name or "Dear User"
        assistant_name = assistant_name or "cLLM"
        self.user_name = user_name
        self.assistant_name = assistant_name

    def format_history_prefix(
            self,
            history: list[list[str]],
            system_message: str,
    ) -> str:
        raise NotImplementedError("NotImplementedYet !")

    def format_message(
            self,
            prompt: str,
            history: list[list[str]],
            system_message: Optional[str],
            prefix: Optional[str]
    ) -> str:
        raise NotImplementedError("NotImplementedYet !")

    def content_finder(
            self,
            prompt: str,
            formatted_prompt: str,
            history: list[list[str]],
            system_message: str,
            external_data: str | Any
    ) -> str:
        raise NotImplementedError("NotImplementedYet !")

    def get_prefix_prompt(
            self,
            system_message: str = None,
            dialogue: list[list[str]] = None
    ) -> str:
        if system_message is None:
            system_message = (
                f"Text transcript of a never-ending dialog, where {self.user_name} interacts with an AI assistant "
                f"named {self.assistant_name}. {self.assistant_name} is helpful, kind, honest, friendly, "
                f"good at writing,"
                f"and never fails to answer {self.user_name}'s "
                "requests immediately and with details and precision."
                f"There are no annotations like (30 seconds passed...) or (to himself), just what {self.user_name} and"
                f" {self.assistant_name} say aloud to each other."
                "The dialog lasts for years, and the entirety of it is shared below. It's 10000 pages long."
                "The transcript only includes text; it does not include markup like HTML and Markdown."

            )
        if dialogue is None:
            dialogue = [
                [
                    f"Hello, {self.assistant_name}!",
                    f"Greetings, {self.user_name}! How may I assist you today?"
                ],
                [
                    "What's the current year?",
                    f"We are currently in the year {datetime.now().year}."
                ],
                [
                    "Can you tell me about the tallest mountain in the world?",
                    "The tallest mountain in the world is Mount Everest, located in the Himalayas."
                ],
                [
                    "What are some interesting facts about Mount Everest?",
                    "Mount Everest is the highest peak on Earth, with a summit reaching 29,032 feet above sea level."
                    " It is a part of the Himalayan range and is known for its challenging climbing routes and "
                    "extreme weather conditions."
                ],
                [
                    "Describe a tiger.",
                    "A tiger is the largest cat species, known for its distinctive orange coat with black stripes. "
                    "It is a powerful predator and is native to various parts of Asia."
                ],
                [
                    "How can I handle errors in a Python program?",
                    "In Python, you can handle errors using try-except blocks. This allows you to catch and handle "
                    "exceptions gracefully, preventing your program from crashing."
                ],
                [
                    "Name a popular programming language.",
                    "Python."
                ],
                [
                    "What time is it?",
                    f"The current time is {datetime.utcnow()} UTC."
                ],
            ]
        prefix = self.format_history_prefix(
            history=dialogue,
            system_message=system_message
        )
        return prefix

    def filter_response(
            self,
            response: str,
    ) -> str:
        response = response.replace(
            self.user_message_token, ""
        ).replace(
            self.assistant_message_token, ""
        )
        return response

    def get_stop_signs(self) -> List[str]:
        return [self.user_message_token, self.end_of_turn_token]

    def basic_generation(
            self,
            inference: InferenceSession,
            user_prompt: str,
            history: Optional[list[list[str]]] = None,
            prefix_chat: Optional[str] = None,
            maximum_generation_depth: int = 3,
            minimum_generation_tokens: int = 3
    ) -> Tuple[str, list[list[str]]]:
        history = history or []
        model_prompt = self.format_message(
            prompt=user_prompt,
            history=history,
            system_message=None,
            prefix=prefix_chat
        )

        def generate_response(depth: int = 0):
            response = ""
            for response_byte in inference.generate(
                    prompt=model_prompt,
            ):
                next_token = response_byte.predictions.text
                response += next_token

            if len(response) < minimum_generation_tokens:
                if depth > maximum_generation_depth:
                    return response
                warnings.warn(f"Re-Generating Response")
                return generate_response(depth + 1)
            return response

        total_model_response = generate_response()
        result = self.filter_response(total_model_response)
        history.append([user_prompt, result])
        return result, history
