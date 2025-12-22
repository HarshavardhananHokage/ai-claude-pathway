import os

import tiktoken as tk
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TokenPlusCostCounter:

    MODEL_COSTS = {
        "gpt-5": {"input" : 1.25, "context_window":400_000, "use_tokenizer" : "API"},
        "gpt-4o": {"input": 2.50, "context_window":128_000, "use_tokenizer": "USE_TIKTOKEN"},
    }

    def __init__(self, model):
        self.prompt = None
        self.token_count = 0
        self.model = model

    def describe(self):
        description = (f"We are going to count tokens and calculate costs for the prompt: {self.prompt}.\n"
                       f"The model used for this purpose is {self.model}")
        print(description)

    def calculate_costs(self, input_rate):
        input_cost = input_rate * (self.token_count/1_000_000)
        return input_cost

    def count_tokens_using_tiktoken(self):
        encoding = tk.encoding_for_model(self.model)
        encoded_tokens = encoding.encode(self.prompt)
        self.token_count = len(encoded_tokens)
        return self.token_count

    def count_tokens_using_api(self):
        client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
        response = client.responses.create(model=self.model,
                                           input=self.prompt,
                                           max_output_tokens=16)
        self.token_count = response.usage.input_tokens
        return self.token_count

    def count_tokens(self, use_tokenizer):
        if "API" == use_tokenizer:
            return self.count_tokens_using_api()
        else:
            return self.count_tokens_using_tiktoken()

    def check_against_context_window(self, context_window):
        print(f"Context window for model: {context_window}")
        print(f"You are using {self.token_count} tokens out of {context_window}")
        if self.token_count / context_window < 0.5:
            return "You are safe. Continue prompting"
        else:
            return "Warning! Prompt greater than 50% of context window. Optimize prompt for better answers."

    def count_tokens_and_calculate_costs(self, prompt):
        self.prompt = prompt
        self.describe()

        if self.model in self.MODEL_COSTS.keys():
            use_tokenizer = self.MODEL_COSTS.get(self.model).get("use_tokenizer")
            input_rate = self.MODEL_COSTS.get(self.model).get("input")
            context_window = self.MODEL_COSTS.get(self.model).get("context_window")

            print(f"Total token count: {self.count_tokens(use_tokenizer)}")
            print(f"Total Cost for prompt: ${self.calculate_costs(input_rate)}")
            print(self.check_against_context_window(context_window))
            print()
        else:
            print("Not supported model")

gpt_4o_checker = TokenPlusCostCounter("gpt-4o")
gpt_4o_checker.count_tokens_and_calculate_costs("Hello. How are you?")

gpt_5_checker = TokenPlusCostCounter("gpt-5")
gpt_5_checker.count_tokens_and_calculate_costs("Hello. How are you?")