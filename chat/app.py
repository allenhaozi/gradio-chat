import argparse
import logging
import os

import gradio as gr
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class OpenAIEndpoint:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "qwen-max")
        self.base_url = os.getenv(
            "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.api_key = os.getenv("API_KEY", "")
        # api base configuration
        self.max_input_length = int(os.getenv("MAX_INPUT_LENGTH", 8192))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 2000))
        self.min_max_tokens = int(os.getenv("MIN_MAX_TOKENS", 16))
        self.max_max_tokens = int(os.getenv("MAX_MAX_TOKENS", 2000))
        self.temperature = float(os.getenv("TEMPERATURE", 0.95))
        self.top_p = float(os.getenv("TOP_P", 0.7))
        self.stop = os.getenv("STOP", None)
        self.stream = os.getenv("STREAM", True)
        self.root_path = os.getenv("ROOT_PATH", "")
        self.server_port = int(os.getenv("SERVER_PORT", 7680))
        self.server_host = os.getenv("SERVER_HOST", "0.0.0.0")


endpoint = OpenAIEndpoint()


client = OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)


def predict(message, history, temperature, top_p, max_tokens):
    logging.info("message: %s", message)
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=endpoint.model_name,
        messages=history_openai_format,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            logging.info("partial_message: %s", partial_message)
            yield partial_message


def start_chat():
    logging.info("starting chat gradio...")
    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask me any question"),
        title="Chat",
        description="Ask me any question",
        theme="default",
        cache_examples=False,
        retry_btn="Try Again",
        undo_btn="Delete Previous",
        clear_btn="Clear",
        additional_inputs=[
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=endpoint.temperature,
                step=0.1,
                interactive=True,
                label="Temperature",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=endpoint.top_p,
                step=0.1,
                interactive=True,
                label="Top P",
            ),
            gr.Slider(
                minimum=endpoint.min_max_tokens,
                maximum=endpoint.max_max_tokens,
                value=endpoint.max_tokens,
                step=64,
                interactive=True,
                label="Max output tokens",
            ),
        ],
        additional_inputs_accordion_name="Parameters",
    ).queue().launch(
        share=False,
        inbrowser=True,
        server_port=endpoint.server_port,
        server_name=endpoint.server_host,
        root_path=endpoint.root_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Chat Gradio Playground")

    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="openai api key",
    )

    parser.add_argument(
        "--env",
        type=str,
        required=False,
        default="prod",
        help="environment",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="qwen-max",
        help="openai model name",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        required=False,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="openai base url",
    )

    args = parser.parse_args()

    if args.env != "prod":
        endpoint.server_host = "127.0.0.1"

    if args.model_name:
        endpoint.model_name = args.model_name

    if args.base_url:
        endpoint.base_url = args.base_url

    logging.info("model_name: %s", endpoint.model_name)
    logging.info("base_url: %s", endpoint.base_url)
    logging.info("env: %s", args.env)

    # pass the api key to the endpoint
    client = OpenAI(api_key=args.api_key, base_url=endpoint.base_url)

    start_chat()
