import numpy as np

import requests
import ollama
from typing import Callable, Iterator, Union, Sequence


class OllamaChat:
    def __init__(self, model_name = "qwen3.6", ollama_url = "http://192.168.69.3:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.client = ollama.Client(host=self.ollama_url)
        # self.client.pull(model=self.model_name)
        self.tools: dict[str, Callable] = {}
        self.tool_callables = []
        self.messages = []
        self.pinned_system_message = ""

    def add_tool(self, name, tool: Callable):
        self.tools[name] = tool
        self.tool_callables.append(tool)

    def pinned_message(self, msg: str):
        self.pinned_system_message = msg

    def limit_messages(self, length: int):
        self.messages = self.messages[-length:]
        self.messages.insert(0, {"role": "system", "content": self.pinned_system_message})

    def msg(self, role: str, msg: str):
        self.messages.append({
            "role": role,
            "content": msg})

    def msg_user(self, msg):
        self.msg("user", msg)

    def msg_system(self, msg):
        self.msg("system", msg)

    def msg_assistant(self, msg):
        self.msg("assistant", msg)

    def msg_tool(self, msg):
        self.msg("tool", msg)

    def chat(self) -> ollama.ChatResponse:
        response: ollama.ChatResponse = self.client.chat(model=self.model_name, messages=self.messages, tools=self.tool_callables, stream=False, think=True)
        tool_calls = response.message.tool_calls or []

        for tool in tool_calls:
            if not tool.function.name in self.tools:
                self.msg_tool(f"Tool {tool.function.name} does not exist!")
                continue
            self.messages.append({
                "role": "tool",
                "name": tool.function.name,
                "content": self.tools[tool.function.name](**tool.function.arguments)})
        self.messages.append(response.message)
        return response

    def embed(self, text: Union[str, Sequence[str]], dimensions: int | None = None):
        return self.client.embed(model=self.model_name, input=text, dimensions=dimensions)

def save_memory(key: str, value: str):
    """
    Save a value into short-term persistent memory. Short-term memory is saved into long-term memory depending on
    tool-internal assigned value, access frequency, and amount. Short-term memory should be assumed to be permanent and
    may only be cleared either by the user or when a predefined memory limit has been reached.

    Args:
        key (str): The key to save the memory under.
        value (str): The memory to save.
    """
    print("model tried to save memory: '", key, "' = '", value, "'")

if __name__ == "__main__":
    chat = OllamaChat("qwen3-embedding:8b")
    chat.add_tool("save_memory", save_memory)
    texts = ["GAY ASS LITTLE BITCH", "the sailors down by the docks"]
    e = [np.array(em) for em in chat.embed(texts).embeddings]

    # for i, embed in enumerate(e1):
        # print(f"({i}) len: {len(embed)}")
    cosine_similarity = np.dot(e[0], e[1]) / (np.linalg.norm(e[0]) * np.linalg.norm(e[1]))
    print(f"Cos: {cosine_similarity}")
    print(f"Dist: {np.linalg.norm(e[0] - e[1])}")
    print(f"Dot: {np.dot(e[0], e[1])}")
    # chat.messages.append({"role": "user", "content": "Hello? Can you try to save some kind of memory? Maybe even save a few different ones. anything you'd like. Make it interesting"})
    # chat.chat()