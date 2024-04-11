from enum import Enum

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)

MAX_LENGTH_OUTPUT = 512


class LLMType(Enum):
    LLAMA2_70B_CHAT = "llama2_70b_chat"
    MIXTRAL_8X7B_INSTRUCT = "mixtral_8x7b_instruct"
    TINY_STORIES_33M = "tiny_stories_33m"

    @property
    def path(self):
        match self:
            case LLMType.LLAMA2_70B_CHAT:
                return "meta-llama/Llama-2-70b-chat-hf"
            case LLMType.MIXTRAL_8X7B_INSTRUCT:
                return "mistralai/Mixtral-8x7B-Instruct-v0.1"
            case LLMType.TINY_STORIES_33M:
                return "roneneldan/TinyStories-33M"

    @property
    def tokenizer(self):
        if self == LLMType.LLAMA2_70B_CHAT:
            return LlamaTokenizerFast
        return AutoTokenizer

    @property
    def model(self):
        if self == LLMType.LLAMA2_70B_CHAT:
            return LlamaForCausalLM
        return AutoModelForCausalLM

    @property
    def dtype(self):
        match self:
            case LLMType.LLAMA2_70B_CHAT:
                return torch.float16
            case LLMType.MIXTRAL_8X7B_INSTRUCT:
                return torch.bfloat16
            case LLMType.TINY_STORIES_33M:
                return torch.float32

    @property
    def context_length(self):
        match self:
            case LLMType.LLAMA2_70B_CHAT:
                return 4096
            case LLMType.MIXTRAL_8X7B_INSTRUCT:
                return 32768
            case LLMType.TINY_STORIES_33M:
                return 2048

    @property
    def max_length_input(self):
        return self.context_length - MAX_LENGTH_OUTPUT - 1

    @property
    def max_length_output(self):
        return MAX_LENGTH_OUTPUT
