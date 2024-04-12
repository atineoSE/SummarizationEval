import argparse
import os
import time
from datetime import timedelta
from decimal import Decimal
from itertools import chain
from math import ceil
from typing import Any

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers.tokenization_utils import PreTrainedTokenizerBase

from LLM_type import LLMType
from prompts import get_summarization_prompt, summarization_prompt

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
cache_dir = "/ephemeral/models"

if HF_TOKEN:
    login(token=HF_TOKEN)


class LLM:
    llm_type: LLMType
    tokenizer: PreTrainedTokenizerBase
    model: Any

    def __init__(self, llm_type: LLMType):
        self.llm_type = llm_type

        # Load and configure tokenizer
        self.tokenizer = llm_type.tokenizer.from_pretrained(
            llm_type.path, padding_side="left", cache_dir=cache_dir
        )
        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
        )

        # Load model onto GPUs
        self.model = llm_type.model.from_pretrained(
            llm_type.path,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=llm_type.dtype,
            cache_dir=cache_dir,
        )

    def encode_prompts(self, prompts):
        input_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_tokens = {
            k: input_tokens[k]
            for k in input_tokens
            if k in ["input_ids", "attention_mask"]
        }

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to("cuda")

        return input_tokens

    def generate_text(self, input_tokens):
        return self.model.generate(
            **input_tokens,
            do_sample=False,
            max_new_tokens=self.llm_type.max_length_output,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def split_output(self, prompt_template, output):
        split_sequence = prompt_template.split("\n")[-2]
        return output.split(split_sequence)[-1].strip()

    def infer(self, prompts, prompt_template):
        input_tokens = self.encode_prompts(prompts)
        output = self.generate_text(input_tokens)
        generations = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return list(map(lambda o: self.split_output(prompt_template, o), generations))

    def get_tokens_per_seconds(self, all_summaries, seconds):
        total_generated_tokens = sum(
            map(lambda x: len(self.tokenizer.tokenize(x)), all_summaries)
        )
        tokens_per_seconds = total_generated_tokens / seconds
        tokens_per_seconds_str = Decimal(tokens_per_seconds).quantize(Decimal("0.01"))

        return tokens_per_seconds_str

    def get_all_summarization_prompts(self, df):
        prompts = []
        for _, row in df.iterrows():
            prompt = get_summarization_prompt(
                self.tokenizer.tokenize,
                self.tokenizer.convert_tokens_to_string,
                row["title"],
                row["transcript"],
                self.llm_type.max_length_input,
            )
            prompts.append(prompt)
        return prompts

    def get_all_summaries(self, df, batch_size=1):
        all_summaries = []
        all_prompts = self.get_all_summarization_prompts(df)
        n = df.shape[0]
        for i in range(0, ceil(n / batch_size)):
            l_i = i * batch_size
            h_i = min((i + 1) * batch_size, n)
            titles = df[l_i:h_i]["title"].str.cat(sep="\n\t")
            print(f"{i}: summarizing titles:\n\t{titles}")
            summaries = self.infer(
                prompts=all_prompts[l_i:h_i], prompt_template=summarization_prompt
            )
            all_summaries += summaries

        return all_summaries

    def store_summaries(self, all_summaries, df):
        df[f"{self.llm_type.name}_summary"] = all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mixtral_8x7b_instruct")
    parser.add_argument(
        "--file", type=str, default="./transcripts/awesome_nature_100.csv"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    model = LLM(LLMType(args.model))
    df = pd.read_csv(args.file)

    start_time = time.time()
    summaries = model.get_all_summaries(df, args.batch_size)
    stop_time = time.time()
    elapsed_time = timedelta(seconds=stop_time - start_time)
    tokens_per_seconds = model.get_tokens_per_seconds(summaries, elapsed_time.seconds)

    print(
        f"Generated {len(summaries)} summaries at "
        + f"{tokens_per_seconds} tokens/seconds in {str(elapsed_time).split('.')[0]} "
        + f"with batch size {args.batch_size}"
    )

    model.store_summaries(summaries, df)
    basename = args.file.split("/")[-1].split(".")[0]
    df.to_csv(f"./summaries/{basename}_{args.model}_summaries.csv")
