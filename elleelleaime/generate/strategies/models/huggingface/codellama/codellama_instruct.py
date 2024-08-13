from elleelleaime.generate.strategies.strategy import PatchGenerationStrategy
from dataclasses import dataclass
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List, Generator

import torch
import threading
import logging


@dataclass
class GenerateSettings:
    name: str
    do_sample: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 10
    max_length: int = 16384


class CodeLLaMAIntruct(PatchGenerationStrategy):
    __SUPPORTED_MODELS = {
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "meta-llama/CodeLlama-13b-Instruct-hf",
        "meta-llama/CodeLlama-34b-Instruct-hf",
        "meta-llama/CodeLlama-70b-Instruct-hf",
    }

    __GENERATION_STRATEGIES = {
        "beam_search": GenerateSettings(
            name="beam_search",
        ),
        "sampling": GenerateSettings(
            name="sampling",
            do_sample=True,
        ),
    }

    __MODEL = None
    __TOKENIZER = None
    __MODELS_LOADED: bool = False
    __MODELS_LOCK: threading.Lock = threading.Lock()

    def __init__(self, model_name: str, **kwargs) -> None:
        assert (
            model_name in self.__SUPPORTED_MODELS
        ), f"Model {model_name} not supported by {self.__class__.__name__}"
        self.model_name = model_name
        self.__load_model(**kwargs)
        # Generation settings
        assert (
            kwargs.get("generation_strategy", "sampling")
            in self.__GENERATION_STRATEGIES
        ), f"Generation strategy {kwargs.get('generation_strategy', 'samlping')} not supported by {self.__class__.__name__}"
        self.generate_settings = self.__GENERATION_STRATEGIES[
            kwargs.get("generation_strategy", "samlping")
        ]
        self.batch_size = kwargs.get("batch_size", 4)
        self.generate_settings.num_return_sequences = kwargs.get(
            "num_return_sequences", GenerateSettings.num_return_sequences
        )
        self.generate_settings.num_beams = kwargs.get(
            "num_beams", GenerateSettings.num_beams
        )
        self.generate_settings.temperature = kwargs.get(
            "temperature", GenerateSettings.temperature
        )

    def __load_model(self, **kwargs):
        # Setup environment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_size = 16384

        # Setup kwargs
        kwargs = dict(
            torch_dtype=torch.bfloat16,
        )

        # Load the model and tokenizer
        with self.__MODELS_LOCK:
            if self.__MODELS_LOADED:
                return

            # Load tokenizer
            self.__TOKENIZER = AutoTokenizer.from_pretrained(self.model_name)
            self.__TOKENIZER.pad_token = self.__TOKENIZER.eos_token
            # Load model
            self.__MODEL = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", **kwargs
            )
            # Load LoRA adapter (if requested)
            if kwargs.get("adapter_name", None) is not None:
                self.__MODEL = PeftModel.from_pretrained(
                    self.__MODEL, kwargs.get("adapter_name")
                )
                self.__MODEL.merge_and_unload()
            self.__MODEL.eval()
            self.__MODELS_LOADED = True

    def __format_prompt(self, prompt: str) -> str:
        return f"<s>[INST] {prompt} [\\INST]"

    def __chunk_list(self, lst: List[str], n: int) -> Generator[List[str]]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _generate_impl(self, chunk: List[str]) -> Any:
        result = []
        for batch in self.__chunk_list(chunk, self.batch_size):
            batch_result = self._generate_batch(batch)
            result.extend(batch_result)
        return result

    def _generate_batch(self, batch: List[str]) -> Any:
        formatted_prompts = [self.__format_prompt(p) for p in batch]

        inputs = self.__TOKENIZER(formatted_prompts, return_tensors="pt")

        with torch.no_grad():
            generated_ids = self.__MODEL.generate(
                **inputs,
                max_length=self.generate_settings.max_length,
                num_beams=self.generate_settings.num_beams,
                num_return_sequences=self.generate_settings.num_return_sequences,
                early_stopping=True,
                do_sample=self.generate_settings.do_sample,
                temperature=self.generate_settings.temperature,
            )

        responses = self.__TOKENIZER.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        responses = [
            r.split("[\\INST]")[1] if "[\\INST]" in r else None for r in responses
        ]

        return responses
