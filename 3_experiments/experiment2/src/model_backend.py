"""Text-only inference backend for causal and multimodal instruction models."""

from typing import List
import time

import torch


class TextGenerationBackend:
    QWEN35_PREFIX = "Qwen/Qwen3.5"
    GEMMA4_PREFIX = "google/gemma-4"
    MULTIMODAL_PREFIXES = (
        QWEN35_PREFIX,
        GEMMA4_PREFIX,
    )
    MISTRAL3_PREFIX = "mistralai/Ministral-3"
    MISTRAL_NO_REPEAT_NGRAM_SIZE = 8

    def __init__(self, model_name: str, dtype):
        self.model_name = model_name
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load(self):
        device_map = "auto" if torch.cuda.is_available() else None

        if self.model_name.startswith(self.QWEN35_PREFIX):
            from transformers import AutoModelForMultimodalLM, AutoTokenizer

            # Qwen 3.5 has a standalone tokenizer and chat template. Using it
            # avoids initializing image/video processors for text-only tasks.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        elif self.model_name.startswith(self.GEMMA4_PREFIX):
            from transformers import AutoModelForMultimodalLM, AutoTokenizer

            # Keep the checkpoint's multimodal model class so that its
            # ``model.language_model.*`` weights are mapped correctly. The
            # standalone tokenizer is sufficient for this text-only experiment
            # and avoids constructing Gemma's optional video processor.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        elif self.model_name.startswith(self.MISTRAL3_PREFIX):
            from transformers import (
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )

            self.tokenizer = MistralCommonBackend.from_pretrained(self.model_name)
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=device_map,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

        self._configure_tokenizer()
        return self.model, self.tokenizer, self

    def generate_texts(
        self, prompts: List[str], max_new_tokens: int, temperature: float
    ) -> List[str]:
        if self.model_name.startswith(self.MISTRAL3_PREFIX):
            return self._generate_mistral_texts(
                prompts, max_new_tokens, temperature
            )

        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        encoder = self.processor or self.tokenizer
        if self.processor is not None:
            inputs = encoder(
                text=rendered_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        else:
            inputs = encoder(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
        inputs = inputs.to(self.model.device)
        input_width = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_new_tokens, temperature),
            )

        return [
            self._decode(output[input_width:])
            for output in outputs
        ]

    def _generate_mistral_texts(
        self, prompts: List[str], max_new_tokens: int, temperature: float
    ) -> List[str]:
        conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        inputs = self.tokenizer.apply_chat_template(
            conversations,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
        )
        inputs = {
            key: value.to(self.model.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        input_width = inputs["input_ids"].shape[1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            input_lengths = [input_width] * len(prompts)
        else:
            input_lengths = attention_mask.sum(dim=1).tolist()

        generation_kwargs = self._generation_kwargs(
            max_new_tokens, temperature
        )
        generation_kwargs["no_repeat_ngram_size"] = (
            self.MISTRAL_NO_REPEAT_NGRAM_SIZE
        )
        generation_started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        elapsed_seconds = time.perf_counter() - generation_started
        generated_ids = [output[input_width:] for output in outputs]
        generated_lengths = [
            self._generated_length(token_ids) for token_ids in generated_ids
        ]
        decoded_texts = [
            self.tokenizer.decode(token_ids[:generated_length]).strip()
            for token_ids, generated_length in zip(
                generated_ids, generated_lengths
            )
        ]
        generated_tokens = sum(generated_lengths)
        tokens_per_second = (
            generated_tokens / elapsed_seconds
            if elapsed_seconds > 0
            else float("inf")
        )

        print(
            "\n" + "=" * 80
            + "\nMISTRAL BATCH GENERATION DIAGNOSTICS"
            + f"\nBatch size: {len(prompts)}"
            + f"\nTotal generated tokens: {generated_tokens}"
            + f"\nElapsed: {elapsed_seconds:.2f} s"
            + f"\nBatch generation speed: {tokens_per_second:.2f} tokens/s"
            + "\n" + "=" * 80,
            flush=True,
        )
        for index, (input_length, generated_length, decoded_text) in enumerate(
            zip(input_lengths, generated_lengths, decoded_texts), start=1
        ):
            print(
                "\n" + "-" * 80
                + f"\nMISTRAL OUTPUT {index}/{len(prompts)}"
                + f"\nInput tokens: {int(input_length)}"
                + f"\nGenerated tokens: {generated_length}/{max_new_tokens}"
                + f"\nReached token limit: {generated_length >= max_new_tokens}"
                + f"\nOUTPUT:\n{decoded_text}"
                + "\n" + "-" * 80,
                flush=True,
            )
        return decoded_texts

    def _generated_length(self, generated_ids: torch.Tensor) -> int:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
            if eos_token_id is not None
            else set()
        )
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        for index, token_id in enumerate(generated_ids.tolist()):
            if token_id in eos_token_ids:
                return index + 1
            if pad_token_id is not None and token_id == pad_token_id:
                return index
        return int(generated_ids.numel())

    def _render_prompt(self, prompt: str) -> str:
        if self.processor is not None:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            template_engine = self.processor
        else:
            messages = [{"role": "user", "content": prompt}]
            template_engine = self.tokenizer

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.model_name.startswith(self.MULTIMODAL_PREFIXES):
            kwargs["enable_thinking"] = False
        try:
            return template_engine.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return template_engine.apply_chat_template(messages, **kwargs)

    def _generation_kwargs(
        self, max_new_tokens: int, temperature: float
    ) -> dict:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id
        return kwargs

    def _decode(self, generated_ids: torch.Tensor) -> str:
        decoder = self.processor or self.tokenizer
        text = decoder.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        if self.processor is not None and hasattr(self.processor, "parse_response"):
            try:
                parsed = self.processor.parse_response(text)
                if isinstance(parsed, str):
                    return parsed.strip()
            except Exception:
                pass
        return text

    def _configure_tokenizer(self) -> None:
        if self.tokenizer is None:
            return
        if (
            getattr(self.tokenizer, "pad_token", None) is None
            and getattr(self.tokenizer, "eos_token", None) is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
