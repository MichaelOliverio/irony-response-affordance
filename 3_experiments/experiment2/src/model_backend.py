"""Text-only inference backend for causal and multimodal instruction models."""

from typing import List

import torch


class TextGenerationBackend:
    QWEN35_PREFIX = "Qwen/Qwen3.5"
    GEMMA4_PREFIX = "google/gemma-4"
    MULTIMODAL_PREFIXES = (
        QWEN35_PREFIX,
        GEMMA4_PREFIX,
    )
    MISTRAL3_PREFIX = "mistralai/Ministral-3"

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
            return [
                self._generate_mistral_text(prompt, max_new_tokens, temperature)
                for prompt in prompts
            ]

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

    def _generate_mistral_text(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        inputs = {
            key: value.to(self.model.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        input_width = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_new_tokens, temperature),
            )[0]
        return self.tokenizer.decode(output[input_width:]).strip()

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
