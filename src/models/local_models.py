from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

from .api_models import _STRUCTVIZ_SYSTEM_PROMPT
from .model_wrapper import BaseModel
from .response_parser import parse_answer


def _resolve_torch_dtype(dtype: str) -> Any:
    """Map string dtype names to torch dtypes."""
    torch_module = importlib.import_module("torch")
    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Expected one of {sorted(dtype_map)}"
        )
    return dtype_map[dtype]


def _prepare_device(device: str) -> str:
    """Resolve requested device with graceful CPU fallback."""
    torch_module = importlib.import_module("torch")
    if device.startswith("cuda") and not torch_module.cuda.is_available():
        return "cpu"
    return device


@dataclass(slots=True)
class LlavaModel(BaseModel):
    """Wrapper for local LLaVA-family models."""

    checkpoint: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    device: str = "cuda"
    dtype: str = "float16"
    use_vllm: bool = False
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = "Answer concisely. Give only the final answer."
    _pipe: Any = None
    _vllm_engine: Any = None

    def _load(self) -> None:
        """Lazily initialize model backend."""
        resolved_device = _prepare_device(self.device)
        torch_dtype = _resolve_torch_dtype(self.dtype)

        if self.use_vllm:
            vllm_module = importlib.import_module("vllm")
            self._vllm_engine = vllm_module.LLM(
                model=self.checkpoint,
                dtype=self.dtype,
                tensor_parallel_size=1,
            )
            return

        device_arg: int | str = 0 if resolved_device.startswith("cuda") else "cpu"
        transformers_module = importlib.import_module("transformers")
        self._pipe = transformers_module.pipeline(
            task="image-text-to-text",
            model=self.checkpoint,
            device=device_arg,
            torch_dtype=torch_dtype,
        )

    def answer(self, question: str, image: object, metadata: dict[str, Any]) -> str:
        """Answer one question with local LLaVA inference."""
        if self._pipe is None and self._vllm_engine is None:
            self._load()

        prompt = f"{self.system_prompt}\nQuestion: {question}"
        task = str(metadata.get("task", "generic"))

        try:
            if self.use_vllm and self._vllm_engine is not None:
                prompt = f"{self.system_prompt}\nQuestion: {question}"
                vllm_module = importlib.import_module("vllm")
                sampling = vllm_module.SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                outputs = self._vllm_engine.generate(
                    [
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": image},
                        }
                    ],
                    sampling,
                )
                raw_response = outputs[0].outputs[0].text
                return parse_answer(raw_response=raw_response, task=task)

            if self._pipe is None:
                raise RuntimeError("LLaVA pipeline failed to initialize")

            generation_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0.0,
                "temperature": self.temperature,
            }
            attempts: list[dict[str, Any]] = [
                {"text": prompt, "images": image},
                {"text": prompt, "image": image},
            ]
            payload_attempts: list[dict[str, Any]] = [
                {"text": prompt, "images": [image]},
                {"text": prompt, "image": image},
            ]

            last_error: Exception | None = None
            generated: Any = None
            for kwargs in attempts:
                try:
                    generated = self._pipe(**kwargs, **generation_kwargs)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc

            if generated is None:
                for payload in payload_attempts:
                    try:
                        generated = self._pipe(payload, **generation_kwargs)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc

            if generated is None:
                raise RuntimeError(f"LLaVA inference failed: {last_error}")

            first_item = generated[0] if isinstance(generated, list) else generated
            raw_response = str(first_item.get("generated_text", ""))
            return parse_answer(raw_response=raw_response, task=task)
        finally:
            torch_module = importlib.import_module("torch")
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()


@dataclass(slots=True)
class QwenVLModel(BaseModel):
    """Wrapper for local Qwen-VL models."""

    checkpoint: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_vllm: bool = False
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = _STRUCTVIZ_SYSTEM_PROMPT
    _processor: Any = None
    _model: Any = None
    _vllm_engine: Any = None

    def _load(self) -> None:
        """Lazily initialize model backend."""
        if self.use_vllm:
            vllm_module = importlib.import_module("vllm")
            transformers_module = importlib.import_module("transformers")
            # Processor is needed for vLLM path too, to apply the Qwen-VL
            # chat template that adds <|vision_start|><|image_pad|><|vision_end|>
            # markers around the image placeholder.
            self._processor = transformers_module.AutoProcessor.from_pretrained(
                self.checkpoint
            )
            self._vllm_engine = vllm_module.LLM(
                model=self.checkpoint,
                dtype=self.dtype,
                tensor_parallel_size=1,
                # Cap context length to fit large vision models on a single
                # 80GB GPU (default would request the full 128k context which
                # OOMs at 32B/72B scale).
                max_model_len=4096,
                gpu_memory_utilization=0.92,
                limit_mm_per_prompt={"image": 1},
                enforce_eager=False,
                trust_remote_code=True,
            )
            return

        transformers_module = importlib.import_module("transformers")
        torch_module = importlib.import_module("torch")
        self._processor = transformers_module.AutoProcessor.from_pretrained(
            self.checkpoint
        )
        self._model = (
            transformers_module.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.checkpoint,
                torch_dtype=torch_module.bfloat16,
                device_map="auto",
            )
        )
        model = self._model
        if model is None:
            raise RuntimeError("Qwen-VL model initialization returned None")
        model.eval()

    def build_messages(self, question: str, image: object | None) -> list[dict[str, Any]]:
        """Build the chat messages for one query.

        When ``image`` is None the user turn contains only the question text
        (text-only / no-image baseline); otherwise the image precedes the text,
        exactly as in the original v1 evaluation path.
        """
        user_content: list[dict[str, Any]] = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": question})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_prompt_text(self, question: str, has_image: bool = True) -> str:
        """Return the chat-templated prompt string (image content independent)."""
        if self._model is None and self._vllm_engine is None:
            self._load()
        if self._processor is None:
            raise RuntimeError("Qwen-VL processor failed to initialize")
        messages = self.build_messages(question, "<image>" if has_image else None)
        if has_image:
            # The template only needs the placeholder type, not the payload.
            messages[1]["content"][0] = {"type": "image"}
        return str(
            self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _generate_raw(self, question: str, image: object | None) -> str:
        """Run greedy generation and return the raw decoded model text."""
        if self._model is None and self._vllm_engine is None:
            self._load()

        if self.use_vllm and self._vllm_engine is not None:
            # Build chat-templated prompt with <|vision_start|>... markers
            # via the processor (this is the correct format for Qwen2.5-VL
            # under vLLM; raw "prompt" without markers leads to malformed
            # multimodal alignment).
            if self._processor is None:
                raise RuntimeError("Qwen-VL processor required for vLLM path")
            templated_prompt = self.build_prompt_text(question, has_image=image is not None)
            vllm_module = importlib.import_module("vllm")
            sampling = vllm_module.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            request: dict[str, Any] = {"prompt": templated_prompt}
            if image is not None:
                request["multi_modal_data"] = {"image": image}
            outputs = self._vllm_engine.generate([request], sampling)
            return str(outputs[0].outputs[0].text)

        if self._processor is None or self._model is None:
            raise RuntimeError("Qwen-VL model failed to initialize")

        qwen_utils_module = importlib.import_module("qwen_vl_utils")
        messages = self.build_messages(question, image)
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = qwen_utils_module.process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=0.0,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return str(
            self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

    def answer_with_raw(
        self,
        question: str,
        image: object | None,
        metadata: dict[str, Any],
    ) -> tuple[str, str]:
        """Answer one question and return ``(raw_response, parsed_answer)``.

        Identical inference to :meth:`answer`; additionally exposes the raw
        model text before ``parse_answer`` post-processing. ``image=None``
        runs the text-only chat path (no vision tokens).
        """
        torch_module = importlib.import_module("torch")
        task = str(metadata.get("task", "generic"))
        try:
            raw_response = self._generate_raw(question, image)
            return raw_response, parse_answer(raw_response=raw_response, task=task)
        finally:
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

    def answer(self, question: str, image: object, metadata: dict[str, Any]) -> str:
        """Answer one question with local Qwen-VL inference."""
        return self.answer_with_raw(question=question, image=image, metadata=metadata)[1]


@dataclass(slots=True)
class InternVLModel(BaseModel):
    """Wrapper for local InternVL models."""

    checkpoint: str = "OpenGVLab/InternVL2_5-8B"
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_vllm: bool = False
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = "Answer concisely. Give only the final answer."
    _processor: Any = None
    _tokenizer: Any = None
    _model: Any = None
    _vllm_engine: Any = None

    def _load(self) -> None:
        """Lazily initialize model backend."""
        resolved_device = _prepare_device(self.device)

        if self.use_vllm:
            vllm_module = importlib.import_module("vllm")
            self._vllm_engine = vllm_module.LLM(
                model=self.checkpoint,
                dtype=self.dtype,
                tensor_parallel_size=1,
            )
            return

        torch_dtype = _resolve_torch_dtype(self.dtype)
        transformers_module = importlib.import_module("transformers")
        self._processor = transformers_module.AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        self._tokenizer = transformers_module.AutoTokenizer.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )
        quantization_config = None
        try:
            quantization_config = transformers_module.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        except Exception:
            quantization_config = None

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_buffers"] = True

        self._model = transformers_module.AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            **model_kwargs,
        )
        model = self._model
        if model is None:
            raise RuntimeError("InternVL model initialization returned None")
        language_model = getattr(model, "language_model", None)
        if language_model is not None:
            generation_utils = importlib.import_module("transformers.generation.utils")
            generation_mixin = generation_utils.GenerationMixin
            required_methods = [
                "generate",
                "_extract_generation_mode_kwargs",
                "_prepare_generation_config",
                "_prepare_model_inputs",
                "_maybe_initialize_input_ids_for_generation",
                "_expand_inputs_for_generation",
                "_prepare_special_tokens",
                "_prepare_position_ids_for_generation",
                "_validate_model_kwargs",
                "_validate_generation_mode",
                "_update_model_kwargs_for_generation",
                "_get_deprecated_gen_repo",
                "_sample",
            ]
            for method_name in required_methods:
                if hasattr(generation_mixin, method_name) and not hasattr(
                    language_model.__class__, method_name
                ):
                    setattr(
                        language_model.__class__,
                        method_name,
                        getattr(generation_mixin, method_name),
                    )

            if hasattr(generation_mixin, "_expand_inputs_for_generation"):

                def _expand_inputs_compat(self: Any, *args: Any, **kwargs: Any) -> Any:
                    torch_module = importlib.import_module("torch")
                    local_kwargs = dict(kwargs)

                    input_ids = local_kwargs.pop("input_ids", None)
                    if input_ids is None and args:
                        input_ids = args[0]

                    expand_size = int(local_kwargs.pop("expand_size", 1))
                    _ = bool(local_kwargs.pop("is_encoder_decoder", False))

                    if input_ids is not None and isinstance(
                        input_ids, torch_module.Tensor
                    ):
                        base_batch = int(input_ids.shape[0])
                        expanded_input_ids = input_ids.repeat_interleave(
                            expand_size, dim=0
                        )
                    else:
                        base_batch = 0
                        expanded_input_ids = input_ids

                    expanded_kwargs: dict[str, Any] = {}
                    for key, value in local_kwargs.items():
                        if (
                            isinstance(value, torch_module.Tensor)
                            and base_batch > 0
                            and value.dim() > 0
                            and int(value.shape[0]) == base_batch
                        ):
                            expanded_kwargs[key] = value.repeat_interleave(
                                expand_size, dim=0
                            )
                        else:
                            expanded_kwargs[key] = value

                    return expanded_input_ids, expanded_kwargs

                language_model.__class__._expand_inputs_for_generation = (
                    _expand_inputs_compat
                )
            if not hasattr(language_model, "generation_config"):
                language_model.generation_config = getattr(
                    model,
                    "generation_config",
                    None,
                )
        if hasattr(model, "to") and not hasattr(model, "hf_device_map"):
            model.to(resolved_device)
        model.eval()

    def answer(self, question: str, image: object, metadata: dict[str, Any]) -> str:
        """Answer one question with local InternVL inference."""
        torch_module = importlib.import_module("torch")

        if self._model is None and self._vllm_engine is None:
            self._load()

        task = str(metadata.get("task", "generic"))
        prompt = f"{self.system_prompt}\nQuestion: {question}"

        try:
            if self.use_vllm and self._vllm_engine is not None:
                vllm_module = importlib.import_module("vllm")
                sampling = vllm_module.SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                outputs = self._vllm_engine.generate(
                    [
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": image},
                        }
                    ],
                    sampling,
                )
                raw_response = outputs[0].outputs[0].text
                return parse_answer(raw_response=raw_response, task=task)

            if self._processor is None or self._model is None:
                raise RuntimeError("InternVL model failed to initialize")

            if hasattr(self._model, "chat") and self._tokenizer is not None:
                torchvision_transforms = importlib.import_module(
                    "torchvision.transforms"
                )
                transform = torchvision_transforms.Compose(
                    [
                        torchvision_transforms.Resize((448, 448)),
                        torchvision_transforms.ToTensor(),
                        torchvision_transforms.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                        ),
                    ]
                )
                model_dtype = _resolve_torch_dtype(self.dtype)
                pixel_values = (
                    transform(image)
                    .unsqueeze(0)
                    .to(
                        device=self._model.device,
                        dtype=model_dtype,
                    )
                )
                generation_config = {
                    "max_new_tokens": self.max_tokens,
                    "do_sample": self.temperature > 0.0,
                    "temperature": self.temperature,
                }
                raw_output = self._model.chat(
                    self._tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                )
                if isinstance(raw_output, tuple):
                    raw_response = str(raw_output[0])
                else:
                    raw_response = str(raw_output)
            else:
                inputs = self._processor(
                    text=[prompt],
                    images=[image],
                    return_tensors="pt",
                )
                tensor_inputs = {
                    key: value.to(self._model.device)
                    for key, value in inputs.items()
                    if isinstance(value, torch_module.Tensor)
                }
                generated_ids = self._model.generate(
                    **tensor_inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0.0,
                    temperature=self.temperature,
                )
                prompt_len = tensor_inputs["input_ids"].shape[-1]
                completion_ids = generated_ids[:, prompt_len:]
                raw_response = self._processor.batch_decode(
                    completion_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
            return parse_answer(raw_response=raw_response, task=task)
        finally:
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
