import os
import time
import json
import torch
import torchaudio
import spacy

from typing import List, Tuple
from spacy.language import Language
from fireredtts2.codec import RedCodecInfer
from fireredtts2.llm import load_llm_model, load_custom_tokenizer
from fireredtts2.llm.utils import Segment
from fireredtts2.utils.device import resolve_device
from fireredtts2.utils.spliter import (
    clean_text,
    split_text,
    process_text_list,
    utf_8_len,
)
from tqdm import tqdm


class FireRedTTS2:
    def __init__(self, pretrained_dir, gen_type, device):
        resolved_device = resolve_device(device)
        self.device = resolved_device

        def _summarize_module(module: torch.nn.Module, name: str):
            return
            try:
                p_dtypes = {}
                p_devices = {}
                for p in module.parameters():
                    p_dtypes[str(p.dtype)] = p_dtypes.get(str(p.dtype), 0) + 1
                    p_devices[str(p.device)] = p_devices.get(str(p.device), 0) + 1
                b_dtypes = {}
                for b in module.buffers():
                    if hasattr(b, "dtype"):
                        b_dtypes[str(b.dtype)] = b_dtypes.get(str(b.dtype), 0) + 1

                print(
                    f"[DEBUG] {name} param dtypes: {sorted(p_dtypes.items())} | devices: {sorted(p_devices.items())}"
                )
                if b_dtypes:
                    print(
                        f"[DEBUG] {name} buffer dtypes: {sorted(b_dtypes.items())}"
                    )
            except Exception as e:
                print(f"[DEBUG] Failed to summarize {name}: {e}")

        assert os.path.exists(pretrained_dir)
        assert gen_type in ["monologue", "dialogue"]
        llm_config_path = os.path.join(pretrained_dir, "config_llm.json")
        if gen_type == "monologue":
            llm_ckpt_path = os.path.join(pretrained_dir, "llm_pretrain.pt")
            # llm_ckpt_path = os.path.join(pretrained_dir, "llm_posttrain.pt")
        else:
            llm_ckpt_path = os.path.join(pretrained_dir, "llm_posttrain.pt")
        codec_config_path = os.path.join(pretrained_dir, "config_codec.json")
        codec_ckpt_path = os.path.join(pretrained_dir, "codec.pt")
        pretrained_qwen_path = os.path.join(pretrained_dir, "Qwen2.5-1.5B")

        # check
        assert os.path.exists(llm_config_path)
        assert os.path.exists(llm_ckpt_path)
        assert os.path.exists(codec_config_path)
        assert os.path.exists(codec_ckpt_path)
        assert os.path.exists(pretrained_qwen_path)

        # ==== Load Torch LLM ====
        llm_config = json.load(open(llm_config_path))
        self._model = load_llm_model(
            configs=llm_config,
            checkpoint_path=llm_ckpt_path,
            device=self.device,
        )
        # Prefer FP16 on MPS; BF16 on CUDA
        if self.device.type == "mps":
            try:
                self._model = self._model.to(dtype=torch.float16)
                print("[INFO] LLM cast to float16 on MPS")
            except Exception as e:
                print(f"[WARN] Could not cast LLM to float16 on MPS: {e}")
        elif self.device.type == "cuda":
            try:
                self._model = self._model.to(dtype=torch.bfloat16)
                print("[INFO] LLM cast to bfloat16 on CUDA")
            except Exception as e:
                print(f"[WARN] Could not cast LLM to bfloat16 on CUDA: {e}")

        self._model.eval()
        self._model.setup_caches(1)
        print("[INFO] LLM Loaded...")
        _summarize_module(self._model, "LLM")

        # ==== Load Qwen2.5 Text Tokenizer ====
        self._text_tokenizer = load_custom_tokenizer(pretrained_qwen_path)
        print("[INFO] Text Tokenizer Loaded...")

        # ==== Load Torch Audio Tokenizer ====
        torch_codec = RedCodecInfer.from_pretrained(codec_config_path, codec_ckpt_path)
        torch_codec.eval()
        self._audio_tokenizer = torch_codec.to(self.device)
        print("[INFO] Codec Loaded...")
        _summarize_module(self._audio_tokenizer, "Codec")

        self.sample_rate = 16000
        self.max_seq_len = 3100
        self._sentence_splitter: Language | None = None

    def load_prompt_audio(self, audio_path) -> torch.Tensor:
        audio, audio_sr = torchaudio.load(audio_path)
        # Audio must be single channel
        if audio.shape[0] > 1:
            audio = audio[0, :].unsqueeze(0)
        audio16k = torchaudio.functional.resample(audio, audio_sr, 16000)
        return audio16k

    def prepare_prompt(self, text, speaker, audio_path) -> Segment:
        audio_tensor = self.load_prompt_audio(audio_path)
        return Segment(text=text, speaker=speaker, audio=audio_tensor)

    def _ensure_sentence_splitter(self) -> Language:
        if self._sentence_splitter is None:
            sentence_splitter = spacy.blank("xx")
            if "sentencizer" not in sentence_splitter.pipe_names:
                sentence_splitter.add_pipe("sentencizer")
            self._sentence_splitter = sentence_splitter
        return self._sentence_splitter

    def _split_sentences(self, text: str) -> List[str]:
        splitter = self._ensure_sentence_splitter()
        doc = splitter(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences if sentences else [text.strip()]

    def _prepare_monologue_segments(
        self, text: str, use_sentence_split: bool, max_length: int = 400
    ) -> List[str]:
        if use_sentence_split:
            raw_segments = self._split_sentences(text)
            prepared_segments: List[str] = []
            for segment in raw_segments:
                cleaned_segment = clean_text(segment)
                if not cleaned_segment:
                    continue
                if utf_8_len(cleaned_segment) > max_length:
                    prepared_segments.extend(
                        split_text(text=cleaned_segment, length=max_length)
                    )
                else:
                    prepared_segments.append(cleaned_segment)
            if prepared_segments:
                return prepared_segments
            text = clean_text(text)
            return split_text(text=text, length=max_length)

        cleaned_text = clean_text(text=text)
        return split_text(text=cleaned_text, length=max_length)

    def _tokenize_text_segment(
        self, text: str, speaker: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text = speaker + "<|text_start|>" + text + "<|text_end|>"
        text_tokens = self._text_tokenizer.encode(text)
        text_frame = torch.zeros(len(text_tokens), 17).long()
        text_frame_mask = torch.zeros(len(text_tokens), 17).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio_length = torch.tensor([audio.shape[1]], dtype=torch.long)
        audio_tokens, token_length = self._audio_tokenizer.encode(
            audio.to(self.device),
            audio_length.to(self.device),
            batch_size=48,
        )

        audio_tokens = audio_tokens.squeeze(0)
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 17).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 17).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len,17), (seq_len, 17)
        """
        text_tokens, text_masks = self._tokenize_text_segment(
            segment.text, segment.speaker
        )
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat(
            [text_masks, audio_masks], dim=0
        )

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: str,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 20,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
            text, speaker
        )
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = (
            torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        )

        max_seq_len = 3100
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
            )
            # eos
            if torch.all(sample == 0):
                break

            samples.append(sample)

            curr_tokens = torch.cat(
                [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [
                    torch.ones_like(sample).bool(),
                    torch.zeros(1, 1).bool().to(self.device),
                ],
                dim=1,
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = (
            self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
            .squeeze(0)
            .squeeze(0)
        )

        return audio

    def generate_single(
        self, context: List[Segment], temperature: float = 0.9, topk: int = 20
    ):
        self._model.reset_caches()
        max_generation_len = 400
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        prompt_tokens = prompt_tokens[:-3, :]
        prompt_tokens_mask = prompt_tokens_mask[:-3, :]

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = (
            torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        )

        num_token = 0
        start_time = time.time()
        for _ in range(max_generation_len):
            sample = self._model.generate_frame(
                curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
            )
            # eos
            if torch.all(sample == 0):
                break

            samples.append(sample)

            curr_tokens = torch.cat(
                [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [
                    torch.ones_like(sample).bool(),
                    torch.zeros(1, 1).bool().to(self.device),
                ],
                dim=1,
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
            num_token += 1
            if num_token == 2:
                end_time = time.time()
                duration = end_time - start_time
                print("---first pack duration:", duration)

        # If the model immediately produced EOS (no samples), avoid crashing here.
        if len(samples) == 0:
            print(
                "[WARN] generate_single: immediate EOS (no samples). Returning empty token sequence."
            )
            return torch.zeros(
                (1, self._model.config.audio_num_codebooks, 0),
                dtype=torch.long,
                device=self.device,
            )

        gen_tokens = torch.stack(samples).permute(1, 2, 0)

        return gen_tokens

    # @torch.inference_mode()
    # def generate_stream(
    #     self,
    #     text: str,
    #     speaker: str,
    #     context: List[Segment],
    #     max_audio_length_ms: float = 90_000,
    #     temperature: float = 0.9,
    #     topk: int = 50,
    # ):
    #     self._model.reset_caches()

    #     max_generation_len = int(max_audio_length_ms / 80)
    #     tokens, tokens_mask = [], []
    #     for segment in context:
    #         segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
    #         tokens.append(segment_tokens)
    #         tokens_mask.append(segment_tokens_mask)

    #     gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
    #         text, speaker
    #     )
    #     tokens.append(gen_segment_tokens)
    #     tokens_mask.append(gen_segment_tokens_mask)

    #     prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
    #     prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

    #     samples = []
    #     curr_tokens = prompt_tokens.unsqueeze(0)
    #     curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
    #     curr_pos = (
    #         torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
    #     )

    #     max_seq_len = 3100
    #     max_context_len = max_seq_len - max_generation_len
    #     if curr_tokens.size(1) >= max_context_len:
    #         raise ValueError(
    #             f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
    #         )

    #     # codec cache
    #     codec_cache = {}
    #     prev_sample = None

    #     for _ in range(max_generation_len):
    #         sample = self._model.generate_frame(
    #             curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
    #         )
    #         # eos
    #         if torch.all(sample == 0):
    #             break

    #         # decode one token
    #         if prev_sample is None:
    #             prev_sample = sample  # sample: (b, nq)
    #         else:
    #             audio_chunk, codec_cache = self._audio_tokenizer.decode_one_token(
    #                 prev_sample.unsqueeze(-1),
    #                 codec_cache,
    #                 last_token=False,
    #             )
    #             yield audio_chunk.squeeze(0)
    #             prev_sample = sample
    #         samples.append(sample)  # sample: (b, nq)

    #         curr_tokens = torch.cat(
    #             [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
    #         ).unsqueeze(1)
    #         curr_tokens_mask = torch.cat(
    #             [
    #                 torch.ones_like(sample).bool(),
    #                 torch.zeros(1, 1).bool().to(self.device),
    #             ],
    #             dim=1,
    #         ).unsqueeze(1)
    #         curr_pos = curr_pos[:, -1:] + 1

    #     audio_chunk, codec_cache = self._audio_tokenizer.decode_one_token(
    #         prev_sample.unsqueeze(-1),
    #         codec_cache,
    #         last_token=True,
    #     )
    #     yield audio_chunk.squeeze(0)

    @torch.inference_mode()
    def generate_dialogue(
        self,
        text_list,
        prompt_wav_list=None,
        prompt_text_list=None,
        temperature=0.9,
        topk=20,
    ):
        all_generated_segments = []
        all_storage_segments = []
        prompt_segments = []
        text_list = process_text_list(text_list=text_list)
        if prompt_wav_list is not None:
            assert len(prompt_wav_list) == len(prompt_text_list)
            # Prepare prompts
            for i in range(len(prompt_wav_list)):
                prompt_wav = prompt_wav_list[i]
                prompt_text = prompt_text_list[i]
                speaker = prompt_text[:4]
                assert speaker in ["[S1]", "[S2]", "[S3]", "[S4]"]
                prompt_segments.append(
                    self.prepare_prompt(
                        text=prompt_text, speaker=speaker, audio_path=prompt_wav
                    )
                )

        for text in tqdm(text_list):
            speaker = text[:4]
            text = text[4:]
            # print("---speaker:", speaker)
            # print("---text:", text)
            assert speaker in ["[S1]", "[S2]", "[S3]", "[S4]"]

            audio_tensor = self.generate(
                text=text,
                speaker=speaker,
                context=prompt_segments + all_generated_segments,
                max_audio_length_ms=30_000,
                temperature=temperature,
                topk=topk,
            )

            # 做上下文管理的时候需要将audio 转到16k
            # Resample on CPU to avoid MPS conv1d limitations
            audio_16k = torchaudio.functional.resample(
                audio_tensor.detach().cpu().unsqueeze(0), 24000, 16000
            )
            all_generated_segments.append(
                Segment(text=text, speaker=speaker, audio=audio_16k)
            )

            all_storage_segments.append(
                Segment(text=text, speaker=speaker, audio=audio_tensor.unsqueeze(0))
            )

        # Concatenate all generations
        all_audio = torch.cat([seg.audio for seg in all_storage_segments], dim=1)
        all_audio = all_audio.cpu()
        return all_audio

    @torch.inference_mode()
    def generate_monologue(
        self,
        text,
        prompt_wav=None,
        prompt_text=None,
        temperature=0.75,
        topk=20,
        sentence_split: bool = False,
    ):
        # step1. construct context
        if prompt_wav is not None:
            assert os.path.exists(prompt_wav)
            assert prompt_text is not None

            all_generated_segments = []
            all_storage_segments = []
            prompt_segments = []
            prompt_text = clean_text(text=prompt_text)
            text_list = self._prepare_monologue_segments(
                text=text, use_sentence_split=sentence_split
            )

            audio_list = []
            history_segments: List[Segment] = []
            history_limit = 1 if sentence_split else 0
            print(
                f"[DEBUG] Using sentence_split={sentence_split} with {len(text_list)} segments"
            )
            for parsed_text in text_list:
                parsed_text = clean_text(text=parsed_text)
                prompt_prefix = prompt_text[:-1] if prompt_text else ""
                input_text = (
                    (prompt_prefix + ", ") if prompt_prefix else ""
                ) + parsed_text
                print(
                    f"[DEBUG] Prompted generation input: '{input_text}'"
                )
                prompt_a = self.prepare_prompt(
                    text=input_text, speaker="[S1]", audio_path=prompt_wav
                )

                context = [prompt_a]
                if history_limit > 0 and history_segments:
                    context = context + history_segments[-history_limit:]

                while True:
                    gen_tokens = self.generate_single(
                        context=context, temperature=temperature, topk=topk
                    )
                    if gen_tokens.shape[2] > 18:
                        break
                    # else:
                    #     print("生成结果小于1s,重新跑")

                gen_tokens = gen_tokens[:, :, 2:]  # cut leading silence
                audio = self._audio_tokenizer.decode(gen_tokens).squeeze(0).squeeze(0)
                audio_list.append(audio.unsqueeze(0))

                if history_limit > 0:
                    # Resample on CPU to avoid MPS conv1d limitations
                    history_audio = torchaudio.functional.resample(
                        audio.detach().cpu().unsqueeze(0), 24000, 16000
                    )
                    history_segments.append(
                        Segment(text=parsed_text, speaker="[S1]", audio=history_audio)
                    )
                    if len(history_segments) > history_limit:
                        history_segments = history_segments[-history_limit:]

            all_audio = torch.cat(tensors=audio_list, dim=1)

            return all_audio

        else:
            # random speaker
            if not sentence_split:
                cleaned_text = clean_text(text=text.strip())
                print("[DEBUG] Using sentence_split=False; invoking single pass generation")
                audio_tensor = self.generate(
                    text=cleaned_text,
                    speaker="[S1]",
                    context=[],
                    max_audio_length_ms=30_000,
                    temperature=temperature,
                    topk=topk,
                )
                return audio_tensor.unsqueeze(0)

            segments = self._prepare_monologue_segments(
                text=text.strip(), use_sentence_split=sentence_split
            )
            print(
                f"[DEBUG] Using sentence_split={sentence_split} with {len(segments)} segments"
            )

            if len(segments) == 1:
                audio_tensor = self.generate(
                    text=segments[0],
                    speaker="[S1]",
                    context=[],
                    max_audio_length_ms=30_000,
                    temperature=temperature,
                    topk=topk,
                )
                return audio_tensor.unsqueeze(0)

            stored_segments: List[torch.Tensor] = []

            for segment_text in segments:
                print(f"[DEBUG] Generating independent segment: '{segment_text}'")
                audio_tensor = self.generate(
                    text=segment_text,
                    speaker="[S1]",
                    context=[],
                    max_audio_length_ms=30_000,
                    temperature=temperature,
                    topk=topk,
                )
                stored_segments.append(audio_tensor.unsqueeze(0))

            concatenated = torch.cat(stored_segments, dim=1)
            return concatenated
