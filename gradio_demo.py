import gc
import re

import gradio as gr
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Literal, List, Tuple

import torch
from fireredtts2.fireredtts2 import FireRedTTS2
from fireredtts2.utils.device import resolve_device, empty_cache


# ================================================
#                   FireRedTTS2 Model
# ================================================
# Global model instance
model: FireRedTTS2 | None = None
model_gen_type: Literal["monologue", "dialogue"] | None = None
model_options: dict[str, str] = {}


def reset_model_cache():
    global model, model_gen_type
    if model is not None:
        model_device = getattr(model, "device", None)
        del model
        gc.collect()
        empty_cache(model_device)
    model = None
    model_gen_type = None


def initiate_model(
    pretrained_dir: str,
    gen_type: Literal["monologue", "dialogue"] = "dialogue",
    device: torch.device | str | None = None,
) -> FireRedTTS2:
    global model, model_gen_type, model_options

    stored_pretrained_dir = model_options.get("pretrained_dir")
    stored_device = model_options.get("device")
    requested_device = device if device is not None else stored_device
    resolved_device = resolve_device(requested_device)
    resolved_device_str = str(resolved_device)
    need_reload = (
        model is None
        or model_gen_type != gen_type
        or stored_pretrained_dir != pretrained_dir
        or stored_device != resolved_device_str
    )

    if need_reload:
        reset_model_cache()
        model = FireRedTTS2(
            pretrained_dir=pretrained_dir,
            gen_type=gen_type,
            device=resolved_device,
        )
        model_gen_type = gen_type

    model_options["pretrained_dir"] = pretrained_dir
    model_options["device"] = resolved_device_str
    return model


# ================================================
#                   Gradio
# ================================================

# i18n
_i18n_key2lang_dict = dict(
    # Title markdown
    title_md_desc=dict(
        en="FireRedTTS-2 🔥 Dialogue Generation",
        zh="FireRedTTS-2 🔥 对话生成",
    ),
    # Voice mode radio
    synthesis_mode_label=dict(
        en="Synthesis Mode",
        zh="合成模式",
    ),
    synthesis_mode_choice_dialogue=dict(
        en="Dialogue",
        zh="对话",
    ),
    synthesis_mode_choice_monologue=dict(
        en="Monologue",
        zh="独白",
    ),
    voice_mode_label=dict(
        en="Voice Mode",
        zh="音色模式",
    ),
    voice_model_choice1=dict(
        en="Voice Clone",
        zh="音色克隆",
    ),
    voice_model_choice2=dict(
        en="Random Voice",
        zh="随机音色",
    ),
    # Speaker1 Prompt
    spk1_prompt_audio_label=dict(
        en="Speaker 1 Prompt Audio",
        zh="说话人 1 参考语音",
    ),
    spk1_prompt_text_label=dict(
        en="Speaker 1 Prompt Text",
        zh="说话人 1 参考文本",
    ),
    spk1_prompt_text_placeholder=dict(
        en="[S1] text of speaker 1 prompt audio.",
        zh="[S1] 说话人 1 参考文本",
    ),
    # Speaker2 Prompt
    spk2_prompt_audio_label=dict(
        en="Speaker 2 Prompt Audio",
        zh="说话人 2 参考语音",
    ),
    spk2_prompt_text_label=dict(
        en="Speaker 2 Prompt Text",
        zh="说话人 2 参考文本",
    ),
    spk2_prompt_text_placeholder=dict(
        en="[S2] text of speaker 2 prompt audio.",
        zh="[S2] 说话人 2 参考文本",
    ),
    # Dialogue input textbox
    dialogue_text_input_label=dict(
        en="Dialogue Text Input",
        zh="对话文本输入",
    ),
    dialogue_text_input_placeholder=dict(
        en="[S1]text[S2]text[S1]text...",
        zh="[S1]文本[S2]文本[S1]文本...",
    ),
    monologue_text_input_label=dict(
        en="Monologue Text Input",
        zh="独白文本输入",
    ),
    monologue_text_input_placeholder=dict(
        en="Enter the text you want to synthesize.",
        zh="请输入需要合成的文本。",
    ),
    # Generate button
    generate_btn_label=dict(
        en="Generate Audio",
        zh="合成",
    ),
    # Generated audio
    generated_audio_label=dict(
        en="Generated Dialogue Audio",
        zh="合成的对话音频",
    ),
    generated_monologue_label=dict(
        en="Generated Monologue Audio",
        zh="合成的独白音频",
    ),
    sentence_split_label=dict(
        en="Split sentences with spaCy",
        zh="使用 spaCy 句子切分",
    ),
    # Warining1: invalid text for prompt
    warn_invalid_spk1_prompt_text=dict(
        en='Invalid speaker 1 prompt text, should strictly follow: "[S1]xxx"',
        zh='说话人 1 参考文本不合规，格式："[S1]xxx"',
    ),
    warn_invalid_spk2_prompt_text=dict(
        en='Invalid speaker 2 prompt text, should strictly follow: "[S2]xxx"',
        zh='说话人 2 参考文本不合规，格式："[S2]xxx"',
    ),
    # Warining2: invalid text for dialogue input
    warn_invalid_dialogue_text=dict(
        en='Invalid dialogue input text, should strictly follow: "[S1]xxx[S2]xxx..."',
        zh='对话文本输入不合规，格式："[S1]xxx[S2]xxx..."',
    ),
    warn_invalid_monologue_text=dict(
        en="Please enter non-empty monologue text.",
        zh="请输入有效的独白文本。",
    ),
    # Warining3: incomplete prompt info
    warn_incomplete_prompt=dict(
        en="Please provide prompt audio and text for both speaker 1 and speaker 2",
        zh="请提供说话人 1 与说话人 2 的参考语音与参考文本",
    ),
)

global_lang: Literal["zh", "en"] = "en"


def i18n(key):
    global global_lang
    return _i18n_key2lang_dict[key][global_lang]


def check_monologue_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True


def check_dialogue_text(text_list: List[str]) -> bool:
    if len(text_list) == 0:
        return False
    for text in text_list:
        if not (
            check_monologue_text(text, "[S1]")
            or check_monologue_text(text, "[S2]")
            or check_monologue_text(text, "[S3]")
            or check_monologue_text(text, "[S4]")
        ):
            return False
    return True


def synthesis_function(
    mode: Literal[0, 1],
    target_text: str,
    sentence_split: bool = False,
    voice_mode: Literal[0, 1] = 0,  # 0 means voice clone
    spk1_prompt_text: str | None = "",
    spk1_prompt_audio: str | None = None,
    spk2_prompt_text: str | None = "",
    spk2_prompt_audio: str | None = None,
):
    if "pretrained_dir" not in model_options:
        gr.Warning(message="Model not initialized.")
        return None

    gen_type = "dialogue" if mode == 0 else "monologue"
    progress_bar = gr.Progress(track_tqdm=True)
    current_model = initiate_model(
        pretrained_dir=model_options["pretrained_dir"],
        gen_type=gen_type,
        device=model_options.get("device"),
    )

    if mode == 0:
        if voice_mode == 0:
            prompt_has_value = [
                spk1_prompt_text != "",
                spk1_prompt_audio is not None,
                spk2_prompt_text != "",
                spk2_prompt_audio is not None,
            ]
            if not all(prompt_has_value):
                gr.Warning(message=i18n("warn_incomplete_prompt"))
                return None
            if not check_monologue_text(spk1_prompt_text, "[S1]"):
                gr.Warning(message=i18n("warn_invalid_spk1_prompt_text"))
                return None
            if not check_monologue_text(spk2_prompt_text, "[S2]"):
                gr.Warning(message=i18n("warn_invalid_spk2_prompt_text"))
                return None

        target_text_list: List[str] = re.findall(r"(\[S[0-9]\][^\[\]]*)", target_text)
        target_text_list = [text.strip() for text in target_text_list]
        if not check_dialogue_text(target_text_list):
            gr.Warning(message=i18n("warn_invalid_dialogue_text"))
            return None

        prompt_wav_list = (
            None if voice_mode != 0 else [spk1_prompt_audio, spk2_prompt_audio]
        )
        prompt_text_list = (
            None if voice_mode != 0 else [spk1_prompt_text, spk2_prompt_text]
        )
        target_audio = current_model.generate_dialogue(
            text_list=target_text_list,
            prompt_wav_list=prompt_wav_list,
            prompt_text_list=prompt_text_list,
            temperature=0.9,
            topk=30,
        )
        return (24000, target_audio.squeeze(0).cpu().numpy())

    text = target_text.strip()
    if not check_monologue_text(text):
        gr.Warning(message=i18n("warn_invalid_monologue_text"))
        return None

    prompt_wav = None
    prompt_text = None
    if voice_mode == 0:
        if spk1_prompt_text == "" or spk1_prompt_audio is None:
            gr.Warning(message=i18n("warn_incomplete_prompt"))
            return None
        if not check_monologue_text(spk1_prompt_text, "[S1]"):
            gr.Warning(message=i18n("warn_invalid_spk1_prompt_text"))
            return None
        prompt_wav = spk1_prompt_audio
        prompt_text = spk1_prompt_text

    target_audio = current_model.generate_monologue(
        text=text,
        prompt_wav=prompt_wav,
        prompt_text=prompt_text,
        sentence_split=sentence_split,
    )
    return (24000, target_audio.squeeze(0).cpu().numpy())


# UI rendering
def render_interface() -> gr.Blocks:
    with gr.Blocks(title="FireRedTTS-2", theme=gr.themes.Default()) as page:
        # ======================== UI ========================
        # A large title
        title_desc = gr.Markdown(value="# {}".format(i18n("title_md_desc")))
        with gr.Row():
            lang_choice = gr.Radio(
                choices=["中文", "English"],
                value="English",
                label="Display Language/显示语言",
                type="index",
                interactive=True,
            )
            mode_choice = gr.Radio(
                choices=[
                    i18n("synthesis_mode_choice_dialogue"),
                    i18n("synthesis_mode_choice_monologue"),
                ],
                value=i18n("synthesis_mode_choice_monologue"),
                label=i18n("synthesis_mode_label"),
                type="index",
                interactive=True,
            )
            voice_mode_choice = gr.Radio(
                choices=[i18n("voice_model_choice1"), i18n("voice_model_choice2")],
                value=i18n("voice_model_choice1"),
                label=i18n("voice_mode_label"),
                type="index",
                interactive=True,
            )
        with gr.Row():
            # ==== Speaker1 Prompt ====
            with gr.Column(scale=1):
                with gr.Group(visible=True) as spk1_prompt_group:
                    spk1_prompt_audio = gr.Audio(
                        label=i18n("spk1_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )  # Audio component returns tmp audio path
                    spk1_prompt_text = gr.Textbox(
                        label=i18n("spk1_prompt_text_label"),
                        placeholder=i18n("spk1_prompt_text_placeholder"),
                        lines=3,
                    )
            # ==== Speaker2 Prompt ====
            with gr.Column(scale=1):
                with gr.Group(visible=False) as spk2_prompt_group:
                    spk2_prompt_audio = gr.Audio(
                        label=i18n("spk2_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    spk2_prompt_text = gr.Textbox(
                        label=i18n("spk2_prompt_text_label"),
                        placeholder=i18n("spk2_prompt_text_placeholder"),
                        lines=3,
                    )
            # ==== Text input ====
            with gr.Column(scale=2):
                dialogue_text_input = gr.Textbox(
                    label=i18n("monologue_text_input_label"),
                    placeholder=i18n("monologue_text_input_placeholder"),
                    lines=18,
                )
                sentence_split_checkbox = gr.Checkbox(
                    label=i18n("sentence_split_label"),
                    value=False,
                    visible=True,
                )
        # Generate button
        generate_btn = gr.Button(
            value=i18n("generate_btn_label"), variant="primary", size="lg"
        )
        # Long output audio
        generate_audio = gr.Audio(
            label=i18n("generated_monologue_label"),
            interactive=False,
        )

        # ======================== Action ========================
        # Language action
        def _change_component_language(lang, mode, voice_mode):
            global global_lang
            global_lang = ["zh", "en"][lang]
            mode_choices = [
                i18n("synthesis_mode_choice_dialogue"),
                i18n("synthesis_mode_choice_monologue"),
            ]
            voice_choices = [
                i18n("voice_model_choice1"),
                i18n("voice_model_choice2"),
            ]
            text_label_key = (
                "dialogue_text_input_label"
                if mode == 0
                else "monologue_text_input_label"
            )
            text_placeholder_key = (
                "dialogue_text_input_placeholder"
                if mode == 0
                else "monologue_text_input_placeholder"
            )
            audio_label_key = (
                "generated_audio_label"
                if mode == 0
                else "generated_monologue_label"
            )
            return [
                # title_desc
                gr.update(value="# {}".format(i18n("title_md_desc"))),
                # mode_choice
                gr.update(
                    choices=mode_choices,
                    value=mode_choices[mode],
                    label=i18n("synthesis_mode_label"),
                ),
                # voice_mode_choice
                gr.update(
                    choices=voice_choices,
                    value=voice_choices[voice_mode],
                    label=i18n("voice_mode_label"),
                ),
                # spk1_prompt_{audio,text}
                gr.update(label=i18n("spk1_prompt_audio_label")),
                gr.update(
                    label=i18n("spk1_prompt_text_label"),
                    placeholder=i18n("spk1_prompt_text_placeholder"),
                ),
                # spk2_prompt_{audio,text}
                gr.update(label=i18n("spk2_prompt_audio_label")),
                gr.update(
                    label=i18n("spk2_prompt_text_label"),
                    placeholder=i18n("spk2_prompt_text_placeholder"),
                ),
                # dialogue_text_input
                gr.update(
                    label=i18n(text_label_key),
                    placeholder=i18n(text_placeholder_key),
                ),
                # sentence_split_checkbox
                gr.update(label=i18n("sentence_split_label")),
                # generate_btn
                gr.update(value=i18n("generate_btn_label")),
                # generate_audio
                gr.update(label=i18n(audio_label_key)),
            ]

        lang_choice.change(
            fn=_change_component_language,
            inputs=[lang_choice, mode_choice, voice_mode_choice],
            outputs=[
                title_desc,
                mode_choice,
                voice_mode_choice,
                spk1_prompt_audio,
                spk1_prompt_text,
                spk2_prompt_audio,
                spk2_prompt_text,
                dialogue_text_input,
                sentence_split_checkbox,
                generate_btn,
                generate_audio,
            ],
        )

        def _handle_mode_change(mode, voice_mode):
            target_gen_type = "dialogue" if mode == 0 else "monologue"
            if model_gen_type is not None and model_gen_type != target_gen_type:
                reset_model_cache()

            text_label_key = (
                "dialogue_text_input_label"
                if mode == 0
                else "monologue_text_input_label"
            )
            text_placeholder_key = (
                "dialogue_text_input_placeholder"
                if mode == 0
                else "monologue_text_input_placeholder"
            )
            audio_label_key = (
                "generated_audio_label"
                if mode == 0
                else "generated_monologue_label"
            )
            show_spk1 = voice_mode == 0
            show_spk2 = voice_mode == 0 and mode == 0
            show_sentence_split = mode == 1

            return [
                gr.update(
                    label=i18n(text_label_key),
                    placeholder=i18n(text_placeholder_key),
                ),
                gr.update(label=i18n(audio_label_key), value=None),
                gr.update(visible=show_spk1),
                gr.update(visible=show_spk2),
                gr.update(visible=show_sentence_split),
            ]

        mode_choice.change(
            fn=_handle_mode_change,
            inputs=[mode_choice, voice_mode_choice],
            outputs=[
                dialogue_text_input,
                generate_audio,
                spk1_prompt_group,
                spk2_prompt_group,
                sentence_split_checkbox,
            ],
        )

        # Voice clone mode action
        def _change_prompt_input_visibility(voice_mode, mode):
            enable = voice_mode == 0
            return [gr.update(visible=enable), gr.update(visible=enable and mode == 0)]

        voice_mode_choice.change(
            fn=_change_prompt_input_visibility,
            inputs=[voice_mode_choice, mode_choice],
            outputs=[spk1_prompt_group, spk2_prompt_group],
        )
        generate_btn.click(
            fn=synthesis_function,
            inputs=[
                mode_choice,
                dialogue_text_input,
                sentence_split_checkbox,
                voice_mode_choice,
                spk1_prompt_text,
                spk1_prompt_audio,
                spk2_prompt_text,
                spk2_prompt_audio,
            ],
            outputs=[generate_audio],
        )
    return page


# ================================================
#                   Options
# ================================================
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained-dir", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Initiate model
    initiate_model(args.pretrained_dir)
    print("[INFO] FireRedTTS-2 loaded")
    # UI
    page = render_interface()
    page.launch(server_name="0.0.0.0")