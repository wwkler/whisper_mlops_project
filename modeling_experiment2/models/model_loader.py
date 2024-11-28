# models/model_loader.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_whisper_model_and_processor(model_name="openai/whisper-tiny"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Configure model for fine-tuning
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.config.suppress_tokens = []

    return model, processor
