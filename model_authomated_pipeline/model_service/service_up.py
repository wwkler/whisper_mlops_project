import torch
import librosa
import numpy as np
import soundfile as sf  # Sound processing library
import bentoml
import time
import gc

from bentoml.io import File, JSON
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# CPU 쓰레드 제한
torch.set_num_threads(2)  # CPU 쓰레드 수를 2개로 제한


def get_deploy_model():
    """
    stage=deploy인 BentoML 모델 로드.
    """
    print("[INFO] Searching for deploy model...")
    for model in bentoml.models.list():
        if model.info.labels.get("stage") == "deploy":
            print(f"[INFO] Deploy model found: {model.tag}")
            return model
    raise ValueError("[ERROR] No deploy model found.")

# 1. stage=deploy 모델 로드
try:
    model_ref = get_deploy_model()
    model_path = model_ref.path
    print(f"[INFO] Using model from path: {model_path}")
except Exception as e:
    print(f"[ERROR] Failed to load deploy model: {e}")
    raise

# 2. Whisper 모델 및 프로세서 초기화
try:
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    model.eval()  # 평가 모드 설정
    print("[INFO] Whisper model and processor initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize Whisper model or processor: {e}")
    raise

# 3. 서비스 정의
svc = bentoml.Service("whisper_service", runners=[])

def split_audio(audio_data, sr, chunk_duration=10):
    """
    Split audio data into smaller chunks.
    :param audio_data: NumPy array of audio samples.
    :param sr: Sampling rate of the audio.
    :param chunk_duration: Duration of each chunk in seconds.
    :return: List of audio chunks.
    """
    chunk_size = sr * chunk_duration  # 청크의 샘플 크기
    chunks = [
        audio_data[i : i + chunk_size]
        for i in range(0, len(audio_data), chunk_size)
    ]
    return chunks


def process_chunks(chunks, processor, model):
    """
    Process each audio chunk with the Whisper model.
    :param chunks: List of audio chunks.
    :param processor: Whisper processor instance.
    :param model: Whisper model instance.
    :return: List of transcriptions for each chunk.
    """
    transcriptions = []
    previous_transcription = ""

    for idx, chunk in enumerate(chunks):
        # Whisper 전처리
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")

        # 이전 문맥 추가 (Prompt 사용)
        if previous_transcription:
            prompt_ids = processor.tokenizer.encode(previous_transcription, add_special_tokens=False)
            inputs["decoder_input_ids"] = torch.tensor([prompt_ids], dtype=torch.long)

        # 모델 추론
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features,  num_beams=1,)  # Beam Search 비활성화

        # 결과 디코딩
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)

        # 이전 청크 결과를 업데이트
        previous_transcription = transcription
        
        # 메모리 해제
        gc.collect()

        print(f"[INFO] Processed chunk {idx + 1}/{len(chunks)}: {transcription}")

    return transcriptions


@svc.api(input=File(), output=JSON())
def transcribe(audio_file):
    """
    Transcribe long audio files by splitting them into smaller chunks.
    """
    try:
        print("[INFO] Received transcription request.")

        # 오디오 로드
        audio_data, sr = sf.read(audio_file)
        if sr != 16000:
            print("[INFO] Resampling audio to 16kHz...")
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        # 오디오를 청크로 분할
        print("[INFO] Splitting audio into chunks...")
        chunks = split_audio(audio_data, sr=16000, chunk_duration=10)  # 10초 단위로 분할

        # 청크별로 추론
        print("[INFO] Processing audio chunks...")
        transcriptions = process_chunks(chunks, processor, model)

        # 결과 병합
        final_transcription = " ".join(transcriptions)
        print("[INFO] Final transcription completed.")
        return {"transcription": final_transcription}

    except Exception as e:
        print(f"[ERROR] Error during transcription: {e}")
        return {"error": str(e)}
