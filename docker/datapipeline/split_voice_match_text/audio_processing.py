'''
음성 파일에 대한 처리를 수행하는 역할 
'''
import librosa
import whisper
import numpy as np

from io import BytesIO
from pydub import AudioSegment

# 오디오 전처리 함수 (노이즈 감소 및 정규화)
def preprocess_audio(audio_np, sr):
    """오디오 데이터 전처리: 노이즈 감소 및 RMS 정규화"""
    # RMS를 사용하여 음성 신호 정규화
    rms = librosa.feature.rms(y=audio_np)[0]
    audio_np = audio_np / (np.max(rms) + 1e-6)  # 정규화

    return audio_np

# 원본 음성 파일을 5분 단위로 자르는 함수 
def split_audio_into_chunks(audio_data, sr, chunk_duration=300):
    """음성 파일을 5분 단위로 자르기"""
    audio_np, _ = librosa.load(BytesIO(audio_data), sr=sr)
    audio_np = preprocess_audio(audio_np, sr)  # 전처리 추가
     
    chunk_samples = chunk_duration * sr  # 5분(300초) 단위로 샘플 수 계산
    total_samples = len(audio_np)

    audio_chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        audio_chunks.append(audio_np[start:end])

    return audio_chunks

# Whisper로 텍스트 및 타임스탬프 추출하는 함수 
def transcribe_audio_with_whisper(audio_np, sr, model_size="small"):
    """Whisper로 텍스트 및 타임스탬프 추출"""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_np, fp16=False)
    return result['segments']

# Whisper 타임스탬프를 샘플 수로 변환하여 오디오 파일을 자르는 함수 
def split_audio_by_timestamps(audio_np, segments, sr):
    """Whisper 타임스탬프를 샘플 수로 변환하여 오디오 파일을 자르기"""
    audio_clips, texts = [], []
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        texts.append(text)

        # 타임스탬프를 샘플 수로 변환 (반올림 사용)
        start_sample = round(start_time * sr)
        end_sample = round(end_time * sr)

        # 해당 구간의 오디오 데이터를 추출
        audio_clip = audio_np[start_sample:end_sample]
        
        audio_clips.append(audio_clip)
        
    return audio_clips, texts
