'''
메인 로직 및 파일 처리 흐름

여기서 시작하는 모든 작업은 동기로 진행한다.
'''

import os 
import json

from gcs_utils import load_audio_from_gcs, upload_audio_to_gcs
from audio_processing import split_audio_into_chunks, transcribe_audio_with_whisper, split_audio_by_timestamps
from dotenv import load_dotenv

# 같은 경로에 있는 env 파일을 불러온다. 
load_dotenv()

def read_metadata_from_file(file_path):
    """메타데이터 파일에서 GCS 정보 읽기"""
    with open(file_path, 'r') as file:
        metadata_list = json.load(file)
    return metadata_list

def process_audio_chunks(audio_data, sr, bucket_name):
    """음성 파일을 5분 단위로 자른 후, 각각의 구간에 대해 Whisper 처리"""
    audio_chunks = split_audio_into_chunks(audio_data, sr)
    
    # print(f"audio_chunks : {audio_chunks}")

    for idx, audio_chunk in enumerate(audio_chunks):
        print(f"Processing chunk {idx+1}/{len(audio_chunks)}...")

        # Whisper로 텍스트 및 타임스탬프 추출
        segments = transcribe_audio_with_whisper(audio_chunk, sr)
        
        print(f"Segments : {segments}" + "\n")

        # Whisper가 제공한 타임스탬프에 따라 음성을 세부적으로 자르기
        audio_clips, texts = split_audio_by_timestamps(audio_chunk, segments, sr)
        
        print(f"{idx} audio_clips : {audio_clips}" + "\n")
        
        print(f"{idx} texts : {texts}" + "\n")

        # 분할된 오디오 및 텍스트 파일을 GCS에 업로드
        for audio_clip, text in zip(audio_clips, texts):
             upload_audio_to_gcs(os.getenv("SPLIT_BUCKET"), audio_clip, text, sr)

def process_audio_from_gcs(metadata_file_path):
    """메타데이터를 기반으로 GCS에서 음성 파일을 처리"""
    metadata_list = read_metadata_from_file(metadata_file_path)

    for metadata in metadata_list:
        bucket_name = metadata['bucketId']
        object_name = metadata['objectId']

        print(f"Processing {object_name} from bucket {bucket_name}...")

        # GCS에서 음성 파일 불러오기
        audio_data = load_audio_from_gcs(bucket_name, object_name)

        # 샘플링 레이트 설정 (Whisper 기본값 16000)
        sr=16000

        # 5분 단위로 나눈 후 Whisper로 처리
        process_audio_chunks(audio_data, sr, bucket_name)

# 실행
metadata_file_path = os.getenv("TEXT_FILE_PATH")
process_audio_from_gcs(metadata_file_path)