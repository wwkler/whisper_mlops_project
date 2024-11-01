'''
메인 로직 및 파일 처리 흐름

아래 2가지 작업은 비동기 작업으로 진행한다. 

 -> Whisper 모델로 음성을 분석하여 타임스탬프와 텍스트를 추출하는 작업 

 -> GCS로 분할된 음성 파일과 텍스트 파일을 업로드하는 작업 
'''

import os 
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
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

def process_chunk(audio_chunk, sr):
    """Whisper로 구간별 타임스탬프와 텍스트를 추출하고 GCS에 업로드"""
    segments = transcribe_audio_with_whisper(audio_chunk, sr)
    audio_clips, texts = split_audio_by_timestamps(audio_chunk, segments, sr)

    upload_results = []
    with ThreadPoolExecutor() as upload_executor:
        upload_futures = [
            upload_executor.submit(upload_audio_to_gcs, os.getenv("SPLIT_BUCKET"), audio_clip, text, sr)
            for audio_clip, text in zip(audio_clips, texts)
        ]
        for future in as_completed(upload_futures):
            upload_results.append(future.result())

    return upload_results

def process_audio_chunks(audio_data, sr, bucket_name):
    """음성 파일을 5분 단위로 자른 후, 각각의 구간에 대해 Whisper 처리"""
    audio_chunks = split_audio_into_chunks(audio_data, sr)
    
    # 5분 간격으로 자른 음성 파일을 병렬로 수행해서 처리하는 작업을 수행한다. 
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, audio_chunk in enumerate(audio_chunks):
            print(f"Processing chunk {idx+1}/{len(audio_chunks)}...")
            futures.append(executor.submit(process_chunk, audio_chunk, sr))
        
        for future in as_completed(futures):
            print(f"Chunk processing result: {future.result()}")
  
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

def clear_metadata_file(file_path):
    """메타데이터 파일을 빈 파일로 초기화"""
    with open(file_path, 'w') as file:
        file.write("")
    print(f"{file_path} has been cleared.")

# 실행
metadata_file_path = os.getenv("TEXT_FILE_PATH")
process_audio_from_gcs(metadata_file_path)
clear_metadata_file(metadata_file_path)