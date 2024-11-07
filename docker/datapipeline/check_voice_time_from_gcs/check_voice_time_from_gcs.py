'''
도합 음성 시간을 구하는 과정은 비동기로 처리한다.
Google Cloud Storage Bucket에 있는 wav 폴더에 있는 여러 음성 시간을 비동기 방식으로 얻어내고 그것을 바탕으로 총 도합 시간을 구한다. 

Google Cloud Storage 'cvtf'(약칭) Bucket에 있는 split된 음성 파일이 도합 50시간이 넘는다면

Google Cloud Storage 'svtf'(약칭) Bucket으로 mv 한다. 
mv 하는 과정도 비동기로 처리한다. 
'''
import os
import io
import wave
import logging

from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 환경 변수 불러오기 -> GCP에 접근하기 위한 서비스 키를 얻어온다. 
load_dotenv() 

def calculate_audio_duration(blob):
    """GCS 블롭에서 음성 파일의 길이를 초 단위로 계산"""
    audio_data = blob.download_as_bytes()
    with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)  # 초 단위 재생 시간
    return duration

def calculate_total_duration(bucket_name, source_wav_folder, max_workers=20):
    """비동기로 GCS의 모든 wav 파일의 총 재생 시간을 계산"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_wav_folder)
    
    total_duration = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_audio_duration, blob): blob.name for blob in blobs if blob.name.endswith(".wav")}
        
        for future in as_completed(futures):
            try:
                duration = future.result()
                logger.info(f"Processed  {futures[future]} - duration : {duration} Seconds")
                # print(f"Processed  {futures[future]} - duration : {duration} Seconds")
                
                total_duration += duration
            except Exception as e:
                logger.info(f"Failed to process {futures[future]}: {e}")
                # print(f"Failed to process {futures[future]}: {e}")

    # 총 시간을 초에서 시간 단위로 변환
    # 총 시간을 초에서 시간 단위에서 분으로 변환 
    total_duration_hours = total_duration / 3600
    total_duration_minutes = total_duration_hours * 60
    return total_duration_hours, total_duration_minutes

def move_single_blob(blob, bucket, destination_folder):
    """단일 파일을 새로운 폴더로 비동기 이동"""
    new_blob_name = f"{destination_folder}/{blob.name.split('/')[-1]}"
    
    # 파일 복사 후 원본 삭제하여 이동 구현
    bucket.copy_blob(blob, bucket, new_blob_name)
    
    # text와 wav 폴더는 사라지지 않게 해야 함 
    blob.delete()
    logger.info(f"Moved {blob.name} to {new_blob_name}")
    # print(f"Moved {blob.name} to {new_blob_name}")
    
def source_folder_move_files_destination_folder(bucket_name, source_folder, destination_folder, max_workers=10):
    """같은 bucket에서 source_folder에서 destination_folder로 텍스트 파일과 음성 파일 이동"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_folder)
    
    # 비동기 이동을 위한 ThreadPoolExecutor 사용
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(move_single_blob, blob, bucket, destination_folder) for blob in blobs]
        
        # 각 작업이 완료될 때마다 결과 출력
        for future in as_completed(futures):
            try:
                future.result()  # 결과가 성공적으로 완료됐는지 확인
            except Exception as e:
                logger.info(f"Error moving file: {e}")
                # print(f"Error moving file: {e}")
   
def main():
    bucket_name = os.getenv("BUCKET_NAME")
    source_wav_folder = os.getenv("SOURCE_WAV_FOLDER")

    # 총 재생 시간 계산
    total_duration_hours, total_duration_minutes = calculate_total_duration(bucket_name, source_wav_folder)
    
    logger.info("\n" + f"Total audio duration in  {bucket_name}/{source_wav_folder} folder: {total_duration_hours:.2f} hours")
    # print("\n" + f"Total audio duration in  {bucket_name}/{source_wav_folder} folder: {total_duration_hours:.2f} hours")
    
    logger.info(f"Total audio duration in {bucket_name}/{source_wav_folder} folder: {total_duration_minutes:.2f} minutes" + "\n")
    # print(f"Total audio duration in {bucket_name}/{source_wav_folder} folder: {total_duration_minutes:.2f} minutes" + "\n")

     # 도합 음성 파일이 0.05시간 이상일 떄 파일 이동 -> 추후 50시간으로 정정해야 한다. 
    if total_duration_hours >= 0.05:
        logger.info(f"Total duration is {total_duration_hours:.2f} hours, exceeding 0.05 hours. Moving files...")
        # print(f"Total duration is {total_duration_hours:.2f} hours, exceeding 0.05 hours. Moving files...")
        
        destination_wav_folder = os.getenv("DESTINATION_WAV_FOLDER")
        source_text_folder = os.getenv("SOURCE_TEXT_FOLDER")
        destination_text_folder = os.getenv("DESTINATION_TEXT_FOLDER")
        
        # wav 폴더의 파일 이동
        source_folder_move_files_destination_folder(bucket_name, source_wav_folder, destination_wav_folder)
        
        # text 폴더의 파일 이동
        source_folder_move_files_destination_folder(bucket_name, source_text_folder, destination_text_folder)
        
        print("airflow_trigger_github_actions")  # Docker Operator에 Xcom 저장
        
    else:
        logger.info("Total duration is under 0.05 hours. No files moved.")
        # print("Total duration is under 0.05 hours. No files moved.")
        
        print("airflow_end_task")  # Docker Operator에 Xcom 저장
        
main()