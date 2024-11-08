'''
Google Cloud Storage 'svtf'(약칭) Bucket에 
현재 시간과 파일을 업로드 한 시간(생성 시간)이 10일 차이가 나면 
해당 음성 파일과 매칭되는 텍스트 파일을 Google Big Query에 저장한다. 
'''
import pytz
import json
import base64
import gzip 
import os
import logging 

from google.cloud import storage, bigquery
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 같은 경로에 .env 파일을 가져온다. GCP와 소통하기 위한 인증 키를 가져온다. 
load_dotenv()

# 클라이언트 초기화
storage_client = storage.Client()
bigquery_client = bigquery.Client(location="asia-northeast3")

# 한국 시간대 설정
KST = pytz.timezone('Asia/Seoul')

def is_old_file(blob, days=10):
    """파일이 지정된 일 수(기본 10일) 이상 한국 시간 기준으로 오래된 경우 True 반환"""
    current_time_kst = datetime.now(KST)  # 한국 시간으로 현재 시간 얻기
    
    # print(f"blob.time_created.astimezone(KST) : {blob.time_created.astimezone(KST)}")
    return blob.time_created.astimezone(KST) < current_time_kst - timedelta(days=days)

def get_matching_text_file(text_folder, wav_filename):
    """텍스트 폴더에서 해당 음성 파일과 매칭되는 텍스트 파일을 찾음"""
    text_file_name = wav_filename.replace('.wav', '.txt')  # 매칭되는 텍스트 파일 이름 생성
    blobs = storage_client.list_blobs(bucket_name, prefix=text_folder)
    for blob in blobs:
        if blob.name.endswith(text_file_name):
            return blob
    return None

def convert_size(size_bytes):
    """파일 크기를 KB 또는 MB로 변환하여 문자열로 반환"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
   
def compress_and_encode_audio(blob):
    """음성 파일을 gzip 압축 후 base64로 인코딩"""
    audio_data = blob.download_as_bytes()
    compressed_data = gzip.compress(audio_data)
    return base64.b64encode(compressed_data).decode('utf-8')
 
def insert_data_to_bigquery(google_bigquery_table_id, wav_data, wav_metadata, text_content):
    """BigQuery에 음성 파일 데이터와 메타데이터, 텍스트 내용을 삽입"""
    rows_to_insert = [
        {
            "filename": wav_metadata["filename"].split('/')[-1],  # 파일명만 저장
            "size": wav_metadata["size"],
            "content_type": wav_metadata["content_type"],
            "creation_time": wav_metadata["creation_time"],
            "audio_data": wav_data,  # base64 인코딩된 오디오 데이터
            "text_content": text_content,
        }
    ]

    errors = bigquery_client.insert_rows_json(google_bigquery_table_id, rows_to_insert)
    if errors:
        logging.info(f"Failed to insert rows: {errors}")
        # print(f"Failed to insert rows: {errors}")
    else:
        logging.info(f"Data inserted into BigQuery for file: {wav_metadata['filename']}")
        # print(f"Data inserted into BigQuery for file: {wav_metadata['filename']}")

def process_single_file(blob, text_folder, google_bigquery_table_id):
    """단일 파일을 BigQuery에 저장하고 GCS에서 삭제하는 비동기 작업"""
    
    # 임시적으로 주석 처리를 한다. 
    # if not is_old_file(blob):
    #     print(f"File {blob.name} is not old enough to process.")
    #     return

    logging.info(f"Processing file: {blob.name}")
    # print(f"Processing file: {blob.name}")
    
    # 매칭되는 텍스트 파일 찾기
    text_blob = get_matching_text_file(text_folder, blob.name.split('/')[-1])
    if text_blob:
        # 압축 및 인코딩된 음성 데이터
        wav_data = compress_and_encode_audio(blob)

        # 파일 메타데이터
        wav_metadata = {
            "filename": str(blob.name).replace(".wav", ""),
            "size": convert_size(blob.size),
            "content_type": blob.content_type,
            "creation_time": blob.time_created.isoformat() # Google Big Query에서 데이터 타입이 TIMESTAMP 속성은 UTC로 적용된다.
        }

        # 텍스트 파일 내용
        text_content = text_blob.download_as_text()

        # BigQuery에 데이터 삽입
        insert_data_to_bigquery(google_bigquery_table_id, wav_data, wav_metadata, text_content)

        # BigQuery에 성공적으로 삽입 후 파일 삭제 -> 임시적으로 주석 처리를 한다. 
        # blob.delete()
        # text_blob.delete()
        # logging.info(f"Deleted files from GCS: {blob.name}, {text_blob.name}")
        # print(f"Deleted files from GCS: {blob.name}, {text_blob.name}")
    else:
        logging.info(f"No matching text file found for {blob.name}")
        # print(f"No matching text file found for {blob.name}")

def process_files(bucket_name, wav_folder, text_folder, google_bigquery_table_id, days=10):
    """오래된 음성 파일을 비동기 방식으로 처리"""
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=wav_folder)
    
    # ThreadPoolExecutor를 사용해 비동기적으로 파일 처리
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [
            executor.submit(process_single_file, blob, text_folder, google_bigquery_table_id)
            for blob in blobs if blob.name.endswith('.wav')
        ]

        for future in as_completed(futures):
            try:
                future.result()  # 예외가 발생하면 여기서 확인
            except Exception as e:
                logging.info(f"Error processing file: {e}")
                # print(f"Error processing file: {e}")
                
# 실행 예시
bucket_name = os.getenv("BUCKET_NAME")
wav_folder = os.getenv("SOURCE_WAV_FOLDER")
text_folder = os.getenv("SOURCE_TEXT_FOLDER")
google_bigquery_table_id = os.getenv("GOOGLE_BIGQUERY_TABLE_ID") # BigQuery 테이블 ID
process_files(bucket_name, wav_folder, text_folder, google_bigquery_table_id)