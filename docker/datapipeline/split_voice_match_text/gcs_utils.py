'''
Google Cloud Storage에서 원본 음성 파일을 가져오는 역할

Goole Cloud Storage에 split된 음성 파일과 매칭되는 text를 저장하는 역할 
'''
import soundfile as sf
import uuid
import logging

from google.cloud import storage
from io import BytesIO
from dotenv import load_dotenv

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 환경 변수 불러오기
load_dotenv() 

# GCS에서 음성 파일을 메모리로 불러오는 함수 
def load_audio_from_gcs(bucket_name, object_name):
    """GCS에서 음성 파일을 메모리로 불러오기"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    return blob.download_as_bytes()

# 분할된 오디오 파일과 매칭되는 텍스트 파일을 GCS에 업로드하는 함수 
def upload_audio_to_gcs(bucket_name, audio_clip, text, sr):
    """분할된 오디오 파일과 텍스트 파일을 GCS에 업로드"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # 랜덤한 UUID 생성하여 파일 이름에 사용
    file_id = str(uuid.uuid4())

    # 오디오 파일을 저장하기 위한 준비
    audio_file = BytesIO()
    sf.write(audio_file, audio_clip, sr, format='wav')
    audio_file.seek(0)

    # 오디오 파일을 GCS에 업로드
    audio_blob = bucket.blob(f"check_voice_time_folder/wav/{file_id}.wav")
    audio_blob.upload_from_file(audio_file, content_type='audio/wav')
    logger.info(f"Uploaded {file_id}.wav to GCS.")
    # print(f"Uploaded {file_id}.wav to GCS.")

    # 텍스트 파일을 UTF-8로 인코딩하여 GCS에 업로드
    text_blob = bucket.blob(f"check_voice_time_folder/text/{file_id}.txt")
    text_blob.upload_from_string(text.encode('utf-8'), content_type='text/plain; charset=utf-8')
    logger.info(f"Uploaded {file_id}.txt to GCS.")
    # print(f"Uploaded {file_id}.txt to GCS.")