'''
GCS 버킷의 특정 폴더에서 파일 리스트를 가져온다. 

GCS에서 파일을 로드하여 데이터를 반환한다. 
'''

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage 

# 같은 경로에 있는 env 파일을 불러온다. 
load_dotenv()

def list_files_in_gcs(bucket_name, prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    file_list = [blob.name for blob in blobs if '.' in blob.name]
    
    return file_list 


def load_gcs_file(bucket_name, file_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.download_as_text() if file_path.endswith('.txt') else blob.download_as_bytes()