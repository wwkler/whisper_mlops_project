'''
음성과 텍스트 파일 데이터셋을 준비한다. 
'''

from gcs_utils import load_gcs_file
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor

def prepare_dataset(bucket_name, matched_files):
    # 딕셔너리 형태로 바로 누적 
    data = {
        'audio' : [],
        'text' : [],
    }
    
    def load_data(pair):
        # GCS에서 파일 로드 
        audio_data = load_gcs_file(bucket_name, pair['audio'])
        text_data = load_gcs_file(bucket_name, pair['text'])
        
        # 각 리스트에 추가 
        data['audio'].append(audio_data)
        data['text'].append(text_data)
    
    # 비동기적으로 파일 로드 및 딕셔너리 누적 
    with ThreadPoolExecutor() as executor:
        executor.map(load_data, matched_files)
        
    # Dataset 생성
    dataset = Dataset.from_dict(data)
    return dataset 
    