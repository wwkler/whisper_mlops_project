'''
음성과 텍스트 파일을 매칭하여 데이터가 위치한 경로를 확인한다. 
'''

import os 
import re

from concurrent.futures import ThreadPoolExecutor
from gcs_utils import list_files_in_gcs

def match_audio_text_files(bucket_name, audio_prefix, text_prefix):
    with ThreadPoolExecutor() as executor:
        audio_files = executor.submit(list_files_in_gcs, bucket_name, audio_prefix).result()
        text_files = executor.submit(list_files_in_gcs, bucket_name, text_prefix).result()
    
    audio_base = {
        os.path.splitext(os.path.basename(f))[0] : f for f in audio_files
    }
    
    text_base = {
        os.path.splitext(os.path.basename(f))[0] : f for f in text_files
    }
    
    # print(f'audio_base : {audio_base}') # {'004abc84-5536-4659-84b4-4ba99764bd63': 'service_voice_text_folder/wav/004abc84-5536-4659-84b4-4ba99764bd63.wav'}
    
    # print(f'text_base : {text_base}') {'004abc84-5536-4659-84b4-4ba99764bd63': 'service_voice_text_folder/text/004abc84-5536-4659-84b4-4ba99764bd63.txt'}
    
    matched_files = [
        {
            'audio' : audio_base[key],
            'text' : text_base[key]
        }
        for key in audio_base.keys() & text_base.keys()
    ]
    
    return matched_files
    
    