'''
Google Cloud Storage에 있는 split된 음성 데이터와 matching 된 텍스트를 가져와서 
Whisper Model Fine Tuning을 하로록 방향을 제공하는 main 함수 
'''

import os 
import optuna 
import json

from dotenv import load_dotenv
from data_matching import match_audio_text_files
from dataset_preparation import prepare_dataset
from whisper_finetuning import objective

# 같은 경로에 있는 env 파일을 불러온다. 
load_dotenv()

# 최적 모델 추적 전역 변수
best_metrics = None
best_model_dir = None 

def main():
    bucket_name = os.getenv("BUCKET_NAME") # GCS 버킷 이름 
    audio_prefix = os.getenv("SOURCE_WAV_FOLDER") # 오디오 폴더 
    text_prefix = os.getenv("SOURCE_TEXT_FOLDER") # 텍스트 폴더 
    
    # 1. 오디오와 텍스트 파일 매칭 
    print("Matching audio and text files...")
    matched_files = match_audio_text_files(bucket_name, audio_prefix, text_prefix)
    
    # 2. 데이터셋 준비
    print("Preparing dataset...")
    dataset = prepare_dataset(bucket_name, matched_files)
    
    # 3. Whisper 모델 파인튜닝 
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize') # 최소화할 매트릭으로 최적화
    study.optimize(lambda trial : objective(trial, dataset), n_trials=2, n_jobs=-1) # 병렬로 4개 작업 실행 
    
    # 최적의 하이퍼파라미터 출력
    print("\n=== Optimization Results ===")
    print("Best metrics:", best_metrics)
    print("Best validation WER:", best_metrics["wer"])
    print("Best validation CER:", best_metrics["cer"])

    # 최적의 하이퍼파라미터 저장 
    with open("best_metrics.json", "w") as f:
        json.dump(best_metrics, f)
    print("Best metrics and model saved")
    
    
if __name__ == "__main__":
    main()
