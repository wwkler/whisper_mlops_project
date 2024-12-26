# main.py
import os
import subprocess
import pickle # 직렬화 모듈 

from dotenv import load_dotenv
from data.gcs_loader import load_data_from_gcs
from data.dataset_splitter import create_dataset_dict
from make_model.model_loader import load_whisper_model_and_processor
from make_model.raytune_finetuning import find_best_hyperparameters # raytune를 활용한 whisper 파인 튜닝 시도 
#from make_model.joblib_finetuning import find_best_hyperparameters # joblib + semaphore를 활용한 whisper 파인 튜닝 시도 
from make_model.final_trainer import fine_tune_best_model
from model_registry.model_registry import model_registry_start
from model_service.service_management import restart_service_with_deploy_model

load_dotenv()
    
# Step 1: Load data from GCS
bucket_name = os.getenv("BUCKET_NAME")
audio_folder = os.getenv("SOURCE_WAV_FOLDER")
text_folder = os.getenv("SOURCE_TEXT_FOLDER")
# data = load_data_from_gcs(bucket_name, audio_folder, text_folder)

# Step 2: Create Hugging Face DatasetDict
# processed_dataset = create_dataset_dict(data)


# Step 3: Load Whisper Model and Processor
#model, processor = load_whisper_model_and_processor()


# Step 4 : raytune를 활용한 Whisper 모델 파인튜닝 시도 
#best_hyperparameters = find_best_hyperparameters(model, processor, processed_dataset)
#print("best_hyperparameters:", best_hyperparameters)


# Step 4 : joblib + semaphore를 활용한 Whisper 모델 파인 튜닝 시도 
#best_params, best_score = find_best_hyperparameters(model, processor, processed_dataset)
#print("Best Hyperparameters :", best_params)
#print("Best Combined Score :", best_score)


# Step 5 : 최적의 하이퍼파라미터 조합을 가진 Whisper 모델 파인 튜닝한 결과물을 로컬에 저장 
# fine_tune_best_model(best_hyperparameters, model, processor, processed_dataset)


# Step 6 : Whisper 파인 튜닝한 모델과 BentoML에 있는 모델을 비교해서 어떤 모델을 서비스할지 결정하는 코드 
# pyenv_python = "/home/kimyw22222/.pyenv/versions/3.11.9/envs/model_registry_env/bin/python"
# model_registry_script = "./model_registry/model_registry.py"

# try:
#     # Subprocess에 데이터 직렬화 후 전달
#     result = subprocess.run(
#         [pyenv_python, model_registry_script],
#         input=pickle.dumps(processed_dataset["validation"]),  # 데이터를 직렬화해서 전달
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         check=True,
#     )

#     # 결과 출력
#     print("Subprocess 결과:", result.stdout.decode())
# except subprocess.CalledProcessError as e:
#     print(f"Subprocess 실행 중 오류 발생: {e.stderr.decode()}")

# Step 7. bentoml model service (기존에 있는 service를 종료하고 다시 service를 띄운다.)
restart_service_with_deploy_model()