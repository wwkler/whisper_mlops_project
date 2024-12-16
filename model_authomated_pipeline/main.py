# main.py
import os

from dotenv import load_dotenv
from data.gcs_loader import load_data_from_gcs
from data.dataset_splitter import create_dataset_dict
from models.model_loader import load_whisper_model_and_processor
from models.raytune_finetuning import find_best_hyperparameters # raytune를 활용한 whisper 파인 튜닝 시도 
#from models.joblib_finetuning import find_best_hyperparameters # joblib + semaphore를 활용한 whisper 파인 튜닝 시도 
from models.final_trainer import fine_tune_best_model

load_dotenv()
    
# Step 1: Load data from GCS
bucket_name = os.getenv("BUCKET_NAME")
audio_folder = os.getenv("SOURCE_WAV_FOLDER")
text_folder = os.getenv("SOURCE_TEXT_FOLDER")
data = load_data_from_gcs(bucket_name, audio_folder, text_folder)

# Step 2: Create Hugging Face DatasetDict
processed_dataset = create_dataset_dict(data)


# Step 3: Load Whisper Model and Processor
model, processor = load_whisper_model_and_processor()


# Step 4 : raytune를 활용한 Whisper 모델 파인튜닝 시도 
best_hyperparameters = find_best_hyperparameters(model, processor, processed_dataset)
print("best_hyperparameters:", best_hyperparameters)

# Step 4 : joblib + semaphore를 활용한 Whisper 모델 파인 튜닝 시도 
#best_params, best_score = find_best_hyperparameters(model, processor, processed_dataset)
#print("Best Hyperparameters :", best_params)
#print("Best Combined Score :", best_score)


# step 5 : Train and Save Final Model
fine_tune_best_model(best_hyperparameters, model, processor, processed_dataset)

