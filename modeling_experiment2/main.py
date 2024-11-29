# main.py
import os

from dotenv import load_dotenv
from data.gcs_loader import load_data_from_gcs
from data.dataset_splitter import create_dataset_dict
from models.model_loader import load_whisper_model_and_processor
from models.hyperopt_tuner import tune_hyperparameters # 순차적 하이퍼파라미터 조합 시도
from models.hyperopt_tuner2 import find_best_hyperparameters # 병렬적 하이퍼파라미터 조합 시도
#from models.final_trainer import fine_tune_best_model

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


# Step 4 : Sequential Hyperparameter Tuning
#mapped_best_params = tune_hyperparameters(model, processor, processed_dataset)
#print("Best Parameters:", mapped_best_params)

# Step 4 : Parallel_Hyperparamter Tuning
best_params, best_score = find_best_hyperparameters(model, processor, processed_dataset)
print("Best Hyperparameters :", best_params)
print("Best Combined Score :", best_score)


# Step 5 : Train and Save Final Model
#fine_tune_best_model(best_params, model, processor, processed_dataset)

