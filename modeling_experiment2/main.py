# main.py
import os 

from dotenv import load_dotenv
from data.gcs_loader import load_data_from_gcs
from data.dataset_splitter import create_dataset_dict
from models.model_loader import load_whisper_model_and_processor
from models.hyperopt_tuner import tune_hyperparameters
#from models.final_trainer import train_and_save_final_model


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


# Step 4 : Hyperparameter Tuning
mapped_best_params = tune_hyperparameters(model, processor, processed_dataset)
print("Best Parameters:", mapped_best_params)


# Step 5 : Train and Save Final Model
#train_and_save_final_model(best_params, model, processor, processed_dataset)

