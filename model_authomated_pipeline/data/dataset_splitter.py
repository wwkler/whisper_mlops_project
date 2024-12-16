# data/dataset_splitter.py
# data/dataset_splitter.py
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import WhisperProcessor


def prepare_and_split_dataset(data, test_size=0.2):
    """
    Split data into training and validation datasets.
    """
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def preprocess_for_whisper(data):
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    """
    Preprocess the dataset for Whisper model by converting audio and transcription into
    input_features and labels.
    """
    def preprocess_function(examples):
        # Extract audio waveform and sampling rate
        waveform = examples["audio"]["audio"]
        sample_rate = examples["audio"]["sample_rate"]

        # Convert audio to log-Mel spectrogram
        input_features = processor.feature_extractor(
            waveform, sampling_rate=sample_rate, return_tensors="np"
        ).input_features[0]

        # Tokenize transcription
        labels = processor.tokenizer(
            examples["transcription"], return_tensors="np", padding="longest"
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}

    # Apply preprocessing
    preprocessed_data = data.map(preprocess_function, remove_columns=["audio", "transcription"])
    return preprocessed_data


def create_dataset_dict(data):
    """
    Create a DatasetDict for training and validation with proper preprocessing for Whisper model.
    """
    # Split data into training and validation sets
    train_dataset, val_dataset = prepare_and_split_dataset(data)

    # Preprocess the datasets
    train_dataset = preprocess_for_whisper(train_dataset)
    val_dataset = preprocess_for_whisper(val_dataset)

    # Combine into a DatasetDict
    splitted_dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    print(f"splitted_dataset : {splitted_dataset}")

    return splitted_dataset
