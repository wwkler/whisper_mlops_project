'''
다양한 하이퍼 파라미터 인자들을 조합해서 Whisper Fine Tuning 하는 코드 
'''

import os
import io
import torch
import soundfile as sf
import numpy as np

from datasets import DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
from optuna import Trial
from evaluate import load as load_metric


# 환경 변수 로드
load_dotenv()

# Whisper Processor 및 WER 메트릭 초기화
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

def preprocess_function(batch):
    processed_batch = {"input_features": [], "labels": []}
    required_length = 3000  # Whisper 모델이 요구하는 고정 길이

    for audio_bytes, text in zip(batch["audio"], batch["text"]):
        try:
            if not audio_bytes or not text:
                continue

            with io.BytesIO(audio_bytes) as audio_file:
                audio, _ = sf.read(audio_file, dtype="float32")

            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
            input_features = inputs.input_features.squeeze(0)

            # Trim or pad input_features to max_length
            if input_features.shape[1] > required_length:
                input_features = input_features[:, :required_length]
            elif input_features.shape[1] < required_length:
                pad_size = required_length - input_features.shape[1]
                input_features = torch.nn.functional.pad(input_features, (0, pad_size), value=0)

            labels = processor(text=text, return_tensors="pt", padding=True).input_ids.squeeze(0)

            # Append as tensors
            processed_batch["input_features"].append(input_features)
            processed_batch["labels"].append(labels)
        except Exception as e:
            print(f"Error processing a sample: {e}")
            continue

    # Convert labels to tensors if they are not already
    processed_batch["labels"] = [
        torch.tensor(label) if not isinstance(label, torch.Tensor) else label
        for label in processed_batch["labels"]
    ]

    return processed_batch


def data_collator(features):
    """
    데이터 Collator: Spectrogram 및 라벨을 패딩
    """
    if not features:
        raise ValueError("Empty batch received by data_collator!")

    try:
        # Ensure input_features are tensors
        input_features = torch.stack([
            torch.tensor(f["input_features"]) if isinstance(f["input_features"], list) else f["input_features"]
            for f in features
        ])  # (batch_size, 80, max_length)

        # Ensure labels are tensors and pad them
        labels = pad_sequence(
            [torch.tensor(f["labels"]) if isinstance(f["labels"], list) else f["labels"] for f in features],
            batch_first=True,
            padding_value=-100,
        )
    except Exception as e:
        print(f"Error in data_collator: {e}")
        raise

    return {"input_features": input_features, "labels": labels}


def compute_metrics(pred):
    """
    평가 메트릭 계산 함수
    """
    try:
        # Handle tuple case
        if isinstance(pred.predictions, tuple):
            predictions = pred.predictions[0]  # Use the first element of the tuple
        else:
            predictions = pred.predictions

        pred_ids = predictions.argmax(-1)  # Compute argmax for token-level predictions
        pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = processor.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_texts, references=label_texts)
        cer = cer_metric.compute(predictions=pred_texts, references=label_texts)

        return {"wer": wer, "cer" : cer}
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        raise



def fine_tune_whisper_tiny(train_dataset, eval_dataset, trial=None):
    """
    Whisper Tiny 모델 Fine-Tuning 함수
    """
    # 하이퍼파라미터 설정
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.01, 0.1])
    batch_size = trial.suggest_categorical("batch_size", [16, 32]) 
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    training_args = TrainingArguments(
        output_dir=f"whisper_tiny_experiment/trial_{trial.number}",
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="no", # 개별 trial 동안은 모델을 저장하지 않음 
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False,  # 불필요한 컬럼 제거 방지
        resume_from_checkpoint=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    return trainer, eval_results 

# Optuna Objective 함수 
def objective(trial, dataset):
    """
    Optuna의 각 trial에서 모델을 학습하고 평가하여 최적의 모델을 추적합니다.
    """
    
    global best_metrics, best_model_dir


    print("Starting data preprocessing...")
    dataset = dataset.map(preprocess_function, batched=True)
    
    print("Splitting dataset into train and test sets...")
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset, eval_dataset = train_test_split["train"], train_test_split["test"]
     
    print("Starting fine-tuning...")
    trainer, eval_results = fine_tune_whisper_tiny(train_dataset, eval_dataset, trial)
    print(f"eval_results : {eval_results}")

    wer, cer = eval_results["wer"], eval_results["cer"]


    # 최적의 모델 갱신 (wer + cer 값이 작을 수록 최적의 모델로 판단)
    combined_metric = wer + cer 

    if best_metrics is None or combined_metric < best_metrics["combined"]:
        best_metrics = {"wer" : wer, "cer" : cer, "combined" : combined_metric}
        best_model_dir = "whisper_tiny_experiment/best_model"
        trainer.save_model(best_model_dir)
        print(f"New best model saved with WER: {wer}, CER: {cer}")

    return combined_metric
