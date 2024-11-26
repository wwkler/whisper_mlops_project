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
        return {"wer": wer}
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
        output_dir="whisper_tiny_experiment",
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_dir="whisper_tiny_experiment/logs",
        logging_steps=100,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False,  # 불필요한 컬럼 제거 방지
        resume_from_checkpoint=True,
        optim="adamw_torch",         # 최적화 알고리즘 ('adamw_torch', 'adamw_apex_fused' 등)
        load_best_model_at_end=True,  # 가장 낮은 평가 손실을 기록한 모델 로드
        metric_for_best_model="wer",  # WER(Word Error Rate) 기준으로 가장 좋은 모델 선택
        greater_is_better=False,      # 낮은 값이 더 나은 성능을 의미 (WER이 낮을수록 좋음)
        save_total_limit=1,               # 최적 모델만 유지
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # 최적의 모델을 명시적으로 저장
    trainer.save_model("whisper_tiny_experiment/best_model")  # 'best_model' 디렉토리에 저장
    
    return trainer.evaluate()["eval_loss"]

def objective(trial, dataset):
    """
    Optuna Objective 함수: Fine-tuning 수행 및 평가
    """
    
    print("Starting data preprocessing...")
    dataset = dataset.map(preprocess_function, batched=True)
    print(f"dataset : {dataset[0]}")  # 데이터의 첫 샘플 출력
    
    print("Splitting dataset into train and test sets...")
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset, eval_dataset = train_test_split["train"], train_test_split["test"]
     
    print("Starting fine-tuning...")
    eval_loss = fine_tune_whisper_tiny(train_dataset, eval_dataset, trial)
    print(f"Evaluation Loss: {eval_loss}")
    
    return eval_loss