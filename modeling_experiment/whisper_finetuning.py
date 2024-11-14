'''
다양한 하이퍼 파라미터 인자들을 조합해서 Whisper Fine Tuning 하는 코드 
'''

import optuna 
import io
import soundfile as sf 
import evaluate

from transformers import WhisperForConditionalGeneration, WhisperProcessor, Trainer, TrainingArguments, DataCollatorWithPadding, BatchFeature

# WER 메트릭 로드
wer_metric = evaluate.load("wer")

class DataCollatorForWhisper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # `input_values`는 오디오 입력을 포함하며, `labels`는 텍스트 라벨을 포함합니다.
        input_features = [feature["input_values"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # 'input_values'를 'BatchFeature'로 감싸서 전달
        batch_features = BatchFeature(data={"input_features" : input_features})
        # padding for input values
        batch = self.processor.feature_extractor.pad(batch_features, padding=True, return_tensors="pt")
        
        # padding for labels
        labels_batch = self.processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"]
        
        return batch

# 평가 지표 계산 함수 
# 수정된 compute_metrics 함수
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # 예측 및 실제 텍스트 생성
    pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # WER 계산
    wer = wer_metric.compute(predictions=pred_texts, references=label_texts)
    
    return {
        "wer" : wer,
    }

    
def fine_tune_whisper(dataset, trial):
    # 하이퍼파라미터 설정 
    # learning_rate = trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True)
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-1])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    num_train_epochs = trial.suggest_categorical("num_train_epochs", [5, 10])
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    # 데이터셋 전처리
    def preprocess_function(batch):
        # GCS에서 업로드한 음성 바이트 데이터를 파일처럼 처리하기 위해 io.BytesIO 사용 
        with io.BytesIO(batch["audio"]) as audio_file:
            audio, _ = sf.read(audio_file, dtype="float32")
            
        # 오디오 데이터를 전처리합니다.
        audio_input = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # 'input_values' 대신 'input_features'로 접근
        audio_input = audio_input.input_features[0]
        
        # 텍스트 데이터를 라벨로 변환
        labels = processor(text=batch["text"], return_tensors="pt").input_ids[0]
        
        batch["input_values"] = audio_input
        batch["labels"] = labels
        
        return batch 
    
    dataset = dataset.map(preprocess_function)
    
    # 데이터셋 크기 확인
    print(f"전체 데이터셋 크기: {len(dataset)}")
    
    # 데이터셋을 학습용과 평가용으로 나누기
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    # 사용자 정의 DataCollator 생성
    data_collator = DataCollatorForWhisper(processor)
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir="optuna_experiment",
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        logging_dir="optuna_experiment/logs",
        logging_steps=10,
        remove_unused_columns=False,  # 필수 열을 유지하도록 설정
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,  # Custom Data Collator 추가
        compute_metrics=compute_metrics,  # 평가 지표 계산 함수 설정
    )
    
    # 모델 학습
    trainer.train()
    
    # 모델 평가 후 검증 손실 반환
    eval_result = trainer.evaluate()
    eval_loss = eval_result["eval_loss"]
    
    print(f"Evaluation Loss : {eval_loss}")
    print(f"Evaluatuin WER : {eval_result['eval_wer']}")
    
    return eval_loss


def objective(trial, dataset):
    print(f"전체 데이터셋 크기 : {len(dataset)}")
    
    eval_loss = fine_tune_whisper(dataset, trial)
    return eval_loss