import torch
import copy
import evaluate
import csv  # CSV 파일 저장을 위한 모듈
import threading

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.nn.utils.rnn import pad_sequence
from itertools import product
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)  # Anomaly Detection 활성화

# 세마포어 설정
max_workers = 5
semaphore = threading.Semaphore(max_workers)


class DataCollatorForWhisper:
     def __call__(self, features):
         input_features = [torch.tensor(f["input_features"]) for f in features]
         labels = [torch.tensor(f["labels"]) for f in features]
         # 텐서 복사
         input_features = pad_sequence(input_features, batch_first=True, padding_value=0.0).clone()
         labels = pad_sequence(labels, batch_first=True, padding_value=-100).clone()
         return {"input_features": input_features, "labels": labels}
    
def train_with_semaphore(params):
     with semaphore:
         return train_function(params)

def train_function(params):
     try:
         model = params["model"]  # 모델 복제
         processor = params["processor"]
         processed_dataset = params["processed_dataset"]  # 데이터 복제

         training_args = Seq2SeqTrainingArguments(
             output_dir="./hyperopt_results",
             eval_strategy="epoch",
             logging_strategy="epoch",
             save_strategy="no",
             eval_steps=1000,
             save_steps=1000,
             logging_steps=100,
             optim="adafactor",  # Adafactor 사용
             per_device_train_batch_size=params["batch_size"],
             learning_rate=params["learning_rate"],
             lr_scheduler_type="constant", # learning_rate 정의한대로 고정
             num_train_epochs=params["num_train_epochs"],
             save_total_limit=1,
             predict_with_generate=True,
             fp16=False,  # FP16 비활성화
             remove_unused_columns=False,
         )

         data_collator = DataCollatorForWhisper()
         trainer = Seq2SeqTrainer(
             model=model,
             args=training_args,
             train_dataset=processed_dataset["train"],
             eval_dataset=processed_dataset["validation"],
             data_collator=data_collator,
         )

         trainer.train()

         predictions = trainer.predict(processed_dataset["validation"])
         decoded_preds = processor.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
         decoded_labels = processor.tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
         wer_metric = evaluate.load("wer")
         cer_metric = evaluate.load("cer")
         wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
         cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
         combined_score = wer + cer

         return {
             "params": params,
             "wer": wer,
             "cer": cer,
             "combined_score": combined_score,
         }
     except Exception as e:
         print(f"Error during training: {e}")
         raise

 # 하이퍼파라미터 탐색
def find_best_hyperparameters(model, processor, processed_dataset):
     learning_rate_choices = [0.001, 0.01]
     batch_size_choices = [16, 32]
     num_train_epoch_choices = [1, 2]

     param_combinations = [
         {
             "learning_rate": lr,
             "batch_size": bs,
             "num_train_epochs": epoch,
             "model": model,
             "processor": processor,
             "processed_dataset": processed_dataset,
         }
         for lr, bs, epoch in product(learning_rate_choices, batch_size_choices, num_train_epoch_choices)
     ]

     # CSV 파일 초기화
     with open("./hyperopt_results.csv", "w", newline="") as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(["learning_rate", "batch_size", "num_train_epochs", "wer", "cer", "combined_score"])

     results = Parallel(n_jobs=-1)(delayed(train_with_semaphore)(param_combination) for param_combination in tqdm(param_combinations))
     for result in results:
         # 결과를 CSV 파일에 추가
         with open("./hyperopt_results.csv", "a", newline="") as csvfile:
             writer = csv.writer(csvfile)
             writer.writerow([
                 result["params"]["learning_rate"],
                 result["params"]["batch_size"],
                 result["params"]["num_train_epochs"],
                 result["wer"],
                 result["cer"],
                 result["combined_score"],
             ])

     best_result = min(results, key=lambda x: x["combined_score"])
     best_params = best_result["params"]
     best_score = best_result["combined_score"]
     return best_params, best_score
