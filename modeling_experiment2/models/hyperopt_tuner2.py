import evaluate
import torch 
import csv

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.nn.utils.rnn import pad_sequence
from itertools import product
from multiprocessing import Pool


# 사용자 정의 Data Collator
class DataCollatorForWhisper:
    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        # Pad input_features and labels
        input_features = pad_sequence(input_features, batch_first=True, padding_value=0.0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_features" : input_features, "labels" : labels}


# 훈련 함수 정의
def train_function(params):
    model, processor, processed_dataset = params["model"], params["processor"], params["processed_dataset"]
    training_args = Seq2SeqTrainingArguments(
        output_dir="./hyperopt_results",
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        lr_scheduler_type="constant",
        warmup_steps=0,
        num_train_epochs=params["num_train_epochs"],
        save_total_limit=1,
        predict_with_generate=True,
        fp16=True,
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

    # CSV 저장
    with open("./hyperopt_results.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            params["learning_rate"], params["batch_size"], params["num_train_epochs"], wer, cer, combined_score])

    return combined_score



# 병렬 실행 함수
def parallel_train(params):
    try:
        score= train_function(params)
        return params, score
    except Exception as e:
        return params, str(e) 

        
# 최적 하이퍼라미터 탐색 모듈
def find_best_hyperparameters(model, processor, processed_dataset):
    """
    Find the best hyperparameter combination using multiprocessing.
    """


    learning_rate_choices = [0.001, 0.01, 0.1]
    batch_size_choices = [16, 32, 64]
    num_train_epoch_choices = [1, 2, 3]


    # 탐색 공간 생성
    param_combinations = [
        {
            "learning_rate" : lr, 
            "batch_size" : bs,
            "num_train_epochs" : epoch,
            "model" : model, 
            "processor" : processor,
            "processed_dataset" : processed_dataset,
        }
        for lr, bs, epoch in product(learning_rate_choices, batch_size_choices, num_train_epoch_choices)
    ]


    # CSV 초기화
    with open("./hyperopt_results.csv", "w", newline="") as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(["learning_rate", "batch_size", "num_train_epochs", "wer", "cer", "combined_score"])


    # 멀티 프로세싱
    num_processes = 3 # 병렬 프로세스 수 (CPU 코어 수에 따라 조정)
    with Pool(processes=num_processes) as pool:
         results = pool.map(parallel_train, param_combinations)

    
    # 최적 파라미터 탐색
    best_params = None
    best_score = float("inf")
    for params, score in results:
        if isinstance(score, float) and score < best_score:
            best_score = score
            best_params = {
                    "learning_rate" : params["learning_rate"],
                    "batch_size" : params["batch_size"],
                    "num_train_epochs" : params["num_train_epochs"]
            }

    return best_params, best_score


