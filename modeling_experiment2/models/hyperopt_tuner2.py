import torch.multiprocessing as mp
import evaluate
import torch 
import csv

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.nn.utils.rnn import pad_sequence
from itertools import product
from multiprocessing import Pool,cpu_count


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
def train_function(rank, param_combinations, model, processor, processed_dataset):
    """
    각 워커 프로세스에서 모델을 훈련하고 결과를 저장합니다.
    """

    params = param_combinations[rank] # 현재 프로세스의 하이퍼파라미터 조합 선
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
        fp16=False,
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

    print(f"Process {rank} completed: {params}, Combined Score : {combined_score}")


        
# 최적 하이퍼라미터 탐색 모듈
def find_best_hyperparameters(model, processor, processed_dataset):
    """
    하이퍼파라미터 탐색을 병렬적으로 수행합니다. 
    """

    # 파라미터 인자 정의
    learning_rate_choices = [0.001, 0.01, 0.1]
    batch_size_choices = [16, 32, 64]
    num_train_epoch_choices = [1, 2, 3]


    # 탐색 공간 생성
    param_combinations = [
        {
            "learning_rate" : lr, 
            "batch_size" : bs,
            "num_train_epochs" : epoch,
        }
        for lr, bs, epoch in product(learning_rate_choices, batch_size_choices, num_train_epoch_choices)
    ]


    # CSV 초기화
    with open("./hyperopt_results.csv", "w", newline="") as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(["learning_rate", "batch_size", "num_train_epochs", "wer", "cer", "combined_score"])


    # 병렬 처리 
    processes = []
    for rank in range(len(param_combinations)):
        p = mp.Process(target=train_function, args=(rank, param_combinations, model, processor, processed_dataset))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    # 최적 파라미터 추출
    best_params = None
    best_score = float("inf")
    with open("./hyperopt_results.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            score = float(row["combined_score"])
            if score < best_score:
                best_params = {
                        "learning_rate" : float(row["learning_rate"]),
                        "batch_size" : int(row["batch_size"]),
                        "num_train_epochs" : int(row["num_train_epochs"]),
                }
    print(f"Best Hyperparameters : {best_params}, Best Combined Score : {best_score}")
    return best_params, best_score


