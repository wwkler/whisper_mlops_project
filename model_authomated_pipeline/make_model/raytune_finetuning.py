'''
ray tune를 활용해서 다양한 하이퍼 파라미터 조합으로 Whisper 모델 파인 튜닝 실행 
'''
import evaluate
import torch
import csv
import ray 

from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


# 사용자 정의 데이터 Collator
class DataCollatorForWhisper:
    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        # Pad input_features and labels
        input_features = pad_sequence(input_features, batch_first=True, padding_value=0.0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored in loss

        return {"input_features": input_features, "labels": labels}


# 모델 학습 함수
def train_function(config, model=None, processor=None, dataset=None, csv_file=None):
    # Load evaluation metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./ray_results",
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=int(config["batch_size"]),
        learning_rate=config["learning_rate"],
        lr_scheduler_type="constant",
        warmup_steps=0,
        num_train_epochs=int(config["num_train_epochs"]),
        save_total_limit=1,
        predict_with_generate=True,
        fp16=False,
        remove_unused_columns=False,
    )

    # Define DataCollator
    data_collator = DataCollatorForWhisper()

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Evaluate model
    predictions = trainer.predict(dataset["validation"])
    decoded_preds = processor.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    combined_score = wer + cer

    # Log results to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([config["learning_rate"], config["batch_size"], config["num_train_epochs"], wer, cer, combined_score])

    # Report to Ray Tune
    session.report({"wer": wer, "cer": cer, "combined_score": combined_score})


# 하이퍼파라미터 튜닝 함수
def find_best_hyperparameters(model, processor, dataset):
    # Ray 초기화
    ray.init()
    
    # Initialize CSV with header
    csv_file = "/home/kimyw22222/project/model_authomated_pipeline/ray_tune_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["learning_rate", "batch_size", "num_train_epochs", "wer", "cer", "combined_score"])

    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(0.0001, 0.1),
        "batch_size": tune.choice([16, 32, 64]),  # Categorical options
        "num_train_epochs": tune.choice([1, 2, 3]),  # Categorical options
    }

    # Define OptunaSearch for search
    search_alg = OptunaSearch(metric="combined_score", mode="min")

    # Define Scheduler (ASHAScheduler)
    scheduler = ASHAScheduler(
        max_t=2,  # Maximum epochs
        grace_period=1,  # Minimum trials before stopping
        reduction_factor=4,  # 상위 1/4에 해당하는 것만 통과 
    )

    # Run Ray Tune
    analysis = tune.run(
        tune.with_parameters(train_function, model=model, processor=processor, dataset=dataset, csv_file=csv_file),
        config=search_space,
        search_alg=search_alg,
        scheduler=scheduler,
        metric="combined_score",
        mode="min",
        resources_per_trial={"cpu": 12},  # Adjust resources per trial
        storage_path=str(Path("./ray_tune_results").absolute()),  # Absolute path for storage
        num_samples=5,  # Number of samples
    )

    # Get best hyperparameters
    best_hyperparameters = analysis.get_best_config(metric="combined_score", mode="min")
    print(f"best_hyperparameters: {best_hyperparameters}")

    return best_hyperparameters
