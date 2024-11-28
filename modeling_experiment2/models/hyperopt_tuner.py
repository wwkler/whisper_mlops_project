# models/hyperopt_tuner.py

import evaluate
import torch 

from hyperopt import fmin, tpe, hp, Trials
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence

# 사용자 정의 데이터 Collator
class DataCollatorForWhisper:
    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        # Pad input_features and labels
        input_features = pad_sequence(input_features, batch_first=True, padding_value=0.0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored in loss

        return {"input_features": input_features, "labels": labels}


def tune_hyperparameters(model, processor, dataset):
    # Load evaluation metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Define custom DataCollator
    data_collator = DataCollatorForWhisper()


    def objective(params):
        training_args = Seq2SeqTrainingArguments(
                output_dir="./hyperopt_results",
                eval_strategy="steps",
                eval_steps=500,
                save_steps=500,
                logging_steps=100,
                per_device_train_batch_size=params["batch_size"],
                # gradient_accumulation_steps=8 // params["batch_size"] if params["batch_size"] > 0 else 1,
                learning_rate=params["learning_rate"],
                lr_scheduler_type="constant",  # 학습률 고정
                warmup_steps=0,  # Warmup 제거
                num_train_epochs=params["num_train_epochs"],
                save_total_limit=1,
                predict_with_generate=True,
                fp16=True,
                remove_unused_columns=False,) 
        
        trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                processing_class=processor.feature_extractor,
                data_collator=data_collator)  

        trainer.train()

        predictions = trainer.predict(dataset["validation"])
        decoded_preds = processor.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
        decoded_labels = processor.tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        combined_score = wer + cer
        return combined_score

     # Define search space
    learning_rate_choices = [0.001, 0.01, 0.1]
    batch_size_choices = [16, 32, 64]
    num_train_epochs_choices = [1, 2, 3]
    
    search_space = {
        "learning_rate": hp.choice("learning_rate", learning_rate_choices),
        "batch_size": hp.choice("batch_size", batch_size_choices),
        "num_train_epochs": hp.choice("num_train_epochs", num_train_epochs_choices)
    }

    # Hyperparameter tuning
    trials = Trials()
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=1, trials=trials)

     # 매핑된 값으로 변환
    mapped_best_params = {
        "learning_rate": learning_rate_choices[best_params["learning_rate"]],
        "batch_size": batch_size_choices[best_params["batch_size"]],
        "num_train_epochs": num_train_epochs_choices[best_params["num_train_epochs"]]
    }

    print(f"Mapped best_params : {mapped_best_params}")

    return mapped_best_params
                    
