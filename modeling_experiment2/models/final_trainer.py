import torch
from transformers import Seq2SeqTrianer, Seq2SeqTrainingArguments

# 사용자 정의 Data Collator
class DataCollatorForWhisper:
    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        input_features = pad_sequence(input_features, batch_first=True, padding_value=0.0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_features": input_features, "labels": labels}

# 최적 파라미터로 모델 재훈련 및 저장
def fine_tune_best_model(best_params, model, processor, processed_dataset):
    """
    Fine-tune the Whisper model using the best hyperparameter combination and save it locally.
    """

    print("Fine-tuning with the best hyperparameters: ", best_params)

    # 훈련 인자 설정
    training_args = Seq2SeqTrainingArguments(
            output_dir="./final_finetuned_model",
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            logging_steps=100,
            per_device_train_batch_size=best_params["batch_size"],
            learning_rate=best_params["learning_rate"],
            lr_scheduler_type="constant",
            warmup_steps=0,
            num_train_epochs=best_params["num_train_epochs"],
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

    save_path = "./final_finetuned_model"
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)

    print(f"Fine-tuning completed. Model saved at '{save_path}'")
