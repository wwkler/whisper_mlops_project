import pickle
import sys
import os 
import shutil
import bentoml
import evaluate
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 기존 모델 및 새로운 모델의 저장 경로
BEST_MODEL_PATH = "./final_finetuned_model"
BENTO_MODEL_TAG = "whisper_best_model"  # BentoML에 저장될 모델 이름

# 성능 평가 메트릭 불러오기
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def load_model_and_processor(model_path):
    """
    모델과 프로세서를 로드합니다.
    """
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    return model, processor


def evaluate_model(model, processor, eval_dataset):
    """
    모델의 성능을 평가하고 WER와 CER을 계산합니다.
    """
    model.eval()
    predictions = []
    references = []

    # 모델 평가
    for batch in eval_dataset:
        input_features = torch.tensor(batch["input_features"]).unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            predicted_ids = model.generate(input_features=input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predictions.append(transcription[0])  # 첫 번째 배치 결과만 추가
    
        if "labels" in batch:
            reference_text = processor.batch_decode([batch["labels"]], skip_special_tokens=True)
            references.append(reference_text[0])  # 첫 번째 레이블 결과 추가

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    combined_score = wer + cer
    return {"WER": wer, "CER": cer, "combined_score": combined_score}


def get_deploy_model():
    """
    현재 BentoML 저장소에서 stage=deploy인 모델을 찾습니다.
    """
    for model in bentoml.models.list():
        if model.info.labels.get("stage") == "deploy":
            return model
    return None


def update_model_stage(model_tag, new_stage):
    """
    기존 deploy 모델을 archived 상태로 전환한다.
    """
    # 1. 기존 BentoML 모델 로드
    bento_model = bentoml.models.get(model_tag)
    model_path = bento_model.path

    # 2. 모델 가중치 파일 경로 확인 및 로드
    saved_model_file = os.path.join(model_path, "saved_model.pt")
    if not os.path.exists(saved_model_file):
        raise FileNotFoundError(f"Saved model file not found: {saved_model_file}")

    # 3. 저장된 모델의 상태를 확인하여 올바르게 로드
    loaded_data = torch.load(saved_model_file, map_location="cpu")
    if isinstance(loaded_data, dict):
        # state_dict로 저장된 경우
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.load_state_dict(loaded_data)
    elif isinstance(loaded_data, WhisperForConditionalGeneration):
        # 모델 객체 자체가 저장된 경우
        model = loaded_data
    else:
        raise TypeError(f"Unexpected type for loaded data: {type(loaded_data)}")

    print("Loaded model weights successfully.")

    # 4. 프로세서 로드
    processor = WhisperProcessor.from_pretrained(f"{model_path}/processor")

    # 5. 메타데이터와 레이블 업데이트
    new_metadata = bento_model.info.metadata.copy()
    new_metadata["stage"] = new_stage

    new_labels = bento_model.info.labels.copy()
    new_labels["stage"] = new_stage

    # 6. 모델을 다시 저장하여 메타데이터와 레이블 업데이트
    updated_model = bentoml.pytorch.save_model(
        name=bento_model.info.name,
        model=model,
        signatures={"generate": {"batchable": True}},
        metadata=new_metadata,
        labels=new_labels,
    )

    # 7. 프로세서 저장
    processor.save_pretrained(f"{updated_model.path}/processor")
    
    # 8. 기존 deploy 모델 삭제
    bentoml.models.delete(model_tag)
    print(f"Deleted original model '{model_tag}'.")

    print(f"Updated model '{updated_model.tag}' with stage '{new_stage}'")


def save_model_with_stage(model, processor, eval_results, stage="archived"):
    """
    BentoML에 모델을 저장하고 stage 메타데이터를 추가합니다.
    """
    
    # Whisper 모델을 Transformer 형식으로 저장
    final_finetuned_model_dir = "/home/kimyw22222/project/model_authomated_pipeline/final_finetuned_model"

    bentoml_model = bentoml.pytorch.save_model(
        name=BENTO_MODEL_TAG,
        model=model,
        signatures={"generate": {"batchable": True}},
        metadata={
            "wer": eval_results["WER"],
            "cer": eval_results["CER"],
            "combined_score": eval_results["combined_score"],
            "stage": stage,
        },
        labels={"stage": stage, "framework": "transformers"},
    )
    
    bentoml_model_path = bentoml_model.path
    for file_name in os.listdir(final_finetuned_model_dir):
        src_file = os.path.join(final_finetuned_model_dir, file_name)
        dst_file = os.path.join(bentoml_model_path, file_name)
        shutil.copy2(src_file, dst_file)
        
        
    print(f"Model saved with stage '{stage}': {bentoml_model.tag}")
    return bentoml_model


def update_best_model_and_save(model, processor, eval_results):
    """
    기존 deploy 모델과 비교하여 더 나은 경우 업데이트.
    """
    # 현재 stage=deploy 모델 불러오기
    deploy_model = get_deploy_model()
    new_model_score = eval_results["combined_score"]

    if deploy_model:
        deploy_model_score = deploy_model.info.metadata["combined_score"]
        print(f"Existing deploy model score: {deploy_model_score}")
        print(f"New model score: {new_model_score}")

        # 새로운 모델이 더 나은 경우
        if new_model_score < deploy_model_score:
            print("New model has better performance. Updating to stage=deploy...")
            # 기존 deploy 모델을 archived로 업데이트
            update_model_stage(deploy_model.tag, "archived")
            # 새 모델을 stage=deploy로 저장
            save_model_with_stage(model, processor, eval_results, stage="deploy")
        else:
            print("New model is worse. Saving to stage=archived...")
            save_model_with_stage(model, processor, eval_results, stage="archived")
    else:
        # deploy 모델이 없는 경우 새 모델을 deploy로 저장
        print("No existing deploy model found. Saving new model as stage=deploy...")
        save_model_with_stage(model, processor, eval_results, stage="deploy")


def model_registry_start(eval_dataset):
    """
    전체 모델 레지스트리 관리 프로세스.
    """
    # 1. 모델 및 프로세서 로드
    model, processor = load_model_and_processor(BEST_MODEL_PATH)

    # 2. 모델 평가
    eval_results = evaluate_model(model, processor, eval_dataset)
    print(f"Evaluation Results: {eval_results}")

    # 3. 기존 모델과 비교 및 저장
    update_best_model_and_save(model, processor, eval_results)
    
   
if __name__ == "__main__":
    # Step 4: 표준 입력으로 데이터 수신
    input_data = sys.stdin.buffer.read()
    eval_dataset = pickle.loads(input_data)  # 데이터를 역직렬화

    # 모델 레지스트리 시작
    model_registry_start(eval_dataset)