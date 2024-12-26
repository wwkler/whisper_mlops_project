# service_management.py
import bentoml
import subprocess
import signal
import os 

SERVICE_PORT = 3000  # Bentoml 서비스 실행 포트
SERVICE_SCRIPT = "/home/kimyw22222/project/model_authomated_pipeline/model_service/service_up.py"  # Bentoml 서비스 파일 경로

def get_deploy_model():
    """
    현재 BentoML 저장소에서 stage=deploy인 모델을 찾습니다.
    """
    for model in bentoml.models.list():
        if model.info.labels.get("stage") == "deploy":
            return model
    return None


def stop_existing_service():
    """
    기존 Bentoml 서비스를 종료합니다.
    """
    try:
        # `lsof`로 지정된 포트를 사용하는 프로세스를 찾고 종료
        result = subprocess.run(
            ["lsof", "-t", f"-i:{SERVICE_PORT}"],
            stdout=subprocess.PIPE,
            text=True,
        )
        pids = result.stdout.strip().split("\n")

        # 유효한 PID가 있는지 확인
        valid_pids = [pid for pid in pids if pid.strip().isdigit()]

        if valid_pids:
            print(f"Stopping existing service on port {SERVICE_PORT}...")
            for pid in valid_pids:
                os.kill(int(pid), signal.SIGTERM)
            print(f"Service on port {SERVICE_PORT} stopped.")
        else:
            print("No existing service found to stop.")
    except Exception as e:
        print(f"Error stopping existing service: {e}")


def start_new_service(model_tag):
    """
    새로운 BentoML 서비스를 시작합니다.
    """
    print(f"[INFO] Starting new service with model '{model_tag}' on host 0.0.0.0...")

    try:
        process = subprocess.Popen(
            [
                "bentoml", "serve", SERVICE_SCRIPT,
                "--port", str(SERVICE_PORT),
                "--host", "0.0.0.0",  # 외부에서 접속 가능하도록 설정
                "--timeout", "1200"  # 서버 타임아웃 설정 (1200초)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # 텍스트 모드로 출력 읽기
        )
        
        # 실시간으로 로그 출력
        for line in iter(process.stdout.readline, ""):
            if line:
                print(f"[SERVICE LOG]: {line.strip()}")  # 서비스 로그 출력
        
        process.stdout.close()
        process.stderr.close()
        process.wait()  
        # process.wait()  # 프로세스 종료 대기
        
        print(f"[INFO] Service started on port {SERVICE_PORT} and host 0.0.0.0.")
    except Exception as e:
        print(f"[ERROR] Error starting new service: {e}")


def restart_service_with_deploy_model():
    """
    stage=deploy 상태의 모델을 서비스합니다.
    """
    # 1. stage=deploy 모델 가져오기
    deploy_model = get_deploy_model()
    if not deploy_model:
        print("No deploy model found. Unable to start service.")
        return

    # 2. 기존 서비스 종료
    stop_existing_service()

    # 3. 새로운 서비스 시작
    start_new_service(deploy_model.tag)
