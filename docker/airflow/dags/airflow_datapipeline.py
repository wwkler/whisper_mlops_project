import os 

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'retries': 1,  # 실패 시 재시도 횟수
    'retry_delay': timedelta(minutes=5),  # 재시도 간격
}

with DAG(
    'airflow_datapipeline',
    default_args=default_args,
    description='A daily batch job DAG',
    schedule_interval='0 0 * * *',  # 매일 새벽 0시 실행
    start_date=datetime(2023, 1, 1),  # 시작 날짜
    catchup=False,  # 과거의 누락된 작업 실행 안 함
    params={'text_file_path': '/home/airflow/metadata.txt'}  # Airflow 컨테이너 상 경로 정의 
) as dag:
    
    # 텍스트 파일에 대한 사이즈를 확인하는 함수
    def check_file_size(**context):
        """파일 크기를 확인하고 분기"""
        file_path = context["params"]["text_file_path"]  # Airflow 컨테이너 상 경로 
        if os.path.getsize(file_path) == 0:
            print("Metadata file is empty. Skipping all downstream tasks.")
            return "airflow_skip_all_tasks"
        else:
            return "airflow_split_voice_match_text"
        
    # 텍스트 파일에 대한 사이즈가 0이여서 TASK가 끝났다는 함수 
    def skip_message():
        """스킵 메시지 출력"""
        print("파일 사이즈가 0이어서 Task가 끝났습니다.")

    
    # Kafka로부터 Consume해서 메타 데이터를 전처리한 다음 텍스트 파일에 저장하는 TASK
    airflow_kafka_consume_to_txt = DockerOperator(
        task_id='airflow_kafka_consume_to_txt',
        image='datapipeline:v2.0',  # 실행할 Docker 이미지
        api_version='auto', # DockerOperator가 Docker Demon과 통신할 떄 사용할 API 버전 명시 
        auto_remove=True, # 컨테이너가 실행된 후 docker container ls -a 했을 떄 작업 이력이 남지 않도록 한다.
        command='python /home/kimyw22222/project/datapipeline/kafka_consume_to_txt/kafka_consume_to_txt.py',  # Python 파일 실행
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            Mount(
                source ="/home/kimyw22222/project/docker/datapipeline/split_voice_match_text/metadata.txt", # 리눅스 호스트 상 경로 
                target="/home/kimyw22222/project/datapipeline/split_voice_match_text/metadata.txt", # 파이썬 컨테이너 상 경로 
                type="bind")
        ], # 리눅스 호스트 에있는 txt 파일을 Python 컨테이너에 Mount
    )
    
    # 텍스트 파일에 대한 사이즈를 확인하는 TASK
    airflow_check_file_task = BranchPythonOperator(
        task_id="airflow_check_file_task",
        python_callable=check_file_size,
        provide_context=True
    )
    
    # 파일이 비어있을 때 종료 메시지를 출력하는 TASK
    airflow_skip_all_tasks = PythonOperator(
        task_id="airflow_skip_all_tasks",
        python_callable=skip_message
    )
    
    # split된 음성과 텍스트 파일을 매칭하여 Google Cloud Stroage Bucket에 저장하는 TASK
    airflow_split_voice_match_text = DockerOperator(
        task_id='airflow_split_voice_match_text',
        image='datapipeline:v2.0',  # 실행할 Docker 이미지
        api_version='auto', # DockerOperator가 Docker Demon과 통신할 떄 사용할 API 버전 명시 
        auto_remove=True, # 컨테이너가 실행된 후 docker container ls -a 했을 떄 작업 이력이 남지 않도록 한다.
        command='python /home/kimyw22222/project/datapipeline/split_voice_match_text/main_async.py',  # Python 파일 실행
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            Mount(
                source ="/home/kimyw22222/project/docker/datapipeline/split_voice_match_text/metadata.txt", # 리눅스 호스트 상 경로 
                target="/home/kimyw22222/project/datapipeline/split_voice_match_text/metadata.txt", # 파이썬 컨테이너 상 경로 
                type="bind")
        ], # 리눅스 호스트 에있는 txt 파일을 Python 컨테이너에 Mount
    )
    
    # Google Cloud Storage에 split된 음성 분량이 50시간 이상아면 다른 Bucket에 mv하여 저장하는 TASK
    airflow_check_voice_time_from_gcs = DockerOperator(
        task_id='airflow_check_voice_time_from_gcs',
        image='datapipeline:v2.0',  # 실행할 Docker 이미지
        api_version='auto', # DockerOperator가 Docker Demon과 통신할 떄 사용할 API 버전 명시 
        auto_remove=True, # 컨테이너가 실행된 후 docker container ls -a 했을 떄 작업 이력이 남지 않도록 한다.
        command='python /home/kimyw22222/project/datapipeline/check_voice_time_from_gcs/check_voice_time_from_gcs.py',  # Python 파일 실행
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
    )
    
    # github action을 trigger 할지 말지 결정하는 함수
    def is_trigger_github_actions(**kwargs):
        # XCom에서 main 함수의 결과를 가져옴
        decision = kwargs['ti'].xcom_pull(task_ids='airflow_check_voice_time_from_gcs')
        
        print(f"decision : {decision}")
        
        return decision  # 'airflow_trigger_github_actions' 또는 'airflow_end_task'로 반환되어 분기됨
    
    # Github Actions를 trigger 할지 결정하는 TASK
    airflow_is_trigger_github_actions = BranchPythonOperator(
        task_id='airflow_is_trigger_github_actions',
        python_callable=is_trigger_github_actions,
        provide_context=True
    )
        
    # GitHub Actions 트리거 태스크
    airflow_trigger_github_actions = DockerOperator(
        task_id='airflow_trigger_github_actions',
        image='datapipeline:v2.0',
        api_version='auto',
        auto_remove=True,
        command="python /home/kimyw22222/project/datapipeline/trigger_github_actions/trigger_github_actions.py",  # GitHub Actions 트리거를 위한 스크립트
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge"
    )
    
    # TASK가 끝다는 것을 알려주는 함수
    def end_task_message():
        """마지막 TASK 메시지 """
        print("마지막 Task가 끝났습니다.")
    
    # TASK가 끝다는 것을 알려주는 TASK
    airflow_end_task = PythonOperator(
        task_id="airflow_end_task",
        python_callable=end_task_message
    )
    
    # Airflow 진행 흐름
    airflow_kafka_consume_to_txt >> airflow_check_file_task >> [airflow_split_voice_match_text, airflow_skip_all_tasks]
    airflow_split_voice_match_text >> airflow_check_voice_time_from_gcs >> airflow_is_trigger_github_actions >> [airflow_trigger_github_actions, airflow_end_task]
    
    
    