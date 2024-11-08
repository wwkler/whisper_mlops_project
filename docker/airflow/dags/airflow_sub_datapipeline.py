import os 

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    'owner': 'airflow',
    'retries': 1,  # 실패 시 재시도 횟수
    'retry_delay': timedelta(minutes=5),  # 재시도 간격
}

with DAG(
    'airflow_sub_datapipeline',
    default_args=default_args,
    description='Week Batch Activate Sunday 00:00 AM',
    schedule_interval='0 0 * * 0',  # 매주 일요일 새벽 0시 실행 -> 일주일에 1번씩 실행 
    start_date=datetime(2023, 1, 1),  # 시작 날짜
    catchup=False,  # 과거의 누락된 작업 실행 안 함 
) as dag:
    
    airflow_service_text_to_bigquery = DockerOperator(
        task_id='airflow_service_text_to_bigquery',
        image='datapipeline:v2.0',  # 실행할 Docker 이미지
        api_version='auto', # DockerOperator가 Docker Demon과 통신할 떄 사용할 API 버전 명시 
        auto_remove=True, # 컨테이너가 실행된 후 docker container ls -a 했을 떄 작업 이력이 남지 않도록 한다.
        command='python /home/kimyw22222/project/datapipeline/service_voice_text_to_bigquery/service_voice_text_to_bigquery.py',  # Python 파일 실행
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge'
    )
    
    airflow_service_text_to_bigquery
    
    