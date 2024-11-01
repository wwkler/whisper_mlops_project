# Python 3.11.9 slim 버전 사용
FROM python:3.11.9-slim

# 작업 디렉토리 설정
WORKDIR /home/kimyw22222

# requirements.txt 복사 및 설치
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 파일 복사
COPY ./check_voice_time_from_gcs /home/kimyw22222/project/datapipeline/check_voice_time_from_gcs
COPY ./gcs_subscribe_to_cloud_run_to_kafka /home/kimyw22222/project/datapipeline/gcs_subscribe_to_cloud_run_to_kafka
COPY ./gcs_subscribe_to_kafka /home/kimyw22222/project/datapipeline/gcs_subscribe_to_kafka
COPY ./kafka_consume_to_txt /home/kimyw22222/project/datapipeline/kafka_consume_to_txt
COPY ./service_voice_text_to_bigquery /home/kimyw22222/project/datapipeline/service_voice_text_to_bigquery
COPY ./split_voice_match_text /home/kimyw22222/project/datapipeline/split_voice_match_text
COPY ./trigger_github_actions /home/kimyw22222/project/datapipeline/trigger_github_actions
COPY ./google_service_account.json /home/kimyw22222/google_service_account.json 

