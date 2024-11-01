'''
Google Cloud Storage Bucket에 업로드 되었던 파일들에 대한 메타 데이터를 GCP Pub/Sub에 저장하고 있다가  
파이썬 코드를 실행해서 구독을 해서 그에 대한 메타 데이터를 가져오고 print하게 하는 코드 

20초 기다렸다가 GCP Pub/Sub에 메시지가 없으면 프로그램 종료까지 할 수 있도록 로직 설계
'''

import os
import time
import json 

from dotenv import load_dotenv
from google.cloud import pubsub_v1
from kafka import KafkaProducer 

# .env 파일에서 Google 서비스 계정 key 및 설정 불러오기
# GOOGLE_APPLICATION_CREDENTIALS에 대한 값을 불러온다. 
load_dotenv()

# Pub/Sub 구독 설정
project_id = os.getenv("GOOGLE_PROJECT_ID")  # 프로젝트 ID
subscription_id = os.getenv("GOOGLE_PUB_SUB_SUBSCRIPTIONS_ID")  # 구독 ID

# Kafka 설정
kafka_bootstrap_servers = "10.178.0.2:9092"  # Kafka 서버 주소 (예: localhost:9092)
kafka_topic = "test"  # Kafka 토픽

# Kafka Producer 설정
producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # JSON 직렬화
)

# 메시지 수신 여부를 확인하기 위한 전역 변수
last_message_time = time.time()
timeout = 20  # 20초 동안 새로운 메시지가 없으면 프로그램 종료


# 메시지 처리 콜백 함수
def callback(message):
    global last_message_time
   
    print(f"Received message: {message.data.decode('utf-8')}")
    attributes = message.attributes
    file_data = {
        "file_name": attributes.get('objectId'),
        "bucket": attributes.get('bucketId'),
        "message_data": message.data.decode('utf-8')
    }
    
    # GCS Bucket에 업로드된 .wav 파일만 Kafka로 메시지 전송
    if file_data["file_name"] and file_data["file_name"].endswith(".wav"):
        try:
            producer.send(kafka_topic, value=file_data)
            producer.flush()  # 즉시 전송
            print(f"Sent message to Kafka topic '{kafka_topic}': {file_data}")
            print()
            print()
        except Exception as e:
            print(f"Failed to send message to Kafka: {e}")
    else:
        print(f"File '{file_data['file_name']}' is not a .wav file. Ignored.")

    # 메시지 확인 (acknowledge)하여 재처리 방지
    message.ack()

    # 마지막 메시지를 받은 시간을 갱신
    last_message_time = time.time()

# 메시지 수신 및 종료
def pull_messages_with_timeout(project_id, subscription_id, timeout=10):
    """비동기 Pub/Sub 구독을 통해 실시간 메시지 수신 및 타임아웃으로 종료"""
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    # 실시간 메시지 수신 대기
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}..\n")

    # 메시지 스트림에 타임아웃을 적용하여 종료하는 간단한 방식
    try:
        while True:
            time_since_last_message = time.time() - last_message_time
            if time_since_last_message > timeout:
                print(f"No messages received for {timeout} seconds. Shutting down...")
                streaming_pull_future.cancel()
                break
            time.sleep(1)  # 1초 간격으로 체크
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        print("Subscription terminated manually.")
    except Exception as e:
        streaming_pull_future.cancel()
        print(f"Listening stopped due to exception: {e}")

# 메시지 수신 실행
if __name__ == "__main__":
    pull_messages_with_timeout(project_id, subscription_id, timeout=20)
