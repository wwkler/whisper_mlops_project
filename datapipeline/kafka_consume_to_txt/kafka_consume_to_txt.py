'''
Kafka에 들어간 음성 파일에 대한 메타 데이터를 Consuming해서  후처리하는 코드 

Kafka Broker Topic에 있는 데이터를 다 가져왔으면 몇초 기다렸다가 프로그램을 종료하도록 설계 
'''
import time
import json

from confluent_kafka import Consumer, KafkaException, KafkaError

# Kafka Consumer 설정
conf = {
    'bootstrap.servers': 'localhost:9092',  # Kafka 브로커 주소
    'group.id': 'my-consumer-group',        # Consumer 그룹 ID
    'auto.offset.reset': 'earliest',        # 처음부터 읽기 옵션 설정
}

consumer = Consumer(conf)

# 전체 메시지를 하나의 JSON 리스트로 저장하고 텍스트 파일에 추가하는 함수
def write_text_file(messages, file_name="/home/kimyw22222/project/split_voice_match_text/metadata.txt"):
    with open(file_name, 'a') as f:
        f.write(json.dumps(messages, indent=4, ensure_ascii=False))  # 전체 리스트를 JSON 형식으로 쓰기

# 메시지 소비 함수
def consume_messages(topic, timeout=30):
    messages = []  # 전체 메시지를 저장할 리스트
    
    try:
        consumer.subscribe([topic])  # 구독할 Kafka 토픽 설정
        partition_eof_count = 0      # 파티션 끝에 도달한 횟수 추적
        last_message_time = time.time()  # 마지막 메시지 수신 시간
        timeout_sec = timeout            # 타임아웃 설정 (초 단위)
        
        # 파티션 정보 가져오기
        partitions = consumer.list_topics(topic).topics[topic].partitions
        total_partitions = len(partitions)

        print(f"Total partitions for topic '{topic}': {total_partitions}")

        while True:
            msg = consumer.poll(timeout=1.0)  # 메시지 가져오기 (1초 대기)
            
            # 타임아웃 확인: 마지막 메시지를 받은 후로 일정 시간이 지났다면 종료
            if time.time() - last_message_time > timeout_sec:
                print(f"No new messages for {timeout_sec} seconds. Exiting...")
                break
            
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # 파티션 끝에 도달한 경우
                    partition_eof_count += 1
                    print(f"End of partition reached {msg.topic()}/{msg.partition()}")

                    # 모든 파티션 끝에 도달하면 종료
                    if partition_eof_count >= total_partitions:
                        print("All partitions have been fully consumed. Exiting...")
                        break
                else:
                    raise KafkaException(msg.error())
            else:
                # 정상적으로 메시지를 수신한 경우
                message = json.loads(msg.value().decode('utf-8'))
                print(f"Received message: {message}" + "\n")
                
                attributes = message.get("message", {}).get("attributes", {})
                bucket_id = attributes.get("bucketId")
                object_id = attributes.get("objectId")

                # bucketId와 objectId가 존재하는 경우 리스트에 추가
                if bucket_id and object_id:
                    messages.append({"bucketId": bucket_id, "objectId": object_id})

                last_message_time = time.time()  # 마지막 메시지 수신 시간 업데이트

    except KeyboardInterrupt:
        print("Consumer interrupted manually.")
    finally:
        # 메시지 소비가 끝난 후 파일에 기록
        if messages:  # 메시지가 있을 때만 파일 기록
            write_text_file(messages)
        
        # 종료 시 Consumer 닫기
        consumer.close()

if __name__ == '__main__':
    consume_messages(topic='test', timeout=30)  # 원하는 Kafka 토픽 이름과 타임아웃 설정
