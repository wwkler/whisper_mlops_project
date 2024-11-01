import logging
import json
import base64
import os

from flask import Flask, request
from kafka import KafkaProducer

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka Producer 설정
producer = KafkaProducer(
    bootstrap_servers=['10.178.0.2:9092'], # Google VPC Connector로 Private IP로 접근 가능 
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Pub/Sub 메시지 수신 및 Kafka로 전송
@app.route('/', methods=['POST'])
def handle_pubsub_message():
    logger.info("handle_pubsub_message가 호출되었습니다.")
    
    # Pub/Sub 메시지 JSON 데이터를 가져옵니다
    envelope = request.get_json()
    
    # objectId가 .wav로 끝나는 경우에만 Kafka로 전송
    object_id = envelope["message"]["attributes"].get("objectId")
    if object_id and object_id.endswith('.wav'):
        # Kafka로 envelope 전체를 전송
        producer.send('test', value=envelope)
        producer.flush()
        logger.info(f"Sent .wav file message to Kafka: {object_id}")
    else:
        logger.info(f"File {object_id} is not a .wav file; skipping Kafka send.")
    
    return 'Message processed', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
