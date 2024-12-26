# 음성 파일을 가지고 API 요청 (테스트)
import requests

# API 엔드포인트
url = "http://localhost:3000/transcribe"

# 로컬 음성 파일 경로
audio_file_path = "/home/kimyw22222/project/model_authomated_pipeline/model_service/2_2587G2A5_2586G2A5_T2_2D08T0348C000034_010922.wav"

# 요청 보내기
with open(audio_file_path, "rb") as audio_file:
    files = {"audio_file": audio_file}
    response = requests.post(url, files=files, timeout=1200)

# 결과 출력
if response.status_code == 200:
    # print("Transcription : ", response.json())
    print("Transcription:", response.json()["transcription"])
else:
    print("Error:", response.status_code, response.text)
