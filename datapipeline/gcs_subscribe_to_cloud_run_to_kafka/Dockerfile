# Python 이미지
FROM python:3.9-slim

# 작업 디렉터리 설정
WORKDIR /app
COPY . /app

# 필요한 패키지 설치
RUN pip install -r requirements.txt

# 8080번 포트를 노출한다.
EXPOSE 8080

# Flask 앱 실행
CMD ["python", "main.py"]