from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 기본 설정
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# DAG 정의
with DAG(
    dag_id='test_dag',  # DAG ID
    default_args=default_args,
    start_date=datetime(2023, 11, 1),
    schedule_interval='@daily',  # 하루마다 실행
    catchup=False,
) as dag:

    def print_hello():
        print("Hello, Airflow!")
        
    # PythonOperator를 사용하여 간단한 Python 함수 실행
    hello_task = PythonOperator(
        task_id='print_hello_task',
        python_callable=print_hello,  # 실행할 함수
    )

    hello_task
