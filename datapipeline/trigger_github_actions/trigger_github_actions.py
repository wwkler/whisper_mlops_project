'''
HTTP POST 방식으로 Github Action을 촉발시킨다. 
그래서 modeling_automated_pipeline를 실행하도록 하는 것을 궁극적 목표로 한다. 
'''
import os
import requests

def trigger_github_action(repo, workflow_id, token, ref="main"):
    """
    GitHub Actions workflow를 트리거하는 함수.

    Parameters:
    - repo: 리포지토리 이름 ("owner/repo" 형식)
    - workflow_id: 실행할 워크플로우 파일 이름 (예: 'trigger_workflow.yml')
    - token: GitHub Personal Access Token
    - ref: 브랜치 이름 (기본값은 "main")

    Returns:
    - HTTP 응답의 상태 코드와 메시지
    """
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "ref": ref  # 트리거할 브랜치
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 204:
        print("Workflow triggered successfully.")
    else:
        print(f"Failed to trigger workflow: {response.status_code}")
        print(response.json())

# 사용 예시
repo = os.getenv("REPO")  # "owner/repo" 형식으로 변경하세요.
workflow_id = os.getenv("WORKFLOW_ID")  # 실행할 워크플로우 파일 이름
token = os.getenv("GITHUB_TOKEN")  # GitHub Personal Access Token 입력
trigger_github_action(repo, workflow_id, token)

