# CD1_OpenCode
2022년 1학기 컴퓨터공학과 캡스톤디자인1 오픈코드팀

## 주의 사항

### git pull

무조건 pull하신 후에 작업하시기를 바랍니다.

### git branch

각각 각자의 branch를 생성하여 작업하시기를 바랍니다.

```console
    git pull
```

### 가상환경 구축 및 실행

```console
    python3 -m venv env
    source ./env/bin/activate
```

### requirement.txt에 있는 library 설치

```console
    pip install -r requirements.txt
```

### 추가 library 설치 후 requirement.txt 최신화

```console
    pip freeze > requirements.txt
```

### 가상환경 종료

```console
    deactivate
```

### ocean-cli 부분

ocean-cli를 설치하고 **oc**라는 command가 없을 시 **ocean**이라고 치면 사용 가능 합니다.