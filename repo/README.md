# Get Repository by using GitHub Api

## 사용법

1. 라이브러리 설치

```console
    npm install
```
2. GitHub 개인 access token 만들기(있으면 넘어가세요~)

[GitHub 개인 access token 만들기][GitHub API access Token]


[GitHub API access Token]: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token "Creating a personal access token"

3. .env 파일 만들기 & 설정

personal-access-token 자리에 개인 access token을 넣으면 된다.

```
    GITHUB_ACCESS_TOKEN=personal-access-token
```

4. 실행

```console
    node index.mjs
```