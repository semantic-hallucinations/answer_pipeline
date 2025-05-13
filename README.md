# ANSWER RAG PIPELINE
Моделька для ответов на вопросы на основе эмбеддингов из qdrant. Для запуска нужен искуственный интеллект). Несколько видов
файлов зависимостей для всех видов менеджеров пакетов). после подгрузки переходите на swagger документацию(/docs). Первые несколько минут, потому что эмбеддинг модель с жесткого диска в оперативу будет лететь). Поднимать через docker-compose up. Можете выбрать другую модель, не дипсик с опенроутера, там все универсально в настройках, можете выбрать какую нибудь побыстрее, тот же квен. дипсик мне понравился по ответам, со своими арбузными приветами(если вы понимаете о чем я). Если кидает Empty Response то у вас 2 ключа кончились, в логах будет писаться о переключении c одного на другой, поэтому будет понятно когда заменять.

Model for answering questions based on embeddings from qdrant.. Several types of dependency files for all types of package managers). after loading, go to swagger documentation (/docs). The first few minutes, because the embedding model from the hard drive will fly into the RAM). Raise via docker-compose up. You can choose another model, not a deepsik from an open router, everything is universal in the settings, you can choose one faster, the same qwen. I liked deepsik for its answers. If it throws Empty Response, then you have run out of 2 keys, the logs will write about switching.

# microservice-template
Template repository for microservice creation

## Structure:
- pre-commit config
- ci config
- cd config
- basic service

## Basic usage

CI runs on every push/pull to main branch, to check your changes llocally use pre-commit
CD runs if CI on main branch finishes succesfully(for organisation)

## Pre-commit

### To run pre-commit locally for ci checkouts:
1. ```pip install pre-commit```
2. ```pre-commit install```
3. ```pre-commit run -a```

### !!!Warning!!!
Pre-commit runs only on files that added to git by ```git add```

## Local running
```docker compose up --build``` and run example.py

## Chech publication history
You can check publishing history in organisation -> packages

## Using docker image
```
services:
  microservice:
    image: ghcr.io/semantic-hallucinations/py-microservice-template:latest   # or commit sha, or tag name instead of <latest>
```
