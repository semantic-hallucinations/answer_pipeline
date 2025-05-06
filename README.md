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
