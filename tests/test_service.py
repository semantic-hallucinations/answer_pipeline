# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# from fastapi.testclient import TestClient
# from main import app

# client = TestClient(app)

# def test_calculate_length_default_multiplier():
#     response = client.post(
#         "/length/",
#         json={"text": "test"},
#         headers={}
#     )
#     assert response.status_code == 200
#     assert response.json() == {"length": 4, "result": 4}

# def test_calculate_length_custom_multiplier():
#     response = client.post(
#         "/length/",
#         json={"text": "hello"},
#         headers={"multiplier": "3"}
#     )
#     assert response.status_code == 200
#     assert response.json() == {"length": 5, "result": 15}

# def test_invalid_data():
#     response = client.post(
#         "/length/",
#         json={"invalid_field": "test"},
#         headers={"multiplier": "2"}
#     )
#     assert response.status_code == 422
