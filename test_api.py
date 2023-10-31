#Test unitaires
import pytest
from fastapi.testclient import TestClient
from api import app  

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

def test_bert_predict():
   
    data = {
        "texts": [
            {"description_complete_prétraite": "Description de test 1"},
            {"description_complete_prétraite": "Description de test 2"}
        ]
    }

    response = client.post("/bert128/predict/", json=data)

    assert response.status_code == 200

    # Assurez-vous que la réponse contient les prédictions (adaptez cette partie en fonction de votre modèle et de sa sortie)
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == len(data["texts"])