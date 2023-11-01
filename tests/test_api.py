import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_Fusion import predict
from PIL import Image, ImageDraw
import io
from unittest.mock import patch, MagicMock

# Créer un client de test pour l'application
client = TestClient(app)

# Fonction pour créer une image simulée en mémoire
def create_simulated_image():
    # Créer une image vide de 100x100 pixels avec un fond blanc
    image = Image.new("RGB", (100, 100), (255, 255, 255))

    # Dessiner quelque chose sur l'image (par exemple, un rectangle rouge)
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 90, 90], fill=(255, 0, 0))

    # Enregistrer l'image en mémoire dans un objet BytesIO
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    
    # Revenir au début du tampon
    image_buffer.seek(0)

    return image_buffer

# Mock pour la fonction predict
def mock_predict(text, image_path):
    # Simuler la prédiction avec des valeurs fictives
    return "123", "Exemple"

# Redéfinition de la fonction predict avec le mock
app.dependency_overrides[predict] = mock_predict

def test_predict_endpoint():
    # Simuler un fichier d'image temporaire pour le test
    simulated_image = create_simulated_image()

    # Texte pour la prédiction
    text_data = "Exemple de texte pour la prédiction"

    # Envoi de la requête POST au point de terminaison
    response = client.post("/model/fusion/predict", data={"text": text_data}, files={"image": ("temp_image.jpg", simulated_image, "image/jpeg")})

    # Vérification de la réponse
    assert response.status_code == 200
    result = response.json()
    assert "prdtypecode" in result
    assert "thematique" in result

# Exécutez le test avec pytest
if __name__ == "__main__":
    pytest.main([__file__])
