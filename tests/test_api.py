import pytest
import tempfile
from fastapi.testclient import TestClient
from app.model_Fusion import predict
from app.main import app
import json
from PIL import Image, ImageDraw, ImageFont
import io
from unittest.mock import patch, MagicMock

client = TestClient(app)

# Créer une image simulée en mémoire
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

def test_predict_endpoint():
    # Simuler un fichier d'image temporaire pour le test
    simulated_image = create_simulated_image()

    # Texte pour la prédiction
    text_data = "Exemple de texte pour la prédiction"

    # Créer un mock pour la fonction predict du module app.model_Fusion
    mock_predict = MagicMock()
    mock_predict.return_value = ("123", "Exemple")

    # Utiliser un patch pour remplacer la fonction predict dans app.model_Fusion par le mock
    with patch("app.model_Fusion.predict", mock_predict):
        # Préparez la requête POST avec l'image simulée
        image_file = ("image", ("temp_image.jpg", simulated_image, "image/jpeg"))

        # Utilisez un autre patch pour remplacer l'emplacement du modèle par un chemin fictif
        with patch("app.model_Fusion.combined_model_path", "/Users/PatriciaWintrebert/Downloads/combined_model_trained.h5"):
            # Envoi requête POST au point de terminaison
            response = client.post("/model/fusion/predict", data={"text": text_data}, files=[image_file])

            # Vérifier la réponse 
            assert response.status_code == 200
            result = response.json()
            assert "prdtypecode" in result
            assert "thematique" in result

# Exécutez le test avec pytest
if __name__ == "__main__":
    pytest.main([__file__])
