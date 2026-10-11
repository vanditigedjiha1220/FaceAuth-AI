
from pathlib import Path
from face_engine import FaceEngine

image_path = Path("photo_test.jpg")

if not image_path.exists():
    raise FileNotFoundError(
        "Place une photo nommée photo_test.jpg "
        "dans le dossier FaceAuth-AI."
    )

engine = FaceEngine()

embedding = engine.extract_embedding(
    image_path.read_bytes()
)

print("Extraction réussie !")
print("Dimensions du vecteur :", embedding.shape)
print("Premières valeurs :", embedding[:5])
