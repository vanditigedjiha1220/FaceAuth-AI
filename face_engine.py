
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self):
        # CPUExecutionProvider permet de fonctionner sans GPU.
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.model.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

    def extract_embedding(self, image_bytes):
        """
        Reçoit les octets d'une image et retourne
        le vecteur facial du visage détecté.
        """
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Le fichier image est invalide.")

        faces = self.model.get(image)

        if len(faces) == 0:
            raise ValueError("Aucun visage détecté.")

        if len(faces) > 1:
            raise ValueError(
                "Plusieurs visages détectés. "
                "Utilise une photo contenant un seul visage."
            )

        embedding = faces[0].normed_embedding

        return embedding.astype(np.float32)
