# FaceAuth-AI — Guide détaillé du code, fichier par fichier

**Auteur :** VANDI TIGE DJIHA  
**Projet :** https://github.com/vanditigedjiha1220/FaceAuth-AI  
**Langage principal :** Python  
**Objectif de ce document :** expliquer le rôle de chaque fichier, les instructions importantes, les paramètres, les fonctions et la raison d'être de chaque bloc de code.

> **Périmètre.** Cette documentation correspond au code actuellement publié sur la branche `main` du dépôt. Les numéros de ligne peuvent évoluer si le code est modifié. Les lignes vides servent uniquement à aérer le code ; les commentaires commençant par `#` servent à expliquer ou délimiter des sections et ne sont pas exécutés.

---

## 1. Vue d'ensemble : comment les fichiers travaillent ensemble

| Fichier | Rôle |
|---|---|
| `app.py` | Application Streamlit : interface, enregistrement, reconnaissance, comparaison, gestion des profils et explications mathématiques. |
| `face_engine.py` | Moteur de vision : décode l'image, détecte le visage et extrait son embedding numérique. |
| `test_face.py` | Petit programme indépendant pour vérifier que le moteur extrait correctement un embedding à partir d'une image de test. |
| `requirements.txt` | Liste les bibliothèques Python et leurs versions à installer. |
| `.gitignore` | Indique à Git quels fichiers locaux ou sensibles ne doivent pas être suivis. |
| `README.md` | Présente le projet aux utilisateurs et explique comment l'installer et l'exécuter. |
| `faceauth.db` | Base SQLite créée à l'exécution par l'application ; elle conserve les profils. Elle ne doit pas être publiée. |
| `.venv/` | Environnement Python isolé créé sur l'ordinateur. Il ne fait pas partie du code source à publier. |

### Flux général

1. L'utilisateur fournit une photo, en l'important ou en utilisant la caméra.
2. `app.py` transmet les octets de la photo à `FaceEngine.extract_embedding()`.
3. `face_engine.py` décode la photo avec OpenCV, détecte les visages avec InsightFace et renvoie un vecteur numérique.
4. Pour un nouvel enregistrement, `app.py` stocke la photo et le vecteur dans SQLite.
5. Pour une reconnaissance, `app.py` compare le vecteur reçu à ceux enregistrés, trie les scores et compare le meilleur score au seuil choisi.
6. Streamlit affiche le résultat dans le navigateur.

---

# 2. Fichier `face_engine.py`

## 2.1. Rôle du fichier

Ce fichier sépare le traitement de l'image de l'interface. Il ne dessine pas l'interface et ne gère pas directement la base de données. Sa responsabilité est de transformer une image en une représentation numérique du visage, appelée **embedding**.

Cette séparation rend le code plus facile à tester et à réutiliser : l'application principale peut demander un embedding sans connaître les détails internes de la détection.

## 2.2. Imports — lignes 1 à 3

```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis
```

- `import cv2` : importe OpenCV. Ici, OpenCV sert principalement à décoder les octets du fichier image en tableau de pixels exploitable par le modèle.
- `import numpy as np` : importe NumPy sous l'alias `np`. NumPy manipule les tableaux numériques et permet de convertir l'embedding dans un type numérique précis.
- `from insightface.app import FaceAnalysis` : importe la classe InsightFace qui fournit les fonctions de détection de visage et d'analyse faciale.

**Pourquoi ces trois imports ?** OpenCV lit l'image, InsightFace analyse le visage, et NumPy prépare les données numériques. Sans ces bibliothèques, le moteur ne pourrait pas exécuter sa mission.

## 2.3. Déclaration de la classe — ligne 5

```python
class FaceEngine:
```

- `class` définit une classe, c'est-à-dire un modèle servant à créer des objets.
- `FaceEngine` est le nom donné au moteur facial.
- Les deux-points `:` annoncent le bloc d'instructions qui appartient à la classe.
- L'indentation indique quelles instructions sont à l'intérieur de la classe.

**Pourquoi une classe ?** Elle permet de regrouper l'initialisation du modèle et la méthode d'extraction dans un même objet. L'application peut créer le moteur une fois et réutiliser cet objet.

## 2.4. Initialisation du moteur — lignes 6 à 16

```python
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
```

- `def __init__(self):` : définit le constructeur. Python l'appelle automatiquement quand on écrit `FaceEngine()`.
- `self` : référence l'objet en cours de création. Il permet de conserver le modèle dans cet objet.
- Le commentaire commençant par `#` n'est pas exécuté ; il explique le choix du processeur.
- `self.model = FaceAnalysis(...)` : crée l'objet InsightFace et le conserve dans l'attribut `model`.
- `name="buffalo_l"` : sélectionne le paquet de modèles InsightFace nommé `buffalo_l`. Il comprend les modèles préentraînés nécessaires à l'analyse faciale.
- `providers=["CPUExecutionProvider"]` : demande à ONNX Runtime d'utiliser le processeur. Cela évite d'exiger un GPU compatible, au prix possible d'une exécution plus lente.
- `self.model.prepare(...)` : prépare le modèle avant son utilisation. Cette étape initialise les composants de détection et d'extraction.
- `ctx_id=-1` : dans cette configuration InsightFace, indique l'utilisation du CPU plutôt que d'un GPU.
- `det_size=(640, 640)` : fixe la taille de travail de la détection à 640 × 640. Cette valeur est un paramètre de traitement, pas la taille obligatoire du fichier original.

**Pourquoi initialiser le modèle ici ?** Le modèle doit être préparé avant chaque analyse. Le constructeur centralise cette préparation afin que le reste du programme n'ait pas à la répéter.

## 2.5. Méthode d'extraction — lignes 17 à 40

```python
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
```

### Signature et documentation

- `def extract_embedding(self, image_bytes):` : déclare une méthode qui reçoit les octets de l'image. `image_bytes` est le contenu binaire du fichier, et non son chemin sur le disque.
- Le texte entre triples guillemets est une **docstring**. Il documente le but de la méthode ; il n'effectue pas le traitement.

### Conversion des octets en image

- `np.frombuffer(image_bytes, dtype=np.uint8)` : interprète les octets comme un tableau de valeurs entières non signées sur 8 bits, adaptées aux valeurs de pixels d'une image courante.
- `dtype=np.uint8` précise le type des éléments du tableau.
- `cv2.imdecode(image_array, cv2.IMREAD_COLOR)` : décode le tableau d'octets en image couleur OpenCV.
- `cv2.IMREAD_COLOR` demande une image couleur. OpenCV représente généralement les canaux dans l'ordre BGR, et non RGB.

**Pourquoi cette conversion ?** Un fichier importé par Streamlit est fourni sous forme d'octets. Le modèle de vision a besoin d'une matrice de pixels ; ces deux instructions réalisent la conversion.

### Vérification du fichier

```python
if image is None:
    raise ValueError("Le fichier image est invalide.")
```

- `if` teste une condition.
- `image is None` vérifie que le décodage n'a produit aucune image.
- `raise ValueError(...)` arrête le traitement et signale une erreur compréhensible à l'appelant.

Cette vérification empêche de transmettre une image invalide au modèle et d'obtenir une erreur plus difficile à interpréter plus loin.

### Détection des visages

```python
faces = self.model.get(image)
```

- `self.model.get(image)` exécute l'analyse sur l'image.
- Le résultat `faces` contient les visages détectés et leurs informations, dont l'embedding.
- Le résultat est une collection : elle peut contenir zéro, un ou plusieurs visages.

### Contrôle du nombre de visages

```python
if len(faces) == 0:
    raise ValueError("Aucun visage détecté.")
```

- `len(faces)` compte les visages détectés.
- `== 0` signifie « égal à zéro ».
- Si aucun visage n'est trouvé, la méthode interrompt l'opération.

```python
if len(faces) > 1:
    raise ValueError(
        "Plusieurs visages détectés. "
        "Utilise une photo contenant un seul visage."
    )
```

- `> 1` signifie qu'il y a plus d'un visage.
- Le message est réparti sur plusieurs lignes pour améliorer la lisibilité.
- En Python, deux chaînes littérales adjacentes entre parenthèses sont concaténées automatiquement.

**Pourquoi refuser plusieurs visages ?** Le prototype doit associer une photo à un seul profil. Si plusieurs personnes apparaissaient sur la photo, le programme ne demanderait pas laquelle choisir ; il préfère donc refuser cette entrée ambiguë.

### Récupération et retour de l'embedding

```python
embedding = faces[0].normed_embedding
return embedding.astype(np.float32)
```

- `faces[0]` désigne le premier élément de la collection. Comme les contrôles précédents ont imposé exactement un visage, cet élément est le visage attendu.
- `.normed_embedding` récupère la représentation numérique normalisée fournie par InsightFace.
- `return` renvoie le résultat à la fonction appelante, par exemple à `app.py`.
- `.astype(np.float32)` convertit les valeurs en nombres flottants de 32 bits. Ce type est compact et adapté aux calculs numériques utilisés ensuite.

Un embedding du modèle employé est généralement un vecteur de 512 valeurs. Il ne s'agit pas du nom de la personne ni d'une photo : c'est une représentation numérique calculée par le réseau neuronal.

---

# 3. Fichier `test_face.py`

## 3.1. Rôle

Ce fichier teste le moteur facial sans lancer toute l'interface Streamlit. Il permet de vérifier rapidement qu'une photo est lisible, qu'un visage est détecté et qu'un embedding est produit.

## 3.2. Explication des instructions

```python
from pathlib import Path
from face_engine import FaceEngine
```

- `from pathlib import Path` importe `Path`, une classe standard pour manipuler les chemins de fichiers de manière portable.
- `from face_engine import FaceEngine` importe la classe créée dans `face_engine.py`. Le fichier de test réutilise ainsi le moteur au lieu de dupliquer son code.

```python
image_path = Path("photo_test.jpg")
```

- Crée un objet chemin qui pointe vers `photo_test.jpg`.
- Le chemin est relatif au répertoire courant depuis lequel le script est lancé.
- La photo doit donc se trouver dans le dossier de projet si le script est lancé depuis ce dossier.

```python
if not image_path.exists():
    raise FileNotFoundError(
        "Place une photo nommée photo_test.jpg "
        "dans le dossier FaceAuth-AI."
    )
```

- `.exists()` vérifie que le fichier existe.
- `not` inverse le résultat : le bloc est exécuté si le fichier n'existe pas.
- `FileNotFoundError` signale explicitement l'absence du fichier.
- Le message indique à l'utilisateur comment corriger le problème.

Cette vérification est utile parce qu'un fichier absent ne peut pas être analysé.

```python
engine = FaceEngine()
```

- Crée une instance du moteur.
- Cette instruction déclenche `__init__` et prépare le modèle InsightFace.
- Le chargement initial peut prendre du temps, notamment la première fois si les modèles doivent être téléchargés.

```python
embedding = engine.extract_embedding(
    image_path.read_bytes()
)
```

- `.read_bytes()` lit le fichier et renvoie son contenu binaire.
- Ce contenu est transmis à `extract_embedding()`.
- Le résultat est conservé dans la variable `embedding`.

```python
print("Extraction réussie !")
print("Dimensions du vecteur :", embedding.shape)
print("Premières valeurs :", embedding[:5])
```

- Le premier `print` confirme que les instructions précédentes ont terminé sans exception.
- `embedding.shape` affiche la forme du tableau. Pour un vecteur d'une dimension, le résultat ressemble généralement à `(512,)`.
- `embedding[:5]` sélectionne les cinq premières valeurs, afin d'inspecter un extrait sans afficher tout le vecteur.
- `print` écrit les résultats dans le terminal.

**Exécution :** place une photo autorisée nommée `photo_test.jpg` à la racine du projet, active l'environnement virtuel, puis exécute `python test_face.py`. Le fichier `photo_test.jpg` est ignoré par Git afin d'éviter de publier une image personnelle.

---

# 4. Fichier `app.py`

## 4.1. Rôle général

`app.py` est le point d'entrée de l'application. Il construit l'interface Streamlit et coordonne les autres éléments : moteur facial, base SQLite, import de photos, comparaison des embeddings et affichage des résultats.

Le code est organisé en sept parties :
1. configuration et base de données ;
2. chargement du moteur ;
3. enregistrement ;
4. reconnaissance automatique ;
5. gestion et suppression des profils ;
6. explication mathématique ;
7. pied de page.

## 4.2. Imports — lignes 1 à 7

```python
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from face_engine import FaceEngine
```

- `sqlite3` : bibliothèque standard Python pour utiliser une base de données SQLite, sans serveur de base de données séparé.
- `Path` : construit le chemin de la base de données indépendamment de la façon dont le chemin est écrit.
- `numpy as np` : convertit les embeddings et calcule notamment leur produit scalaire.
- `pandas as pd` : transforme les scores de comparaison en tableau de données affichable.
- `streamlit as st` : fournit les composants de l'interface web, comme les onglets, boutons, messages et tableaux.
- `FaceEngine` : importe le moteur défini dans `face_engine.py`.

Les lignes vides séparent visuellement la bibliothèque standard, les bibliothèques externes et le module du projet.

## 4.3. Configuration de l'interface — lignes 12 à 16

```python
st.set_page_config(
    page_title="FaceAuth-AI",
    page_icon="🔐",
    layout="wide"
)
```

- `st.set_page_config(...)` configure la page Streamlit.
- `page_title` définit le titre associé à l'onglet du navigateur.
- `page_icon` choisit l'icône affichée dans l'onglet.
- `layout="wide"` demande une disposition large, utile pour les tableaux et les quatre onglets.

Cette configuration est placée au début, avant les autres éléments de l'interface, conformément à l'usage attendu de Streamlit.

## 4.4. Chemin de la base — ligne 18

```python
DB_PATH = Path(__file__).with_name("faceauth.db")
```

- `DB_PATH` est une constante par convention : son nom en majuscules signale qu'elle est destinée à rester stable.
- `__file__` représente le chemin du fichier Python en cours (`app.py`).
- `Path(__file__)` convertit ce chemin en objet `Path`.
- `.with_name("faceauth.db")` remplace le nom `app.py` par `faceauth.db` tout en conservant le même dossier.

**Pourquoi cette méthode ?** La base sera placée à côté de `app.py`, même si l'application est lancée depuis un autre répertoire courant.

## 4.5. Initialisation de SQLite — lignes 20 à 30

```python
def init_db():
    """Créer la table des profils si elle n'existe pas."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY,
                photo BLOB NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
```

- `def init_db():` déclare une fonction d'initialisation de la base.
- La docstring explique sa responsabilité.
- `sqlite3.connect(DB_PATH)` ouvre une connexion à la base. Si le fichier n'existe pas, SQLite le crée.
- `with ... as conn` utilise un gestionnaire de contexte pour encadrer la connexion et gérer proprement la fin de la transaction/connexion selon le comportement de SQLite.
- `conn` est la variable qui représente la connexion.
- `conn.execute(...)` exécute une instruction SQL.
- Les triples guillemets permettent d'écrire la requête SQL sur plusieurs lignes.
- `CREATE TABLE IF NOT EXISTS profiles` crée la table `profiles` uniquement si elle n'existe pas déjà. Les redémarrages ne recréent donc pas la table.
- `user_id TEXT PRIMARY KEY` stocke l'identifiant sous forme de texte et en fait la clé primaire : deux profils ne peuvent pas avoir le même identifiant.
- `photo BLOB NOT NULL` conserve les octets de la photo. `BLOB` est un type adapté aux données binaires ; `NOT NULL` interdit une valeur vide SQL `NULL`.
- `embedding BLOB NOT NULL` stocke le vecteur sous forme binaire.
- `created_at TEXT DEFAULT CURRENT_TIMESTAMP` conserve la date et l'heure de création si aucune valeur n'est fournie lors de l'insertion.
- `"""` ferme la chaîne SQL.

**Pourquoi SQLite ?** La base est locale, légère et ne demande pas de serveur séparé. Elle convient à un prototype individuel, mais elle n'est pas chiffrée par ce code.

## 4.6. Enregistrement ou remplacement d'un profil — lignes 32 à 48

```python
def save_profile(user_id, photo_bytes, embedding):
    """Enregistrer ou remplacer un profil."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO profiles (user_id, photo, embedding)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                photo = excluded.photo,
                embedding = excluded.embedding,
                created_at = CURRENT_TIMESTAMP
        """, (
            user_id,
            sqlite3.Binary(photo_bytes),
            sqlite3.Binary(
                np.asarray(embedding, dtype=np.float32).tobytes()
            )
        ))
```

- La fonction reçoit l'identifiant, les octets de la photo et l'embedding.
- La connexion ouvre la base pour exécuter l'écriture.
- `INSERT INTO profiles (...)` demande d'ajouter un enregistrement dans la table.
- `VALUES (?, ?, ?)` utilise trois paramètres. Les `?` sont des paramètres liés, et non du texte assemblé manuellement. Cette méthode évite de construire la requête SQL à partir de l'identifiant.
- `ON CONFLICT(user_id) DO UPDATE` définit le comportement si l'identifiant existe déjà.
- `excluded.photo` désigne la nouvelle photo proposée à l'insertion ; `excluded.embedding` désigne le nouvel embedding.
- Les trois affectations remplacent l'ancienne photo, l'ancien embedding et actualisent la date.
- Le tuple après la requête fournit les valeurs correspondant aux trois `?`, dans le même ordre.
- `sqlite3.Binary(photo_bytes)` indique à SQLite que la valeur est binaire.
- `np.asarray(embedding, dtype=np.float32)` garantit une représentation NumPy en flottants 32 bits.
- `.tobytes()` transforme le tableau en octets afin de le stocker dans le champ `BLOB`.

**Conséquence importante :** réenregistrer le même identifiant remplace son profil précédent. Ce code ne conserve pas plusieurs photos de référence par personne.

## 4.7. Lecture des profils — lignes 50 à 69

```python
def get_profiles():
    """Charger les profils depuis SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT user_id, photo, embedding, created_at
            FROM profiles
            ORDER BY user_id
        """).fetchall()

    profiles = {}
    for user_id, photo, embedding_bytes, created_at in rows:
        profiles[user_id] = {
            "photo": photo,
            "embedding": np.frombuffer(
                embedding_bytes, dtype=np.float32
            ).copy(),
            "created_at": created_at
        }

    return profiles
```

- `SELECT` demande à lire les colonnes indiquées.
- `FROM profiles` précise la table interrogée.
- `ORDER BY user_id` trie les lignes selon l'identifiant.
- `.fetchall()` récupère toutes les lignes du résultat dans une collection.
- La connexion est utilisée dans le bloc `with` ; après ce bloc, les données lues restent disponibles dans `rows`.
- `profiles = {}` crée un dictionnaire vide. Un dictionnaire associe une clé à une valeur.
- `for ... in rows` parcourt chaque ligne. Les quatre variables reçoivent les colonnes dans l'ordre du `SELECT`.
- `profiles[user_id] = {...}` utilise l'identifiant comme clé du dictionnaire.
- `"photo": photo` conserve les octets de la photo.
- `np.frombuffer(embedding_bytes, dtype=np.float32)` reconstruit un tableau NumPy à partir des octets stockés.
- `.copy()` crée une copie indépendante du tableau reconstruit, ce qui évite de dépendre directement du tampon d'origine.
- `"created_at": created_at` conserve la date d'enregistrement.
- `return profiles` renvoie le dictionnaire à l'appelant.

Le dictionnaire a une forme conceptuelle semblable à :

```python
{
    "Vandi": {
        "photo": b"...",
        "embedding": array([...]),
        "created_at": "..."
    }
}
```

Les valeurs `b"..."` et `array([...])` sont ici des représentations simplifiées, pas les données réelles.

## 4.8. Suppression d'un profil — lignes 71 à 77

```python
def delete_profile(user_id):
    """Supprimer définitivement un profil de la base locale."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM profiles WHERE user_id = ?",
            (user_id,)
        )
```

- La fonction reçoit l'identifiant du profil à supprimer.
- `DELETE FROM profiles` demande de supprimer des lignes de la table.
- `WHERE user_id = ?` limite la suppression à la ligne dont l'identifiant correspond au paramètre.
- `(user_id,)` est un tuple contenant un seul élément. La virgule est nécessaire pour que Python le reconnaisse comme tuple à un élément.
- Le paramètre lié évite de construire la requête avec une concaténation de chaînes.

La suppression retire de la base la photo et l'embedding associés au profil. Elle n'efface pas nécessairement d'éventuelles copies externes ou sauvegardes.

## 4.9. Initialisation et chargement du moteur — lignes 82 à 104

```python
init_db()

@st.cache_resource
def load_face_engine():
    return FaceEngine()
```

- `init_db()` est appelé immédiatement lorsque Streamlit exécute le script. Cela garantit que la table existe avant les opérations de lecture ou d'écriture.
- `@st.cache_resource` est un décorateur Streamlit. Il demande à Streamlit de conserver une ressource coûteuse entre les réexécutions de l'application.
- `def load_face_engine():` déclare la fonction qui fournit le moteur.
- `return FaceEngine()` crée et renvoie le moteur lors du chargement initial ; les appels suivants peuvent réutiliser la ressource mise en cache.

**Pourquoi le cache ?** Charger un modèle de vision peut prendre du temps et consommer de la mémoire. Sans cache, les réexécutions de l'interface risqueraient de recharger le modèle trop souvent.

```python
st.title("🔐 FaceAuth-AI")
st.caption(
    "Reconnaissance faciale, représentation numérique "
    "et explication mathématique"
)
```

- `st.title` affiche le titre principal.
- `st.caption` affiche un texte secondaire, plus discret.
- La chaîne est répartie sur plusieurs lignes, mais les deux morceaux sont concaténés en une phrase.

```python
try:
    engine = load_face_engine()
except Exception as e:
    st.error(f"Impossible de charger le moteur facial : {e}")
    st.stop()
```

- `try` tente une opération susceptible d'échouer.
- `load_face_engine()` récupère le moteur.
- `except Exception as e` intercepte une exception et la place dans la variable `e`.
- La f-string `f"..."` insère le texte de l'erreur dans le message.
- `st.error` affiche une erreur dans l'interface.
- `st.stop()` arrête l'exécution Streamlit du script courant. Il évite de continuer sans moteur utilisable.

```python
profiles = get_profiles()
st.metric("Personnes enregistrées", len(profiles))
```

- Recharge les profils persistants depuis la base.
- `len(profiles)` compte les identifiants enregistrés.
- `st.metric` affiche ce nombre dans un composant de type indicateur.

## 4.10. Création des quatre onglets — lignes 106 à 111

```python
tab_register, tab_recognize, tab_manage, tab_math = st.tabs([
    "💾 👤 Enregistrer",
    "🔎🤖 Reconnaître",
    "👤📁 Gérer les profils",
    "🧮 Logique mathématique"
])
```

- `st.tabs([...])` crée quatre onglets dans l'interface.
- La liste contient les titres affichés à l'utilisateur.
- Les quatre variables à gauche reçoivent les objets représentant les onglets, dans le même ordre que la liste.
- `tab_register` désigne l'enregistrement, `tab_recognize` la reconnaissance, `tab_manage` la gestion et `tab_math` les explications.

Les noms de variables sont choisis pour rendre le code lisible. Les emojis ne participent pas au calcul ; ils servent uniquement à l'interface.

## 4.11. Onglet « Enregistrer » — lignes 116 à 204

### Afficher le titre et les instructions

```python
with tab_register:
    st.subheader("Enregistrer une personne")
    st.write(
        "Choisis un identifiant, puis importe une photo "
        "ou prends une photo avec la caméra."
    )
```

- `with tab_register:` place les composants suivants dans l'onglet Enregistrer.
- `st.subheader` affiche un sous-titre.
- `st.write` affiche du texte.

### Saisir un identifiant

```python
user_id = st.text_input(
    "Nom ou identifiant de la personne",
    placeholder="Ex. Vandi",
    key="register_id"
)
```

- `st.text_input` affiche un champ de saisie texte.
- Le premier argument est le libellé.
- `placeholder` affiche un exemple tant que le champ est vide.
- `key` fournit un identifiant stable à Streamlit pour distinguer ce champ des autres champs.

### Choisir l'origine de la photo

```python
source = st.radio(
    "Source de la photo",
    ["Importer une photo", "Utiliser la caméra"],
    key="register_source"
)
```

- `st.radio` crée un choix entre plusieurs options, avec une seule option sélectionnable à la fois.
- La valeur sélectionnée est stockée dans `source`.
- La `key` évite la confusion avec le choix de source utilisé dans l'onglet Reconnaître.

### Importer ou capturer l'image

```python
if source == "Importer une photo":
    photo_file = st.file_uploader(
        "Sélectionner une photo",
        type=["jpg", "jpeg", "png"],
        key="register_upload"
    )
else:
    photo_file = st.camera_input(
        "Prendre une photo du visage",
        key="register_camera"
    )
```

- `if` teste l'option choisie.
- `==` compare deux valeurs.
- `st.file_uploader` permet de sélectionner un fichier depuis l'appareil.
- `type=[...]` limite les extensions proposées à JPG, JPEG et PNG. Cela ne garantit pas à lui seul que tout fichier est valide : le moteur vérifie ensuite le décodage.
- `else` couvre l'autre choix, celui de la caméra.
- `st.camera_input` demande une photo à la caméra prise en charge par le navigateur.
- Dans les deux cas, le résultat est conservé dans `photo_file`, ce qui permet d'utiliser ensuite le même code.

### Prévisualiser la photo

```python
if photo_file is not None:
    photo_bytes = photo_file.getvalue()
    st.image(
        photo_bytes,
        caption="Photo à enregistrer",
        width=300
    )
```

- `is not None` vérifie qu'un fichier ou une capture a bien été fourni.
- `.getvalue()` récupère les octets de la photo.
- `st.image` affiche la photo.
- `caption` donne une légende.
- `width=300` limite la largeur d'affichage à 300 pixels environ dans l'interface.

### Réagir au bouton d'enregistrement

```python
if st.button("Enregistrer le profil", type="primary"):
```

- `st.button` affiche le bouton.
- Le bloc indenté est exécuté lorsque l'utilisateur clique sur ce bouton.
- `type="primary"` donne au bouton le style principal de Streamlit.

```python
if not user_id.strip():
    st.warning("Saisis un nom ou un identifiant.")
elif photo_file is None:
    st.warning("Choisis une photo ou utilise la caméra.")
```

- `user_id.strip()` enlève les espaces au début et à la fin.
- `not user_id.strip()` détecte une chaîne vide ou ne contenant que des espaces.
- `st.warning` affiche un avertissement.
- `elif` signifie « sinon, si » ; le second contrôle est testé seulement si le premier n'a pas été satisfait.
- Le second contrôle interdit de lancer l'enregistrement sans photo.

```python
else:
    try:
        photo_bytes = photo_file.getvalue()
        embedding = engine.extract_embedding(photo_bytes)
        clean_id = user_id.strip()
```

- `else` correspond au cas où l'identifiant et la photo sont présents.
- `try` encadre le traitement pour gérer les erreurs.
- Les octets de la photo sont récupérés.
- Le moteur transforme la photo en embedding. Une erreur de décodage, l'absence de visage ou la présence de plusieurs visages peut interrompre cette étape.
- `clean_id` conserve l'identifiant sans espaces superflus.

```python
save_profile(
    clean_id,
    photo_bytes,
    embedding
)
```

- Appelle la fonction de persistance.
- Les trois arguments correspondent à l'identifiant, à la photo binaire et au vecteur numérique.
- La fonction insère le profil ou remplace celui qui possède déjà cet identifiant.

```python
st.success(
    f"Le profil « {clean_id} » a été enregistré."
)
st.write("**Informations sur l'embedding créé**")
st.write(f"Dimension du vecteur : {embedding.size}")
st.write(f"Type numérique : {embedding.dtype}")
```

- `st.success` affiche un message de réussite.
- La f-string insère l'identifiant nettoyé.
- `st.write` affiche un titre en Markdown, puis la dimension du vecteur et son type.
- `.size` compte le nombre d'éléments du tableau.
- `.dtype` indique le type numérique de ces éléments.

```python
with st.expander("Voir les premières valeurs de cet embedding"):
    st.code(
        np.array2string(
            embedding[:20],
            precision=6,
            separator=", "
        )
    )
```

- `st.expander` crée une zone que l'utilisateur peut ouvrir ou fermer.
- `embedding[:20]` sélectionne les vingt premières valeurs.
- `np.array2string` transforme l'extrait du tableau en texte.
- `precision=6` limite l'affichage à six chiffres de précision après la virgule, selon la représentation NumPy.
- `separator=", "` sépare les valeurs par une virgule et un espace.
- `st.code` affiche le texte avec une mise en forme de code.

```python
st.caption(
    "Affichage des 20 premières valeurs seulement. "
    "Le vecteur complet est conservé dans la base."
)
st.rerun()
```

- `st.caption` précise que seul un extrait est visible.
- `st.rerun()` demande à Streamlit de réexécuter le script afin que les compteurs et les profils affichés reflètent immédiatement l'enregistrement.

### Gestion des erreurs et avertissement

```python
except ValueError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Erreur pendant l'enregistrement : {e}")
```

- Le premier `except` traite explicitement les erreurs de validation produites par le moteur, par exemple aucun visage.
- `str(e)` convertit l'erreur en texte.
- Le second `except` capture les autres exceptions pour afficher un message d'erreur plutôt que laisser l'interface échouer silencieusement.
- Il reste préférable, dans une application de production, de journaliser les erreurs techniques sans révéler de détails sensibles à l'utilisateur.

```python
st.info(
    "Si tu enregistres à nouveau un identifiant déjà existant, "
    "son ancienne photo et son ancien embedding seront remplacés."
)
```

- `st.info` affiche une information non bloquante.
- Le message avertit de la conséquence de la clé primaire et de la requête `ON CONFLICT` dans `save_profile`.

## 4.12. Onglet « Reconnaître » — lignes 209 à 387

### Vérifier qu'il existe des profils

```python
with tab_recognize:
    st.subheader("Identifier une personne")

    if not profiles:
        st.info("Enregistre d'abord au moins une personne.")
    else:
        ...
```

- `with tab_recognize` place le contenu dans l'onglet de reconnaissance.
- `if not profiles` est vrai si le dictionnaire est vide.
- Sans profils de référence, une comparaison n'aurait aucun sens ; l'application affiche donc une information et ne propose pas le traitement normal.

### Choisir la photo à reconnaître

Le bloc `st.radio` ressemble à celui de l'enregistrement, mais utilise une clé différente (`recognize_source`). Le choix est conservé dans `source_query`.

- `st.file_uploader` permet d'importer la photo candidate.
- `st.camera_input` permet de la capturer.
- `query_photo` contient le fichier ou la capture sélectionnée.
- Les clés `recognize_upload` et `recognize_camera` séparent ces widgets de ceux de l'autre onglet.

### Régler le seuil

```python
threshold = st.slider(
    "Seuil minimal de similarité",
    min_value=0.30,
    max_value=0.90,
    value=0.60,
    step=0.01,
    help=(
        "Ce seuil est expérimental. Il doit être évalué "
        "avec des photos représentatives avant tout usage réel."
    )
)
```

- `st.slider` affiche un curseur.
- `min_value=0.30` fixe la borne basse.
- `max_value=0.90` fixe la borne haute.
- `value=0.60` fixe la valeur initiale.
- `step=0.01` permet d'ajuster le seuil par pas de 0,01.
- `help` ajoute une aide contextuelle.
- La valeur choisie est placée dans `threshold`.

**Interprétation :** un seuil plus élevé exige une ressemblance plus forte, mais peut aussi rejeter une personne légitime. Un seuil plus bas accepte plus facilement des correspondances, mais augmente le risque de fausse acceptation. `0.60` est une valeur de départ expérimentale, pas une garantie de sécurité.

### Prévisualiser et lancer l'analyse

```python
if query_photo is not None:
    st.image(
        query_photo.getvalue(),
        caption="Photo à identifier",
        width=300
    )
```

Le code vérifie que la photo existe et l'affiche avant le traitement.

```python
if st.button("Lancer la reconnaissance", type="primary"):
    if query_photo is None:
        st.warning("Fournis une photo à analyser.")
    else:
        try:
            candidate = engine.extract_embedding(
                query_photo.getvalue()
            )
```

- Le bouton déclenche le processus.
- Si aucune photo n'a été fournie, l'application affiche un avertissement.
- Sinon, le moteur calcule `candidate`, l'embedding de la photo soumise.
- Le nom `candidate` signifie ici « vecteur candidat à identifier ».

### Comparer avec tous les profils

```python
results = []
for name, profile in profiles.items():
    reference = profile["embedding"]

    if reference.shape != candidate.shape:
        continue

    similarity = float(
        np.dot(reference, candidate)
    )
    results.append({
        "Personne": name,
        "Similarité cosinus": similarity
    })
```

- `results = []` crée une liste vide qui contiendra les comparaisons.
- `profiles.items()` parcourt les paires clé-valeur du dictionnaire : `name` reçoit l'identifiant et `profile` reçoit les informations correspondantes.
- `reference` récupère l'embedding de référence enregistré.
- `.shape` décrit la forme du vecteur. Si les dimensions diffèrent, les deux vecteurs ne sont pas comparables par le produit scalaire ; `continue` passe au profil suivant.
- `np.dot(reference, candidate)` calcule le produit scalaire des deux vecteurs.
- Comme les embeddings fournis ici sont normalisés, ce produit scalaire correspond au score de similarité cosinus, sous réserve de vecteurs de même dimension.
- `float(...)` convertit le résultat en flottant Python.
- `results.append({...})` ajoute un dictionnaire avec le nom et le score à la liste.

### Traiter l'absence de résultat

```python
if not results:
    st.error("Aucun profil compatible à comparer.")
```

Si aucun profil n'a pu être comparé — par exemple parce que les dimensions ne correspondent pas — l'application affiche une erreur.

### Trier les scores et prendre le meilleur

```python
results.sort(
    key=lambda item: item["Similarité cosinus"],
    reverse=True
)
best = results[0]
best_name = best["Personne"]
best_score = best["Similarité cosinus"]
```

- `.sort(...)` trie la liste directement.
- `key=lambda item: ...` indique que le critère de tri est le score contenu dans chaque dictionnaire.
- `lambda` définit ici une petite fonction anonyme.
- `reverse=True` trie du score le plus élevé au plus faible.
- `results[0]` devient donc le meilleur résultat.
- `best_name` récupère l'identifiant de ce résultat.
- `best_score` récupère son score.

Le meilleur résultat est seulement le profil **le plus proche parmi ceux disponibles**. Il ne faut pas automatiquement l'accepter : il faut encore vérifier le seuil.

### Décider à partir du seuil

```python
if best_score >= threshold:
    st.success(
        f"Meilleure correspondance : {best_name}"
    )
else:
    st.error(
        "Visage inconnu : aucun profil "
        "n'a atteint le seuil minimal."
    )
```

- `>=` signifie « supérieur ou égal à ».
- Si le meilleur score atteint le seuil, l'application propose le profil.
- Sinon, elle affiche « visage inconnu ».
- Ce résultat reste une décision expérimentale fondée sur le score ; ce n'est pas une preuve d'identité.

```python
st.metric(
    "Meilleur score de similarité",
    f"{best_score:.4f}"
)
```

- `st.metric` affiche le meilleur score.
- `:.4f` formate le nombre avec quatre chiffres après la virgule.

Le texte suivant utilise `st.caption` pour rappeler que le score mesure une proximité entre embeddings et n'est pas une probabilité d'identité.

### Afficher le tableau complet

```python
df_results = pd.DataFrame(results)
df_results["Seuil atteint"] = (
    df_results["Similarité cosinus"] >= threshold
)
```

- `pd.DataFrame(results)` convertit la liste de dictionnaires en tableau Pandas.
- Une colonne `Seuil atteint` est ajoutée.
- L'expression compare chaque score au seuil et produit des valeurs booléennes : `True` ou `False`.

```python
st.dataframe(
    df_results,
    use_container_width=True,
    hide_index=True
)
```

- `st.dataframe` affiche le tableau dans l'interface.
- `use_container_width=True` utilise la largeur disponible.
- `hide_index=True` masque l'index technique de Pandas.

### Détailler les calculs

Le bloc `with st.expander(..., expanded=True)` crée une section de détails ouverte par défaut.

- `candidate.size` affiche la dimension de l'embedding candidat.
- `np.linalg.norm(candidate)` calcule sa norme euclidienne.
- `candidate[:20]` montre seulement les vingt premières valeurs.
- `np.array2string(...)` formate cet extrait pour l'affichage.
- `st.latex(...)` affiche la formule mathématique de la similarité cosinus.
- `profiles[best_name]["embedding"]` récupère le vecteur du profil retenu.
- `reference.size` et `np.linalg.norm(reference)` donnent la dimension et la norme du vecteur de référence.
- `st.code("similarity = np.dot(reference, candidate)")` affiche la ligne de calcul comme illustration.
- `best_score` et `threshold` permettent de montrer le score obtenu et le seuil utilisé.
- Le `if best_score >= threshold` final affiche une explication textuelle de la décision.

Les formules de `st.latex` servent à expliquer le calcul ; elles ne réalisent pas elles-mêmes la comparaison. Le calcul est effectué par NumPy.

### Erreurs dans la reconnaissance

```python
except ValueError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Erreur pendant la reconnaissance : {e}")
```

Le premier bloc affiche les erreurs attendues de validation du visage. Le second traite les autres exceptions. Cela évite que l'utilisateur ne voie uniquement une trace technique dans le navigateur.

## 4.13. Onglet « Gérer les profils » — lignes 392 à 427

```python
with tab_manage:
    st.subheader("Profils enregistrés")

    if not profiles:
        st.info("Aucun profil enregistré.")
```

- Place le contenu dans l'onglet de gestion.
- Affiche un message si la base ne contient aucun profil.

```python
else:
    st.write(
        "Les profils ci-dessous sont chargés depuis la base SQLite."
    )
    for name, profile in profiles.items():
        with st.expander(f"👤 {name}"):
```

- Si des profils existent, affiche une explication.
- `for` parcourt chaque profil.
- L'expandeur crée une section repliable par personne.

À l'intérieur de chaque expandeur :

- `st.image(profile["photo"], ...)` affiche la photo de référence.
- `caption=f"Photo de référence : {name}"` personnalise sa légende.
- `profile['created_at']` affiche la date d'enregistrement.
- `profile['embedding'].size` affiche la dimension de l'embedding.

```python
if st.button(
    f"Supprimer le profil {name}",
    key=f"delete_{name}",
    type="secondary"
):
    delete_profile(name)
    st.success(f"Le profil « {name} » a été supprimé.")
    st.rerun()
```

- Chaque profil a un bouton de suppression.
- La f-string intègre le nom dans le libellé.
- `key=f"delete_{name}"` rend la clé du bouton distincte pour chaque profil.
- `type="secondary"` utilise un style secondaire.
- Après le clic, `delete_profile(name)` supprime la ligne correspondante de SQLite.
- `st.success` affiche la confirmation.
- `st.rerun()` recharge les données et met à jour l'affichage.

Le dernier `st.warning` rappelle que la suppression concerne la photo et l'embedding dans la base locale.

## 4.14. Onglet « Logique mathématique » — lignes 432 à 575

Cette partie est pédagogique : elle explique ce que l'application calcule.

### Présentation du processus

```python
st.markdown("""
Le processus comprend plusieurs étapes :
1. **Image d'entrée** : le programme reçoit une photo.
2. **Détection** : InsightFace localise le visage.
3. **Embedding** : le réseau neuronal transforme le visage
   en un vecteur numérique.
4. **Comparaison** : le vecteur reçu est comparé aux vecteurs
   enregistrés.
5. **Décision** : le meilleur score est confronté au seuil choisi.
""")
```

- `st.markdown` interprète le texte en Markdown.
- Les triples guillemets permettent une chaîne sur plusieurs lignes.
- Les `**...**` mettent les mots en gras.
- Les numéros présentent les étapes dans l'ordre logique.

`st.divider()` trace une séparation visuelle. Les `st.expander(...)` qui suivent permettent de consulter chaque notion sans encombrer l'écran.

### Expandeur 1 — embedding

Le texte explique qu'un embedding est un vecteur de caractéristiques calculé par le réseau neuronal, pas une simple liste de pixels.

- `st.latex(...)` affiche la notation mathématique du vecteur.
- `d` désigne sa dimension ; elle est généralement de 512 valeurs pour le modèle utilisé.
- `st.code(...)` montre un exemple d'appel à la méthode et l'inspection de sa forme et de ses premières valeurs.

### Expandeur 2 — normalisation

La norme euclidienne d'un vecteur mesure sa longueur. La normalisation divise le vecteur par sa norme pour obtenir une longueur égale à 1, si la norme est non nulle.

- Les deux appels `st.latex` affichent la formule de la norme puis celle de la normalisation.
- Le texte précise qu'InsightFace fournit déjà `normed_embedding`.
- `st.code` montre la ligne qui récupère ce vecteur normalisé.

### Expandeur 3 — similarité cosinus

La formule affichée compare l'orientation de deux vecteurs :

\[
\operatorname{sim}(\mathbf{a},\mathbf{b})
=\frac{\sum_{i=1}^{d}a_i b_i}
{\sqrt{\sum_{i=1}^{d}a_i^2}\sqrt{\sum_{i=1}^{d}b_i^2}}
\]

Lorsque les deux vecteurs sont normalisés, leurs normes valent 1 ; la formule se réduit alors à :

\[
\operatorname{sim}(\mathbf{a},\mathbf{b})
=\sum_{i=1}^{d}a_i b_i
\]

- `st.latex` affiche les formules.
- `st.code` montre l'instruction NumPy utilisée pour calculer le score.
- Le texte précise qu'une similarité élevée signifie une proximité numérique plus forte, pas une certitude d'identité.

### Expandeur 4 — seuil de décision

La règle affichée est :

\[
\text{Décision} =
\begin{cases}
\text{Correspondance}, & s \geq t \\
\text{Visage inconnu}, & s < t
\end{cases}
\]

- \(s\) est le score calculé.
- \(t\) est le seuil choisi.
- Un seuil trop bas augmente les fausses acceptations.
- Un seuil trop haut augmente les faux rejets.
- Le code rappelle que `0.60` n'est pas une garantie de sécurité et doit être évalué sur des données de test.

### Expandeur 5 — déroulement de la reconnaissance

Le bloc `st.code` illustre les étapes :
- extraire l'embedding candidat ;
- parcourir les profils ;
- récupérer l'embedding de référence ;
- calculer le produit scalaire ;
- retenir le meilleur score et vérifier le seuil.

Il s'agit d'un extrait pédagogique du processus déjà implémenté dans l'onglet Reconnaître, et non d'une seconde fonction exécutée automatiquement.

### Limites affichées

Le dernier `st.markdown` rappelle :
- la qualité, l'éclairage et l'angle influencent les résultats ;
- le score n'est pas une probabilité ;
- une photo affichée à la caméra peut être utilisée pour tenter de tromper le système : aucune détection de présence réelle (*liveness detection*) n'est implémentée ;
- les photos et embeddings sont des données biométriques sensibles ;
- la base SQLite est locale et non chiffrée par ce code.

## 4.15. Pied de page — lignes 580 à 584

```python
st.divider()
st.caption(
    "FaceAuth-AI — Prototype pédagogique. "
    "Les profils sont conservés dans faceauth.db sur cette machine."
)
```

- `st.divider()` sépare le contenu principal du pied de page.
- `st.caption` affiche un rappel discret.
- Le texte indique le statut de prototype et l'emplacement local des profils.

---

# 5. Fichier `requirements.txt`

## 5.1. À quoi sert-il ?

Ce fichier donne à `pip` la liste des bibliothèques nécessaires à l'application et les versions à installer. Une ligne comme `streamlit==1.65.0` signifie que le projet demande précisément cette version de Streamlit.

La commande suivante lit ce fichier :

```bat
python -m pip install -r requirements.txt
```

- `python -m pip` lance `pip` avec l'interpréteur Python actif.
- `install` demande l'installation.
- `-r` signifie « lire les exigences depuis un fichier ».
- `requirements.txt` est le fichier à lire.

## 5.2. Rôle des dépendances présentes dans le dépôt

Le fichier publié contient les dépendances directes et transitives avec des versions précises. Une dépendance **directe** est utilisée explicitement par le projet ; une dépendance **transitive** est nécessaire à l'une des bibliothèques installées.

| Paquet | Rôle principal |
|---|---|
| `streamlit` | Construire l'interface web interactive. |
| `insightface` | Détecter et analyser les visages, produire les embeddings. |
| `opencv-python` | Décoder et traiter les images. |
| `numpy` | Tableaux numériques, conversion des embeddings et calculs. |
| `pandas` | Préparer le tableau des scores de reconnaissance. |
| `onnxruntime` | Exécuter les modèles ONNX utilisés par InsightFace. |
| `onnx` | Outils et format de représentation des modèles ONNX. |
| `scikit-image` | Fonctions de traitement d'image utilisées dans l'écosystème de dépendances. |
| `scipy` | Fonctions scientifiques et numériques requises par certaines dépendances. |
| `pillow` | Lecture et manipulation d'images utilisées par certaines bibliothèques. |
| `ImageIO` | Entrées/sorties d'images pour certaines dépendances. |
| `tifffile` | Lecture/écriture d'images TIFF, dépendance de traitement d'image. |
| `ml_dtypes` | Types numériques utilisés dans certains calculs de modèles. |
| `flatbuffers` | Format de sérialisation utilisé par certains composants logiciels. |
| `protobuf` | Sérialisation de messages de données pour certaines bibliothèques. |
| `altair` | Bibliothèque de graphiques utilisée par l'écosystème Streamlit. |
| `pyarrow` | Formats de données en colonnes et échanges de données. |
| `pydeck` | Visualisation de données géographiques, notamment dans Streamlit. |
| `narwhals` | Couche de compatibilité entre bibliothèques de données. |
| `networkx` | Manipulation de graphes, dépendance possible d'outils scientifiques. |
| `jsonschema` | Validation de données selon un schéma JSON. |
| `jsonschema-specifications` | Schémas utilisés par l'écosystème JSON Schema. |
| `referencing` | Résolution de références de schémas. |
| `rpds-py` | Structures de données persistantes utilisées par certaines dépendances. |
| `attrs` | Outils de définition de classes et d'attributs. |
| `anyio` | Abstraction de programmation asynchrone. |
| `starlette` | Composants web asynchrones utilisés par certains serveurs et outils. |
| `uvicorn` | Serveur ASGI ; il n'est pas nécessaire de lancer séparément pour l'application Streamlit actuelle. |
| `httptools` | Outils HTTP bas niveau utilisés par certains serveurs. |
| `websockets` | Support de communications WebSocket dans des dépendances. |
| `h11` | Implémentation du protocole HTTP/1.1 utilisée par des dépendances. |
| `requests` | Effectuer des requêtes HTTP ; pas utilisé directement dans le code principal actuel. |
| `urllib3` | Couche HTTP utilisée par `requests` et d'autres bibliothèques. |
| `certifi` | Fournit un ensemble de certificats d'autorités de confiance. |
| `idna` | Gestion des noms de domaine internationalisés. |
| `charset-normalizer` | Détection/normalisation d'encodages de texte. |
| `click` | Création d'interfaces en ligne de commande, utilisée par des outils comme Streamlit. |
| `colorama` | Gestion de couleurs dans les terminaux, surtout sous Windows. |
| `Jinja2` | Moteur de templates utilisé par certaines bibliothèques web. |
| `MarkupSafe` | Échappement de texte dans les templates. |
| `itsdangerous` | Signature et sécurisation de certaines données sérialisées. |
| `packaging` | Comparaison et traitement des versions de paquets. |
| `python-dateutil` | Fonctions de dates et heures. |
| `six` | Couche de compatibilité Python utilisée par des dépendances. |
| `typing_extensions` | Extensions de typage pour Python. |
| `lazy-loader` | Chargement différé de composants par certaines bibliothèques. |
| `tqdm` | Barres de progression dans les traitements. |
| `toml` | Lecture de fichiers de configuration TOML. |
| `tzdata` | Données de fuseaux horaires. |
| `watchdog` | Surveillance de fichiers, utilisée par Streamlit pour le rechargement. |

### Pourquoi figer les versions avec `==` ?

Cela aide à reproduire un environnement comparable d'une installation à l'autre. En contrepartie, des versions très récentes ou incompatibles avec la version de Python peuvent rendre l'installation difficile sur une autre machine. Si une installation échoue, il faut examiner le message précis plutôt que modifier toutes les versions au hasard.

**À noter :** `sqlite3`, `pathlib` et `print` font partie de Python et ne nécessitent pas d'installation séparée. Il ne faut donc pas ajouter ces noms au fichier `requirements.txt`.

---

# 6. Fichier `.gitignore`

Le contenu actuel du fichier est organisé par catégories :

```gitignore
# Environnement Python
.venv/
venv/
__pycache__/
*.py[cod]

# Données biométriques et base locale
faceauth.db
*.db
*.sqlite
*.sqlite3
photo_test.jpg

# Fichiers temporaires
.streamlit/secrets.toml
.env

# Système et éditeur
.vscode/
.DS_Store
Thumbs.db
```

Explication ligne par ligne :

- `# Environnement Python` : commentaire de classement ; Git ne l'interprète pas comme un motif.
- `.venv/` : ignore l'environnement virtuel nommé `.venv`.
- `venv/` : ignore également un environnement appelé `venv`.
- `__pycache__/` : ignore les dossiers de cache créés par Python.
- `*.py[cod]` : ignore les fichiers Python compilés, par exemple `.pyc`, `.pyo` et `.pyd`.
- `# Données biométriques et base locale` : annonce la catégorie suivante.
- `faceauth.db` : ignore explicitement la base du projet.
- `*.db` : ignore les fichiers se terminant par `.db`.
- `*.sqlite` et `*.sqlite3` : ignorent d'autres extensions courantes de bases SQLite.
- `photo_test.jpg` : évite de publier la photo locale utilisée pour le test.
- `# Fichiers temporaires` : commentaire de section.
- `.streamlit/secrets.toml` : évite de publier un fichier de secrets de Streamlit.
- `.env` : évite de publier les variables de configuration qui pourraient contenir des secrets.
- `# Système et éditeur` : commentaire de section.
- `.vscode/` : ignore les paramètres locaux de VS Code.
- `.DS_Store` : ignore un fichier de métadonnées de macOS.
- `Thumbs.db` : ignore un fichier de cache de miniatures Windows.

**Important :** `.gitignore` n'efface pas un fichier déjà suivi par Git. Si une base ou une photo a déjà été commitée, il faut la retirer de l'index Git et vérifier l'historique du dépôt. Les données biométriques ne doivent pas être publiées, même si le dépôt est public.

---

# 7. Fichier `README.md`

`README.md` est la présentation du dépôt sur GitHub. Contrairement aux fichiers `.py`, il ne contient pas les instructions exécutées par l'application.

Le Markdown utilise notamment :

- `# Titre` : titre principal ;
- `## Sous-titre` : section ;
- `- élément` : liste à puces ;
- `1. élément` : liste numérotée ;
- `` `nom_fichier` `` : code en ligne ;
- les blocs entourés de trois accents graves pour afficher du code ;
- `**texte**` : texte en gras ;
- `[texte](URL)` : lien cliquable ;
- `>` : citation ou avertissement.

Le README du projet explique l'objectif, les fonctionnalités, les technologies, le fonctionnement mathématique, l'installation, l'utilisation, les limites et la confidentialité. Il aide une personne qui découvre le dépôt à comprendre le projet avant de lire les fichiers Python.

---

# 8. Notions Python essentielles rencontrées dans ce projet

| Élément | Signification | Exemple du projet |
|---|---|---|
| `import` | Charger une bibliothèque ou un module. | `import numpy as np` |
| `from ... import ...` | Importer un élément précis d'un module. | `from face_engine import FaceEngine` |
| `class` | Définir une classe. | `class FaceEngine:` |
| `def` | Définir une fonction ou une méthode. | `def get_profiles():` |
| `self` | Référence à l'objet courant dans une méthode. | `self.model` |
| `return` | Renvoyer une valeur à l'appelant. | `return profiles` |
| `if / elif / else` | Exécuter un bloc selon une condition. | Vérifier le seuil. |
| `for` | Parcourir une collection. | Parcourir les profils. |
| `try / except` | Intercepter et gérer des erreurs. | Gérer une photo invalide. |
| `with` | Encadrer l'utilisation d'une ressource. | Connexion SQLite ou onglet Streamlit. |
| `@decorateur` | Modifier le comportement d'une fonction. | `@st.cache_resource` |
| `[]` | Liste, indexation ou sélection. | `results[0]`, `embedding[:20]` |
| `{}` | Dictionnaire ou ensemble selon le contexte. | Dictionnaire des profils. |
| `()` | Appel de fonction, tuple ou regroupement d'expressions. | `FaceEngine()`, `(user_id,)` |
| `==` | Tester l'égalité. | `source == "Utiliser..."` |
| `is None` | Tester l'absence de valeur. | `photo_file is None` |
| `>=` | Comparer avec « supérieur ou égal ». | `best_score >= threshold` |
| `#` | Commencer un commentaire Python ou Git ignore. | `# Étape 1...` |
| `f"..."` | Construire une chaîne avec insertion de variables. | `f"Profil {name}"` |
| `"""..."""` | Chaîne multilignes ; utilisée aussi pour docstrings et requêtes SQL. | Requête `CREATE TABLE`. |

---

# 9. Comprendre la logique mathématique

## 9.1. De la photo au vecteur

Une image est constituée de pixels. Le réseau neuronal d'InsightFace transforme les caractéristiques du visage en un vecteur de nombres :

\[
\mathbf{x}=(x_1,x_2,\ldots,x_d)
\]

Pour le modèle utilisé, `d` vaut généralement 512. Les valeurs ne sont pas des mesures directement interprétables comme « couleur des yeux » ou « largeur du nez » ; elles forment une représentation apprise par le modèle.

## 9.2. Similarité cosinus

Pour deux vecteurs \(\mathbf{a}\) et \(\mathbf{b}\) :

\[
\operatorname{sim}(\mathbf{a},\mathbf{b})
=\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}
\]

Dans le code, `np.dot(reference, candidate)` réalise le produit scalaire. Comme le moteur utilise `normed_embedding`, les vecteurs sont normalisés et le produit scalaire donne directement la similarité cosinus, si leurs dimensions correspondent.

## 9.3. Décision

Le programme cherche d'abord le profil avec le score le plus élevé. Il accepte ensuite une correspondance uniquement si :

\[
\text{score maximal} \geq \text{seuil}
\]

Le seuil ne rend pas le modèle infaillible. Il faut mesurer les faux positifs et les faux négatifs sur un jeu de test représentatif avant toute utilisation sensible.

---

# 10. Limites et précautions de sécurité

1. **Prototype pédagogique :** le projet n'est pas un système d'authentification certifié.
2. **Pas de détection de présence réelle :** une photo ou un écran pourrait tromper le système. Il n'y a pas de mécanisme de *liveness detection* dans cette version.
3. **Données biométriques sensibles :** les photos et les embeddings sont stockés localement dans `faceauth.db`.
4. **Pas de chiffrement applicatif :** le code présenté ne chiffre pas la base SQLite.
5. **Seuil expérimental :** la valeur 0,60 doit être testée et ne doit pas être considérée comme une norme de sécurité.
6. **Une photo par identifiant :** un nouvel enregistrement du même identifiant remplace le précédent.
7. **Un seul visage par photo :** le moteur refuse zéro ou plusieurs visages.
8. **Dépôt public :** ne jamais y ajouter de photos personnelles, d'embeddings, de base SQLite, de mots de passe ou de secrets.

Pour un usage réel, il faudrait notamment évaluer le modèle, mettre en place une détection d'attaque, protéger les données au repos, contrôler les accès, gérer le consentement et examiner les obligations légales applicables aux données biométriques.

---

# 11. Commandes utiles sous Windows CMD

Depuis le dossier du projet :

```bat
.venv\Scripts\activate
python test_face.py
streamlit run app.py
```

- La première commande active l'environnement virtuel.
- La deuxième lance le test du moteur, à condition que `photo_test.jpg` soit présent.
- La troisième démarre l'application Streamlit.

Après avoir ajouté ce document à la racine du projet :

```bat
git status
git add EXPLICATION_CODE.md
git commit -m "docs: explain project files and code"
git push
```

- `git status` montre les fichiers modifiés ou non suivis.
- `git add` place le nouveau document dans la zone de préparation.
- `git commit` crée un point d'historique local.
- `git push` envoie le commit vers GitHub.

---

## Conclusion

Le projet est organisé en séparant l'interface (`app.py`), le traitement facial (`face_engine.py`), le test (`test_face.py`), les dépendances (`requirements.txt`), les exclusions Git (`.gitignore`) et la documentation (`README.md`). Cette séparation aide à comprendre, tester et faire évoluer le code.

Le principe central est : **photo → détection du visage → embedding → comparaison des vecteurs → décision selon un seuil**. La qualité du résultat dépend du modèle, des photos et du seuil choisi ; le score ne constitue pas à lui seul une preuve d'identité.
