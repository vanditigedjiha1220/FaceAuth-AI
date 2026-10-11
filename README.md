# 🔐 FaceAuth-AI — Reconnaissance faciale et identification automatique

**FaceAuth-AI** est une application de reconnaissance faciale développée avec Python. Elle permet d'enregistrer des profils faciaux, de comparer les représentations numériques des visages et d'identifier automatiquement la personne enregistrée qui correspond le mieux à une nouvelle image.

Le projet a également une vocation pédagogique : il permet d'explorer les embeddings faciaux, la similarité cosinus et le mécanisme de décision utilisé lors de la reconnaissance.

> **Statut :** prototype pédagogique en développement. Ce projet n'est pas un système de sécurité certifié.

## Objectifs du projet

- Enregistrer plusieurs profils faciaux à partir de photographies.
- Capturer une image avec une caméra ou importer une image existante.
- Détecter un visage et extraire sa représentation numérique (*embedding*).
- Comparer un visage à plusieurs profils enregistrés.
- Afficher les scores de similarité et proposer la meilleure correspondance.
- Signaler un visage inconnu lorsque le meilleur score reste inférieur au seuil choisi.
- Conserver localement les profils et permettre leur suppression.
- Expliquer les fondements mathématiques du processus de reconnaissance.

## Fonctionnalités

- **Enregistrement facial :** association d'un identifiant à une photo et à son embedding.
- **Reconnaissance automatique :** comparaison du visage reçu avec les profils enregistrés.
- **Importation et caméra :** deux moyens de fournir une image.
- **Gestion des profils :** consultation et suppression des profils.
- **Explication mathématique :** visualisation des embeddings, de leur dimension, de la similarité cosinus et du seuil de décision.
- **Stockage local :** utilisation de SQLite pour conserver les profils entre les sessions.

## Technologies utilisées

| Technologie | Rôle |
|---|---|
| Python | Langage principal |
| Streamlit | Interface utilisateur interactive |
| InsightFace | Détection faciale et extraction des embeddings |
| OpenCV | Traitement et décodage des images |
| NumPy | Manipulation des vecteurs et calculs numériques |
| SQLite | Stockage local des profils |
| Pandas | Présentation des résultats de comparaison |

## Comment fonctionne le système ?

Le processus de reconnaissance suit plusieurs étapes.

### 1. Acquisition de l'image

L'utilisateur importe une photographie ou prend une photo avec la caméra.

### 2. Détection du visage

InsightFace détecte le visage dans l'image. Le prototype attend une image contenant un seul visage.

### 3. Extraction de l'embedding

Le modèle transforme le visage en un vecteur numérique représentant des caractéristiques apprises par le réseau neuronal.

On peut représenter ce vecteur par :

\[
\mathbf{x}=(x_1,x_2,\ldots,x_d)
\]

Avec le modèle utilisé dans ce projet, la dimension de l'embedding est généralement de 512 valeurs.

### 4. Calcul de la similarité cosinus

Le vecteur du visage reçu est comparé à celui de chaque profil enregistré.

La formule générale est :

\[
\operatorname{sim}(\mathbf{a},\mathbf{b})=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}
\]

Les embeddings utilisés ici sont normalisés. La similarité se calcule donc par leur produit scalaire :

\[
\operatorname{sim}(\mathbf{a},\mathbf{b})
=\mathbf{a}\cdot\mathbf{b}
=\sum_{i=1}^{d}a_i b_i
\]

### 5. Décision

Le système classe les profils selon leur score de similarité et examine le meilleur résultat.

- Si le meilleur score atteint le seuil défini, le système propose le profil correspondant.
- Si aucun score n'atteint ce seuil, le visage est signalé comme inconnu.

Le score de similarité n'est pas une probabilité d'identité. Le seuil doit être évalué expérimentalement sur des images représentatives.

## 🏗️ Architecture du projet

```text
FaceAuth-AI/
├── app.py              # Interface Streamlit et logique applicative
├── face_engine.py      # Détection faciale et extraction des embeddings
├── test_face.py        # Test du moteur facial
├── requirements.txt    # Dépendances Python
├── .gitignore          # Fichiers exclus de Git
├── README.md           # Documentation du projet
└── faceauth.db         # Base SQLite locale, créée à l'exécution
```

La base `faceauth.db` est générée localement et ne doit pas être publiée sur GitHub.

## ⚙️ Installation

### Prérequis

- Python 3.11 recommandé pour l'environnement de développement.
- Git.
- Une caméra ou des photographies de visages pour les tests.
- Une connexion Internet lors de la première installation, notamment pour télécharger les dépendances et les modèles nécessaires.

### 1. Cloner le dépôt

```bash
git clone https://github.com/vanditigedjiha1220/FaceAuth-AI.git
cd FaceAuth-AI
```

### 2. Créer un environnement virtuel

Sous Windows avec CMD :

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre normalement dans le navigateur. Si elle ne s'ouvre pas automatiquement, utilise l'adresse locale indiquée dans le terminal.

## 🧪 Utilisation

1. Ouvrir l'onglet **Enregistrer**.
2. Saisir un identifiant et importer une photo ou utiliser la caméra.
3. Enregistrer plusieurs personnes avec des identifiants distincts.
4. Ouvrir l'onglet **Reconnaître**.
5. Soumettre une nouvelle photo et lancer la reconnaissance.
6. Examiner le meilleur score et les résultats de comparaison.
7. Consulter **Logique mathématique** pour comprendre la vectorisation et les calculs.
8. Utiliser **Gérer les profils** pour consulter ou supprimer les profils enregistrés.

## 🔬 Expérimentations possibles

Ce projet peut servir de base à plusieurs travaux pratiques :

- Étudier l'influence du seuil sur les fausses acceptations et les faux rejets.
- Comparer plusieurs photos d'une même personne selon l'éclairage et l'angle.
- Mesurer le temps nécessaire à l'extraction et à la comparaison des embeddings.
- Comparer les résultats obtenus sur des personnes enregistrées et inconnues.
- Étudier l'effet du nombre de profils sur le temps de recherche.
- Évaluer les performances du système sur un jeu de test contrôlé.

## 🔒 Sécurité et confidentialité

Les photographies et les embeddings faciaux sont des données biométriques sensibles.

Dans cette version, ils sont conservés dans une base SQLite locale. Le stockage n'est pas chiffré par défaut.

- Ne publiez pas les photographies personnelles, les embeddings ou le fichier `faceauth.db`.
- Protégez l'accès à la machine et aux fichiers locaux.
- Utilisez des images de test avec l'autorisation des personnes concernées.
- Ne considérez pas une simple comparaison faciale comme une preuve d'identité.
- Prévoyez des mécanismes de détection des attaques par photo ou écran avant tout usage sensible.

## 🚧 Limites actuelles

- Le système dépend de la qualité des photographies et de la détection faciale.
- Les variations d'éclairage, d'angle et d'expression peuvent affecter les scores.
- Le seuil utilisé est expérimental et n'a pas été validé pour un déploiement réel.
- La reconnaissance faciale seule ne garantit pas que la personne soit physiquement présente devant la caméra.
- Les profils sont stockés localement et ne sont pas synchronisés entre plusieurs appareils.

## 🔮 Perspectives d'amélioration

- Autoriser plusieurs photos de référence par personne.
- Évaluer les performances avec un jeu de données de test contrôlé.
- Ajouter des graphiques de distribution des scores et des métriques d'évaluation.
- Mettre en place une protection contre les attaques de présentation (*liveness detection*).
- Renforcer la protection et la gestion des données biométriques.
- Étudier des méthodes d'indexation pour accélérer les recherches sur un grand nombre de profils.

## 👨‍💻 Auteur

**VANDI TIGE DJIHA**

Étudiant en Data Science , intéressé par l'intelligence artificielle, l'apprentissage automatique et le développement d'applications intelligentes.

GitHub : [@vanditigedjiha1220](https://github.com/vanditigedjiha1220)

## 📄 Licence

Aucune licence n'est encore définie pour ce dépôt. Avant de redistribuer ou de réutiliser le code, consultez les conditions applicables aux bibliothèques et aux modèles utilisés.