import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from face_engine import FaceEngine


# =========================================================
# 1. CONFIGURATION ET BASE DE DONNEES
# =========================================================

st.set_page_config(
    page_title="FaceAuth-AI",
    page_icon="🔐",
    layout="wide"
)

DB_PATH = Path(__file__).with_name("faceauth.db")


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


def delete_profile(user_id):
    """Supprimer définitivement un profil de la base locale."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM profiles WHERE user_id = ?",
            (user_id,)
        )


# =========================================================
# 2. CHARGEMENT DU MODELE
# =========================================================

init_db()


@st.cache_resource
def load_face_engine():
    return FaceEngine()


st.title("🔐 FaceAuth-AI")
st.caption(
    "Reconnaissance faciale, représentation numérique "
    "et explication mathématique"
)

try:
    engine = load_face_engine()
except Exception as e:
    st.error(f"Impossible de charger le moteur facial : {e}")
    st.stop()


# Recharger les données persistantes depuis SQLite.
profiles = get_profiles()

st.metric("Personnes enregistrées", len(profiles))

tab_register, tab_recognize, tab_manage, tab_math = st.tabs([
    "💾 👤 Enregistrer",
    "🔎🤖 Reconnaître",
    "👤📁 Gérer les profils",
    "🧮 Logique mathématique"
])


# =========================================================
# 3. ENREGISTREMENT
# =========================================================

with tab_register:
    st.subheader("Enregistrer une personne")

    st.write(
        "Choisis un identifiant, puis importe une photo "
        "ou prends une photo avec la caméra."
    )

    user_id = st.text_input(
        "Nom ou identifiant de la personne",
        placeholder="Ex. Vandi",
        key="register_id"
    )

    source = st.radio(
        "Source de la photo",
        ["Importer une photo", "Utiliser la caméra"],
        key="register_source"
    )

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

    if photo_file is not None:
        photo_bytes = photo_file.getvalue()

        st.image(
            photo_bytes,
            caption="Photo à enregistrer",
            width=300
        )

    if st.button("Enregistrer le profil", type="primary"):
        if not user_id.strip():
            st.warning("Saisis un nom ou un identifiant.")
        elif photo_file is None:
            st.warning("Choisis une photo ou utilise la caméra.")
        else:
            try:
                photo_bytes = photo_file.getvalue()

                # Étape 1 : transformer la photo en embedding.
                embedding = engine.extract_embedding(photo_bytes)

                # Étape 2 : enregistrer photo et embedding.
                clean_id = user_id.strip()

                save_profile(
                    clean_id,
                    photo_bytes,
                    embedding
                )

                st.success(
                    f"Le profil « {clean_id} » a été enregistré."
                )

                st.write("**Informations sur l'embedding créé**")
                st.write(f"Dimension du vecteur : {embedding.size}")
                st.write(f"Type numérique : {embedding.dtype}")

                with st.expander(
                    "Voir les premières valeurs de cet embedding"
                ):
                    st.code(
                        np.array2string(
                            embedding[:20],
                            precision=6,
                            separator=", "
                        )
                    )
                    st.caption(
                        "Affichage des 20 premières valeurs seulement. "
                        "Le vecteur complet est conservé dans la base."
                    )

                st.rerun()

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Erreur pendant l'enregistrement : {e}")

    st.info(
        "Si tu enregistres à nouveau un identifiant déjà existant, "
        "son ancienne photo et son ancien embedding seront remplacés."
    )


# =========================================================
# 4. RECONNAISSANCE AUTOMATIQUE
# =========================================================

with tab_recognize:
    st.subheader("Identifier une personne")

    if not profiles:
        st.info("Enregistre d'abord au moins une personne.")
    else:
        source_query = st.radio(
            "Source de la photo à reconnaître",
            ["Importer une photo", "Utiliser la caméra"],
            key="recognize_source"
        )

        if source_query == "Importer une photo":
            query_photo = st.file_uploader(
                "Importer la photo à identifier",
                type=["jpg", "jpeg", "png"],
                key="recognize_upload"
            )
        else:
            query_photo = st.camera_input(
                "Prendre la photo à identifier",
                key="recognize_camera"
            )

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

        if query_photo is not None:
            st.image(
                query_photo.getvalue(),
                caption="Photo à identifier",
                width=300
            )

        if st.button("Lancer la reconnaissance", type="primary"):
            if query_photo is None:
                st.warning("Fournis une photo à analyser.")
            else:
                try:
                    # A. Embedding du visage reçu.
                    candidate = engine.extract_embedding(
                        query_photo.getvalue()
                    )

                    # B. Comparaison avec tous les profils.
                    results = []

                    for name, profile in profiles.items():
                        reference = profile["embedding"]

                        if reference.shape != candidate.shape:
                            continue

                        # Les embeddings sont normalisés.
                        similarity = float(
                            np.dot(reference, candidate)
                        )

                        results.append({
                            "Personne": name,
                            "Similarité cosinus": similarity
                        })

                    if not results:
                        st.error(
                            "Aucun profil compatible à comparer."
                        )
                    else:
                        results.sort(
                            key=lambda item: item["Similarité cosinus"],
                            reverse=True
                        )

                        best = results[0]
                        best_name = best["Personne"]
                        best_score = best["Similarité cosinus"]

                        st.subheader("Résultat")

                        if best_score >= threshold:
                            st.success(
                                f"Meilleure correspondance : {best_name}"
                            )
                        else:
                            st.error(
                                "Visage inconnu : aucun profil "
                                "n'a atteint le seuil minimal."
                            )

                        st.metric(
                            "Meilleur score de similarité",
                            f"{best_score:.4f}"
                        )

                        st.caption(
                            "Le score mesure la similarité entre "
                            "deux embeddings. Ce n'est pas une "
                            "probabilité d'identité."
                        )

                        st.subheader("Comparaison avec tous les profils")

                        df_results = pd.DataFrame(results)
                        df_results["Seuil atteint"] = (
                            df_results["Similarité cosinus"] >= threshold
                        )

                        st.dataframe(
                            df_results,
                            use_container_width=True,
                            hide_index=True
                        )

                        # C. Montrer le calcul et la vectorisation.
                        with st.expander(
                            "Voir la vectorisation et le calcul "
                            "effectué pour cette photo",
                            expanded=True
                        ):
                            st.write(
                                "**1. Embedding de la photo reçue**"
                            )
                            st.write(
                                f"Dimension : {candidate.size}"
                            )
                            st.write(
                                f"Norme du vecteur : "
                                f"{np.linalg.norm(candidate):.6f}"
                            )

                            st.code(
                                np.array2string(
                                    candidate[:20],
                                    precision=6,
                                    separator=", "
                                )
                            )

                            st.caption(
                                "Les 20 premières valeurs sont affichées ; "
                                "le modèle utilise le vecteur complet."
                            )

                            st.write(
                                "**2. Calcul de la similarité**"
                            )

                            st.latex(
                                r"s(\\mathbf{a},\\mathbf{b})"
                                r"=\\frac{\\mathbf{a}\\cdot\\mathbf{b}}"
                                r"{\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|}"
                            )

                            reference = profiles[best_name]["embedding"]

                            st.write(
                                f"Dimension du vecteur de référence : "
                                f"{reference.size}"
                            )
                            st.write(
                                f"Norme du vecteur de référence : "
                                f"{np.linalg.norm(reference):.6f}"
                            )

                            st.code(
                                "similarity = np.dot(reference, candidate)"
                            )

                            st.write(
                                f"Score calculé : {best_score:.6f}"
                            )
                            st.write(
                                f"Seuil sélectionné : {threshold:.2f}"
                            )

                            if best_score >= threshold:
                                st.write(
                                    "Décision : le score dépasse le seuil."
                                )
                            else:
                                st.write(
                                    "Décision : le score ne dépasse pas "
                                    "le seuil."
                                )

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Erreur pendant la reconnaissance : {e}")


# =========================================================
# 5. GESTION ET SUPPRESSION DES PROFILS
# =========================================================

with tab_manage:
    st.subheader("Profils enregistrés")

    if not profiles:
        st.info("Aucun profil enregistré.")
    else:
        st.write(
            "Les profils ci-dessous sont chargés depuis la base SQLite."
        )

        for name, profile in profiles.items():
            with st.expander(f"👤 {name}"):
                st.image(
                    profile["photo"],
                    caption=f"Photo de référence : {name}",
                    width=220
                )

                st.write(
                    f"Date d'enregistrement : {profile['created_at']}"
                )
                st.write(
                    f"Dimension de l'embedding : "
                    f"{profile['embedding'].size}"
                )

                if st.button(
                    f"Supprimer le profil {name}",
                    key=f"delete_{name}",
                    type="secondary"
                ):
                    delete_profile(name)
                    st.success(f"Le profil « {name} » a été supprimé.")
                    st.rerun()

    st.warning(
        "La suppression retire de la base locale la photo et "
        "l'embedding associés au profil."
    )


# =========================================================
# 6. EXPLICATION MATHEMATIQUE
# =========================================================

with tab_math:
    st.subheader("Comment fonctionne la reconnaissance faciale ?")

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

    st.divider()

    with st.expander("1. Qu'est-ce qu'un embedding ?", expanded=True):
        st.write(
            "Un embedding est une représentation numérique calculée "
            "par un réseau neuronal à partir des caractéristiques "
            "du visage. Ce n'est pas une simple liste de pixels."
        )

        st.latex(
            r"\\mathbf{x} = [x_1, x_2, \\ldots, x_d]"
        )

        st.write(
            "La dimension d dépend du modèle. Pour le modèle utilisé "
            "ici, elle est généralement égale à 512. Le programme "
            "affiche la dimension réelle dans les résultats."
        )

        st.code("""
embedding = engine.extract_embedding(photo_bytes)
print(embedding.shape)
print(embedding[:20])
""")

    with st.expander("2. Pourquoi normaliser le vecteur ?"):
        st.write(
            "La normalisation ramène le vecteur à une norme égale "
            "à 1. Elle permet de comparer principalement la direction "
            "des vecteurs plutôt que leur amplitude."
        )

        st.latex(
            r"\\|\\mathbf{x}\\|_2"
            r"=\\sqrt{\\sum_{i=1}^{d}x_i^2}"
        )

        st.latex(
            r"\\widehat{\\mathbf{x}}"
            r"=\\frac{\\mathbf{x}}{\\|\\mathbf{x}\\|_2}"
        )

        st.write(
            "Dans notre application, InsightFace fournit déjà "
            "normed_embedding, qui est un embedding normalisé."
        )

        st.code("""
embedding = faces[0].normed_embedding
""")

    with st.expander("3. Comment calculer la similarité cosinus ?"):
        st.write(
            "La similarité cosinus compare l'orientation de deux "
            "vecteurs. Elle s'obtient en divisant leur produit "
            "scalaire par le produit de leurs normes."
        )

        st.latex(
            r"\\operatorname{sim}(\\mathbf{a},\\mathbf{b})"
            r"=\\frac{\\sum_{i=1}^{d}a_i b_i}"
            r"{\\sqrt{\\sum_{i=1}^{d}a_i^2}"
            r"\\sqrt{\\sum_{i=1}^{d}b_i^2}}"
        )

        st.write(
            "Lorsque les deux vecteurs sont normalisés, leurs normes "
            "valent 1. Le calcul se simplifie donc en un produit scalaire."
        )

        st.latex(
            r"\\operatorname{sim}(\\mathbf{a},\\mathbf{b})"
            r"=\\sum_{i=1}^{d}a_i b_i"
        )

        st.code("""
similarity = float(np.dot(reference, candidate))
""")

        st.write(
            "Une similarité élevée indique une plus grande proximité "
            "entre les représentations, mais ne garantit pas à elle "
            "seule qu'il s'agit de la même personne."
        )

    with st.expander("4. Comment fonctionne le seuil ?"):
        st.latex(
            r"\\text{Décision} = "
            r"\\begin{cases}"
            r"\\text{Correspondance}, & s \\geq t \\\\"
            r"\\text{Visage inconnu}, & s < t"
            r"\\end{cases}"
        )

        st.write(
            "s désigne le score de similarité et t le seuil choisi. "
            "Un seuil trop bas augmente les risques de fausses "
            "correspondances ; un seuil trop haut peut rejeter "
            "des personnes légitimes."
        )

        st.write(
            "Le seuil doit être évalué sur des photos de test variées. "
            "La valeur 0,60 utilisée par défaut n'est pas une garantie "
            "de sécurité."
        )

    with st.expander("5. Que fait le programme lors d'une reconnaissance ?"):
        st.code("""
# Extraire le vecteur du visage reçu.
candidate = engine.extract_embedding(photo_bytes)

# Comparer avec chaque profil enregistré.
for name, profile in profiles.items():
    reference = profile["embedding"]
    similarity = float(np.dot(reference, candidate))

# Retenir le meilleur score, puis vérifier le seuil.
""")

        st.write(
            "Le programme calcule un score par profil, classe les "
            "scores du plus élevé au plus faible et ne propose une "
            "correspondance que si le meilleur score dépasse le seuil."
        )

    st.divider()

    st.subheader("Limites à connaître")

    st.markdown("""
    - Le modèle peut commettre des erreurs, notamment selon la qualité
      des images, l'éclairage, l'angle et les caractéristiques des visages.
    - Le score de similarité n'est pas une probabilité.
    - Le système ne prouve pas qu'une photo provient d'une personne
      présente devant la caméra : une protection contre les attaques
      par photo ou écran serait nécessaire.
    - Les photos et embeddings sont des données biométriques sensibles.
      Cette version les stocke localement dans SQLite, sans chiffrement.
      Protège le fichier `faceauth.db` et ne l'utilise pas pour un
      contrôle d'accès réel sans mesures de sécurité supplémentaires.
    """)


# =========================================================
# 7. PIED DE PAGE
# =========================================================

st.divider()
st.caption(
    "FaceAuth-AI — Prototype pédagogique. "
    "Les profils sont conservés dans faceauth.db sur cette machine."
)
