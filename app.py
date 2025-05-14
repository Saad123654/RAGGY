import streamlit as st
import requests
import time  # Pour simuler un délai d'attente

# Titre de la page
st.title("Bienvenue sur RAGGY !")
st.subheader("Entrez votre requête et obtenez une réponse du système RAG.")

# Créer un formulaire d'entrée
query = st.text_input("Votre requête :", "")

# Si une requête est soumise, on appelle l'API ou un traitement
if query:
    st.write(f"Vous avez soumis la requête : {query}")

    # Affichage de la barre de progression
    progress_bar = st.progress(0)  # Initialisation de la barre à 0%

    try:
        # Pour simuler un délai d'attente (retirer ceci pour l'usage en production)
        for percent_complete in range(0, 101, 10):  # Progression de 10% à chaque itération
            time.sleep(0.2)  # Simuler un petit délai (remplacer par la logique réelle si nécessaire)
            progress_bar.progress(percent_complete)  # Met à jour la barre de progression

        # Appel à ton API FastAPI
        response = requests.get(f"http://127.0.0.1:8000/query/", params={"query": query})

        # Si la requête est réussie (status code 200), on affiche la réponse
        if response.status_code == 200:
            result = response.json()  # Parse la réponse JSON de l'API
            st.write(f"**Réponse du système RAG :** {result['response']}")
        else:
            st.error(f"Erreur API: {response.status_code}")
    
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {str(e)}")
