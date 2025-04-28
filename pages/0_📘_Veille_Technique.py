import streamlit as st

st.title("📚 Veille Technique")

with st.expander("🔍 Qu'est-ce qu'un Perceptron Multicouche (PMC) ?"):
    st.markdown("""
Le **Perceptron Multicouche (PMC)** est un réseau neuronal à **propagation avant** composé de :
- 🟢 Une couche d'entrée
- 🟡 Une ou plusieurs couches cachées
- 🔴 Une couche de sortie

Chaque couche contient un nombre variable de **neurones** qui transforment les données d'entrée jusqu'à produire une sortie (classification ou régression).
""")

with st.expander("🧠 Architecture PMC selon la tâche"):
    st.markdown("""
Selon la tâche, l'**architecture change** :
- 🔷 **Régression** : activation linéaire en sortie
- 🔶 **Classification binaire** : activation sigmoïde
- 🌈 **Classification multiclasse** : activation softmax
""")

with st.expander("📘 Définitions clés du Deep Learning"):
    st.markdown("""
- **Fonction d’activation** : transforme les signaux des neurones (ReLU, Sigmoid, Tanh, Softmax)
- **Propagation** : passage des données de la couche d'entrée à la sortie
- **Rétropropagation** : calcul de l’erreur + mise à jour des poids via le gradient
- **Loss Function** : mesure l'écart entre la prédiction et la vérité
- **Descente de gradient** : méthode d’optimisation pour réduire l’erreur
- **Vanishing Gradients** : problème où les gradients deviennent trop petits (ReLU > Sigmoid)
""")

with st.expander("⚙️ Hyperparamètres importants d’un MLP"):
    st.markdown("""
1. **Batch size** : 32-64 recommandé pour bonne généralisation  
2. **Fonction d’activation** : ReLU (couches cachées), Sigmoid / Softmax (sortie)  
3. **Poids initiaux** : He pour ReLU, Xavier pour sigmoid  
4. **Dropout** : 0.2 à 0.5 pour éviter le sur-apprentissage  
5. **Nombre de couches cachées** : 2-3 pour débuter, plus pour modèles profonds  
6. **Learning rate** : 0.001 (Adam) ou 0.01 (SGD)
7. **Optimiseur** : Adam (robuste), SGD+momentum (plus fin)
""")