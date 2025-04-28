import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from MLP_class import MLP
import matplotlib.pyplot as plt

st.title("🧠 Entraînement d'un MLP (Multi-Layer Perceptron)")

@st.cache_data
def load_data():
    return pd.read_csv("data/data.csv", sep=';')

df = load_data()

df_filtered = df[df["Target"].isin(["Graduate", "Dropout"])].copy()
y = df_filtered["Target"].apply(lambda x: 1 if x == "Graduate" else 0)

# Liste des features par défaut
default_features = [
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (grade)',
    'Scholarship holder',
    'Tuition fees up to date',
    'Application mode',
    'Course',
    'Previous qualification (grade)',
    "Mother's occupation",
    'Admission grade',
    'Age at enrollment',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 2nd sem (evaluations)',
    'Debtor',
    'Gender',
    'Curricular units 1st sem (credited)',
    'Previous qualification',
    "Father's qualification",
    'Unemployment rate',
    "Mother's qualification",
    "Father's occupation"
]

# Liste des colonnes disponibles (hors target)
available_features = df_filtered.columns.drop('Target')

# Sélection via multiselect avec valeurs par défaut
selected_features = st.multiselect(
    "✅ Choisissez les colonnes à utiliser comme variables d'entrée :",
    options=available_features,
    default=[col for col in default_features if col in available_features]  # sécurise si une colonne est manquante
)


if not selected_features:
    st.warning("🚫 Veuillez sélectionner au moins une variable pour l'entraînement.")
    st.stop()

X = df_filtered[selected_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
X_train = X_train.T
X_test = X_test.T
y_train = y_train.values.reshape(1, -1)
y_test = y_test.values.reshape(1, -1)

# Choix du nombre de couches
st.markdown("### ⚙️ Configuration du modèle")
nb_couches = st.selectbox("🧱 Nombre de couches cachées", [1, 2, 3, 4, 5], index=2)

# Initialiser une liste vide
hidden_layers = []

# Afficher les sliders dynamiquement après le choix
with st.expander("🔧 Configurer les neurones de chaque couche", expanded=True):
    for i in range(nb_couches):
        neurones = st.slider(f"➡️ Couche cachée {i+1} : neurones", min_value=1, max_value=128, value=16, key=f"neurons_layer_{i}")
        hidden_layers.append(neurones)

# Autres paramètres
learning_rate = st.number_input("🚀 Learning rate", 0.0001, 1.0, 0.01, step=0.001, format="%.4f")
n_iter = st.number_input("🔁 Nombre d'itérations", 100, 10000, 3000, step=100)

# Bouton d'entraînement
if st.button("🎯 Entraîner le modèle"):
    mlp = MLP(
        input_dim=X_train.shape[0],
        output_dim=1,
        hidden_layers=tuple(hidden_layers),
        learning_rate=learning_rate,
        n_iter=n_iter
    )

    with st.spinner("⏳ Entraînement en cours..."):
        history = mlp.fit(X_train, y_train)

    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    train_acc = accuracy_score(y_train.flatten(), y_pred_train.flatten())
    test_acc = accuracy_score(y_test.flatten(), y_pred_test.flatten())

    st.success("✅ Entraînement terminé")
    st.write(f"📊 **Accuracy entraînement** : `{train_acc:.2%}`")
    st.write(f"🧪 **Accuracy test** : `{test_acc:.2%}`")

    # --- Partie matrice de confusion ---

    import plotly.graph_objects as go

    # --- Partie matrice de confusion (Plotly) ---

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test.flatten(), y_pred_test.flatten())

    st.subheader("📊 Matrice de confusion")

    # Créer la heatmap avec Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Abandon", "Réussi"],  # Colonnes
        y=["Abandon", "Réussi"],  # Lignes
        colorscale='Blues',
        text=cm,  # Pour afficher les valeurs dans les cases
        texttemplate="%{text}",
        hoverinfo="skip"
    ))

    fig.update_layout(
        xaxis_title="Prédit",
        yaxis_title="Vrai",
        title="Matrice de confusion",
        autosize=False,
        width=500,
        height=500,
    )

    # Afficher dans Streamlit
    st.plotly_chart(fig)






