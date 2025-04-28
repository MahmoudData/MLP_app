import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Exploration des données - Projet ANN")

@st.cache_data
def load_data():
    return pd.read_csv("data.csv", sep=';')

df = load_data()

st.write(f"🧮 Le dataset contient **{df.shape[0]} lignes** et **{df.shape[1]} colonnes**.")

if st.checkbox("📄 Afficher les premières lignes du dataset"):
    st.dataframe(df.head())

colonnes = df.columns.tolist()

st.subheader("📌 Visualisation de la distribution d'une variable")
colonne = st.selectbox("Choisis une colonne", colonnes)

# Affichage dynamique selon le type de colonne
if df[colonne].dtype == 'object':
    # Graphique pour les colonnes catégorielles
    data = df[colonne].value_counts().reset_index()
    data.columns = [colonne, 'Nombre']
    fig = px.bar(data, x=colonne, y='Nombre', title=f"Répartition de '{colonne}'")
else:
    # Histogramme pour colonnes numériques
    fig = px.histogram(df, x=colonne, nbins=20, title=f"Distribution de '{colonne}'")


st.plotly_chart(fig, use_container_width=True)

# Matrice de corrélation
st.subheader("🧊 Matrice de corrélation (variables numériques)")

numeric_df = df.select_dtypes(include=['number'])

corr_matrix = numeric_df.corr()

# Heatmap 
fig = px.imshow(corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Corrélation entre les variables numériques",
                aspect="auto")

st.plotly_chart(fig, use_container_width=True)