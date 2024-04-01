import streamlit as st
import pickle
from datetime import datetime

# import bibliotek
import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Wczytaj model
filename = "model.pkl"  # Poprawiona nazwa pliku
model = pickle.load(open(filename, 'rb'))

# Słowniki dla konwersji danych
pclass_d = {1: "Pierwsza", 2: "Druga", 3: "Trzecia"}
embarked_d = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
sex_d = {"female": "Kobieta", "male": "Mężczyzna"}

def main():
    st.set_page_config(page_title="Titanic Survival Predictor")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/300px-RMS_Titanic_3.jpg")

    with overview:
        st.title("Titanic Survival Predictor")

    with left:
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])

    with right:
        age_slider = st.slider("Wiek", min_value=1, max_value=100)  # Zwiększony zakres wieku
        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=0, max_value=10)
        parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=0, max_value=6)
        fare_slider = st.slider("Cena biletu", min_value=0, max_value=513, step=1)  # Zwiększony zakres ceny biletu

    embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2, format_func=lambda x: embarked_d[x])

    data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival[0]] * 100))  # Poprawka indeksu

if __name__ == "__main__":
    main()
