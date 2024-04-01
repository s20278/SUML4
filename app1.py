import streamlit as st
import pickle
import pandas as pd

# Wczytanie wytrenowanego modelu
model_filename = "model.pkl"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Słowniki do mapowania wartości na czytelne etykiety
pclass_d = {1: "Pierwsza", 2: "Druga", 3: "Trzecia"}
embarked_d = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
sex_d = {"female": "Kobieta", "male": "Mężczyzna"}


def main():
    st.set_page_config(page_title="Titanic Survival Predictor")

    st.title("Titanic Survival Predictor")

    # Wybór cech przez użytkownika
    pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
    sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
    age_slider = st.slider("Wiek", min_value=1, max_value=70, step=1)  # Zmiana max_value na int
    sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=0, max_value=8, step=1)
    parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=0, max_value=6, step=1)
    fare_slider = st.slider("Cena biletu", min_value=0, max_value=93, step=1)  # Zmiana max_value na int
    embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2, format_func=lambda x: embarked_d[x])

    # Przygotowanie danych do predykcji
    input_df = pd.DataFrame({
        'Pclass': [pclass_radio],
        'Sex': [sex_radio],
        'Age': [age_slider],
        'SibSp': [sibsp_slider],
        'Parch': [parch_slider],
        'Fare': [fare_slider],
        'Embarked': [embarked_radio]
    })

    # Wykonanie kodowania kategorycznego
    input_df = pd.get_dummies(input_df, columns=['Sex', 'Embarked'])

    # Przekonwertowanie ramki danych na tablicę numpy
    input_data = input_df.values

    # Predykcja
    survival = model.predict(input_data)
    s_confidence = model.predict_proba(input_data)

    # Wyświetlenie wyników
    st.subheader("Czy taka osoba przeżyłaby katastrofę?")
    st.subheader("Tak" if survival[0] == 1 else "Nie")
    st.write("Pewność predykcji: {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
