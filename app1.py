import streamlit as st
import pickle
import pandas as pd

# Wczytanie wytrenowanego modelu
model_filename = "model.pkl"
model = pickle.load(open(model_filename, 'rb'))

# Słowniki dla konwersji klas, płci i portów zaokrętowania
pclass_d = {1: "Pierwsza", 2: "Druga", 3: "Trzecia"}
sex_d = {"male": "Mężczyzna", "female": "Kobieta"}
embarked_d = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}

# Wczytanie danych z pliku CSV
data_filename = "DSP_1.csv"
data = pd.read_csv(data_filename)

def main():
    st.set_page_config(page_title="Titanic Survival Predictor")
    st.title("Titanic Survival Predictor")

    st.sidebar.title("Parametry podróżnika")
    pclass_radio = st.sidebar.radio("Klasa", data['Pclass'].unique(), format_func=lambda x: pclass_d[x])
    sex_radio = st.sidebar.radio("Płeć", data['Sex'].unique(), format_func=lambda x: sex_d[x])
    age_slider = st.sidebar.slider("Wiek", min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=int(data['Age'].mean()))
    sibsp_slider = st.sidebar.slider("Liczba rodzeństwa i/lub partnera", min_value=int(data['SibSp'].min()), max_value=int(data['SibSp'].max()))
    parch_slider = st.sidebar.slider("Liczba rodziców i/lub dzieci", min_value=int(data['Parch'].min()), max_value=int(data['Parch'].max()))
    fare_slider = st.sidebar.slider("Cena biletu", min_value=data['Fare'].min(), max_value=data['Fare'].max(), value=data['Fare'].mean())
    embarked_radio = st.sidebar.radio("Port zaokrętowania", data['Embarked'].unique(), format_func=lambda x: embarked_d[x])

    # Przewidywanie przeżycia na podstawie danych użytkownika
    user_data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][prediction]

    st.subheader("Wynik predykcji")
    if prediction == 1:
        st.write("Osoba przetrwała.")
    else:
        st.write("Osoba nie przetrwała.")

    st.write(f"Prawdopodobieństwo przetrwania: {probability:.2f}")


if __name__ == "__main__":
    main()