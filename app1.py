import numpy as np
import tkinter as tk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


# main

root = tk.Tk()
canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

def main():
    label = tk.Label(root, text='Obliczenia zostały przeprowadzone', fg='green', font=('calibri', 10))
    canvas.create_window(130, 200, window=label)
    randnums = np.random.randint(0, 6, size=(2, 4))
    print(randnums, end='\n\n')

    base_data = pd.read_csv("DSP_1.csv")
    cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    data = base_data[cols].copy()
    data["Age"].fillna((data["Age"].mean()), inplace=True)
    data.dropna(subset=['Embarked'], inplace=True)
    encoder = LabelEncoder()
    data.loc[:,"Sex"] = encoder.fit_transform(data.loc[:,"Sex"])
    data.loc[:,"Embarked"] = encoder.fit_transform(data.loc[:,"Embarked"])
    y_train = data.iloc[0:600:,0]
    y_test = data.iloc[600:700:,0]
    X_train = data.iloc[0:600:,1:8]
    X_test = data.iloc[600:700:,1:8]
    model = RandomForestClassifier(criterion='gini', n_estimators=20, max_depth=7)
    model.fit(X_train, y_train)
    acc = accuracy_score(model.predict(X_test), y_test)
    print(f'Model accuracy: {acc*100:.2f} %')
    pickle.dump(model, open('./model.sv', 'wb'))

button = tk.Button(root, text='Generuj', bd=5, command=main, bg='green', fg='white')
canvas.create_window(150, 150, window=button)

root.mainloop()