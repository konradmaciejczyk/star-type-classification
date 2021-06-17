import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers
from algorithms import KNN, Nearest_centroid

def load_data(verbose = True):
    datapd = pd.read_csv("stars.csv")
    star_types = ["Brown Dwarf", "Red Dwarf", "White Dwarf", "Main Sequence", "Supergiant", "Hypergiant"]
    
    if verbose:
        print("Ilosć rekordów: {}\nCechy: ".format(datapd['Star type'].count()), end="")
        for feature in datapd.keys():
            print(feature, end="     ")
        print("\nStar types: ", end="")
        for typ in star_types:
            print(typ, end="     ")
        print("\n")
        datapd.head()

    return datapd, star_types

def show_data(datapd):
    fig = plt.figure(figsize=(20,10))
    i=1
    data_to_draw = datapd.drop(['Star color'], axis=1)
    for feature in data_to_draw.keys():
        plt.subplot(3, 3, i)
        i += 1
        plt.hist(data_to_draw[feature], bins=100)
        plt.ylabel('ilość wystąpień')
        plt.xlabel(feature)
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(30,10))
    plt.hist(datapd['Star color'], bins=100)
    plt.ylabel('ilość wystąpień')
    plt.xlabel("Star color")
    plt.show()

def cat_data_handle(datapd, verbose=True):
    data = pd.get_dummies(data=datapd)
    
    if verbose:
        data.head()

    return data

def data_split(data, verbose=True):
    X = data.drop("Star type", axis = 1).to_numpy()
    y = data['Star type'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=.2)

    if verbose:
        print("x_train={}, x_test={}, y_train={}, y_test={}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    return x_train, x_test, y_train, y_test

def normalize_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test

def run_knn(x_train, x_test, y_train, y_test):
    my_knn_clf = KNN(k=1)
    my_knn_clf.fit(x_train, y_train)
    print("Testing set score accuracy: {:.2f}%". format(my_knn_clf.score(x_test, y_test)*100))

def run_nearest_centroid(x_train, x_test, y_train, y_test):
    my_min_centroid_clf = Nearest_centroid()
    my_min_centroid_clf.fit(x_train, y_train)
    print("Testing set score accuracy: {:.2f}%". format(my_min_centroid_clf.score(x_test, y_test)*100))

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(6, activation="softmax"))

    model.compile(optimizer = "adam", loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def run_neural_network(model, x_train, x_test, y_train, y_test):
    history = model.fit(x_train, y_train, epochs=256, batch_size=32, validation_split=.2, verbose=0)

    loss_test, test_acc = model.evaluate(x_test, y_test)
    print("Testing accuracy: {:.2f}%".format(test_acc*100))

if __name__ == "__main__":
    datapd, star_types = load_data()
    show_data(datapd)
    datapd = cat_data_handle(datapd)
    x_train, x_test, y_train, y_test = data_split(datapd)
    x_train, x_test = normalize_data(x_train, x_test)

    run_knn(x_train, x_test, y_train, y_test)
    run_nearest_centroid(x_train, x_test, y_train, y_test)




