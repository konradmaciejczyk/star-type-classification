#Konrad Maciejczyk, 2021
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers
from algorithms import KNN, Nearest_centroid

def load_data(path, verbose = True):
    datapd = pd.read_csv("stars.csv")
    star_types = ["Brown Dwarf", "Red Dwarf", "White Dwarf", "Main Sequence", "Supergiant", "Hypergiant"]
    
    if verbose:
        print("Number of stars in dataset: {}\nFeatues: ".format(datapd['Star type'].count()), end="")
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
    plt.ylabel('Number of instances')
    plt.xlabel("Star color")
    plt.show()

def cat_data_handle(datapd, verbose=True):
    data = pd.get_dummies(data=datapd)
    
    if verbose:
        data.head()

    return data

def data_split(data, random_state = 42, test_size=.2, verbose=True):
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

def run_knn(x_train, x_test, y_train, y_test, neighbors = 3):
    my_knn_clf = KNN(k=neighbors)
    my_knn_clf.fit(x_train, y_train)
    print("Testing set score accuracy: {:.2f}% for KNN({})". format(my_knn_clf.score(x_test, y_test)*100, neighbors))

def run_nearest_centroid(x_train, x_test, y_train, y_test):
    my_min_centroid_clf = Nearest_centroid()
    my_min_centroid_clf.fit(x_train, y_train)
    print("Testing set score accuracy: {:.2f}% for NearestCentroid". format(my_min_centroid_clf.score(x_test, y_test)*100))

def build_model(layers_, optimizer="adam", loss = 'sparse_categorical_crossentropy', metrics=['accuracy']):
    model = models.Sequential()

    for layer in layers_:
        model.add(layers.Dense(layer[0], activation=layer[1]))
    model.add(layers.Dense(6, activation="softmax"))

    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    return model

def run_neural_network(model, x_train, x_test, y_train, y_test, epochs, batch_size, validation_split=.2, verbose=0):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=validation_split, verbose=verbose)

    loss_test, test_acc = model.evaluate(x_test, y_test)
    print("Testing accuracy: {:.2f}% for NN(epochs = {}, batch_size = {}, validation_split = {}".format(test_acc*100, epochs, batch_size, validation_split))

if __name__ == "__main__":
    #Loading star dataset from .csv file and saving to pandas DataFrame
    datapd, star_types = load_data("./stars.csv")
    #Plotting dataset using matplotlib
    show_data(datapd)
    #Handling categorical data using one-hot encoding
    datapd = cat_data_handle(datapd)
    #Spliting dataset into training and testing datasets
    x_train, x_test, y_train, y_test = data_split(datapd)
    #Normalizing data
    x_train, x_test = normalize_data(x_train, x_test)

    print("\n", '*'*60, sep="", end="\n")
    #Running KNN
    run_knn(x_train, x_test, y_train, y_test, 1)
    #Running NearestCentroid
    run_nearest_centroid(x_train, x_test, y_train, y_test)

    #Building Neural Network
    print("\n\n", '*'*60, sep="", end="\n")
    model = build_model([(128, "relu"), (64, "relu")])
    #Running Neural Network
    run_neural_network(model, x_train, x_test, y_train, y_test, 32, 16)




