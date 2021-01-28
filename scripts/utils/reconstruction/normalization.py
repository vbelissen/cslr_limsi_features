import numpy as np

# Fonction pour normaliser les donnees
# Chaque ligne correspond a un pas de temps
# Sur chaque ligne on a : x0 y0 z0 x1 y1 z1 ... x(n-1) y(n-1) z(n-1)
# avec n le nombre de points du squelette
def normalizedata(data_set):

    temp1 = sorted(list(range(0, data_set.shape[1], 3))+list(range(2, data_set.shape[1], 3)))

    X = data_set[:, temp1]
    Y = data_set[:, 1::3]

    return normalizedataXY(X,Y)

# Idem qu'avant mais ici on a separe (x,z) d'un cote (nomme X)
# et (y) d'un autre cote (nomme Y)
def normalizedataXY(X,Y):

    temp3 = X[:, 1::2].std(axis=1)
    temp4 = X[:, 0::2].std(axis=1)
    temp5 = (temp3+temp4)/2

    X = np.divide(X, np.outer(temp5, np.ones((1, X.shape[1]))))
    Y = np.divide(Y, np.outer(temp5, np.ones((1, Y.shape[1]))))

    X[:, 1::2] = X[:, 1::2] - np.outer(X[:, 1::2].mean(axis = 1), np.ones((1, int((X.shape[1])/2))))
    X[:, 0::2] = X[:, 0::2] - np.outer(X[:, 0::2].mean(axis = 1), np.ones((1, int((X.shape[1])/2))))
    Y = Y - np.outer(Y.mean(axis = 1), np.ones((1, Y.shape[1])))

    return X, Y, temp5

def normalizedataX(X):

    temp3 = X[:, 1::2].std(axis=1)
    temp4 = X[:, 0::2].std(axis=1)
    temp5 = (temp3+temp4)/2
    X = np.divide(X, np.outer(temp5, np.ones((1, X.shape[1]))))

    X[:, 1::2] = X[:, 1::2] - np.outer(X[:, 1::2].mean(axis = 1), np.ones((1, int((X.shape[1])/2))))
    X[:, 0::2] = X[:, 0::2] - np.outer(X[:, 0::2].mean(axis = 1), np.ones((1, int((X.shape[1])/2))))

    return X, temp5
