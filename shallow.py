import gzip, numpy, torch

if __name__ == '__main__':
    hlayer_size = 10
    batch_size = 5  # nombre de données lues à chaque fois000
    nb_epochs = 10  # nombre de fois que la base de données sera lue
    eta = 0.001  # taux d'apprentissage

    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

    # on initialise le modèle et ses poids
    w1 = torch.empty((data_train.shape[1], hlayer_size), dtype=torch.float)
    w2 = torch.empty((hlayer_size, label_train.shape[1]), dtype=torch.float)
    b1 = torch.empty((1, hlayer_size), dtype=torch.float)
    b2 = torch.empty((1, label_train.shape[1]), dtype=torch.float)

    torch.nn.init.uniform_(w1, -0.001, 0.001)
    torch.nn.init.uniform_(w2, -0.001, 0.001)
    torch.nn.init.uniform_(b1, -0.001, 0.001)
    torch.nn.init.uniform_(b2, -0.001, 0.001)

    nb_data_train = data_train.shape[0]
    nb_data_test = data_test.shape[0]
    indices = numpy.arange(nb_data_train, step=batch_size)
    for n in range(nb_epochs):
        # on mélange les (indices des) données
        numpy.random.shuffle(indices)
        # on lit toutes les données d'apprentissage
        for i in indices:
            # on récupère les entrées
            x = data_train[i:i + batch_size]
            # on calcule la sortie du modèle
            y1 = 1 / (1 + numpy.exp(-torch.mm(x, w1) + b1))  #si erreur regarder ici (ordre multiplication matrice)!
            y2 = torch.mm(y1, w2) + b2

            # on regarde les vrais labels
            t = label_train[i:i + batch_size]
            # on met à jour les poids
            delta2 = (t - y2)
            delta1 = y1 * (1 - y1) * (torch.mm(delta2, w2.T))

            w1 += eta * torch.mm(x.T, delta1)
            w2 += eta * torch.mm(y1.T, delta2)
            b1 += eta * delta1.sum()
            b2 += eta * delta2.sum()

        # test du modèle (on évalue la progression pendant l'apprentissage)
        acc = 0.
        # on lit toutes les donnéees de test
        for i in range(nb_data_test):
            # on récupère l'entrée
            x = data_test[i:i + 1]
            # on calcule la sortie du modèle
            y1 = 1 / (1 + numpy.exp(-torch.mm(x, w1) + b1))  # si erreur regarder ici (ordre multiplication matrice)!
            y2 = torch.mm(y1, w2) + b2
            # on regarde le vrai label
            t = label_test[i:i + 1]
            # on regarde si la sortie est correcte
            acc += torch.argmax(y2, 1) == torch.argmax(t, 1)
        # on affiche le pourcentage de bonnes réponses
        print(acc / nb_data_test)