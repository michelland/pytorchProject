# Rapport TP
## Yohan Michelland, Victor Favre

### Question 1
- data_train est de taille 63000,784
- label_train est de taille 63000,10
- W contient 784 x 10 éléments, soit 7840 éléments
    28x28 pixels = 784 et 10 sorties
- b est de taille 1,10 : un élément pour chaque sortie 
- X est de taille batch_size, nombre_pixels, c'est a dire 
    de taille 5,784
- Y est de taille 5,10
- t est de taille 5,10
- grad = t - Y donc de la meme dimension 5,10

Partie test du modèle:
- x est une seule ligne de data_test, donc de dimension 1,784
- y est la multiplication de 1,784 x 784,10 donc de dimension 1,10
- t est de dimension 1,10


### Question 2

eta = 0.01 => 0.0979
eta = 0.0001 => 0.8613
eta = 0.00001 => 0.8539
eta = 0.000001 => 0.8014
Il faut un eta pas trop petit et pas trop grand. De plus, il faut corréler eta et nombre de batch

w = 0.001 => 0.8533
w = 0.01 => 0.8551
w = 0.1 => 0.7660
Il faut des poids initiaux pas trop grands et pas trop petits




