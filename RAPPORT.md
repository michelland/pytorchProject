# Rapport TP
## Yohan Michelland, Victor Favre

## Partie 1
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

## Partie 2
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

### Question 4

TODO


## Partie 3

Nous avons effectué les tests avec les hyper-paramètres suivant:
`batch_size = 256`, `nb_epochs = 10`, `eta = 0.001`
De plus, pour réaliser tous nos tests, nous avons utilisé la méthode de gradient Adam, une fonction d'activation ReLU, et la fonction de coût cross entropie.
De plus tous ces tests ont été effectués sur GPU

#### 1. Couches

Premièrement nous allons chercher qu'elle est l'influence du nombre de neurones en gardant le meme layout. Nous aurons donc pour les tests suivant 2 couches cachées avec la couche 1 ayant 2 fois plus de neurones que le 2. 

| Neuronnes total | Couche 1 | Couche 2 | Score |
| --- | ----------- | ---------| ------ |
| 30 | 20 | 10 | 94.24
| 60 | 40 | 20 | 96.17
| 120 | 80 | 40 | 97.36
| 240 | 160 | 80 | 97.89
| 240 | 1280 | 640 | 98.10

On remarque donc comme attendu que plus le nombre de neuronnes augmente plus le score augmente. On remarque aussi que le gain de score n'est pas linerairement lié au nombre de neuronnes et diminue progressivement plus ceux-ci augmentent en nombre.


Pour analyser de la forme des couches cachées nous avons décidé de garder un nombre fixe de neuronnes ( à 1 près pour les cas impaire ). Nous testerons ensuite différent layout avec ce nombre de neuronnes.

Nous avons fixé arbitrairement ce nombre a 240. Il a été choisi pour nous permettre d'augmenter le nombre de couches en évitant que celle-ci n'ait un nombre trop faible de neurones et aussi car d'apres nos tests précédant il obtient un score raisonnablement haut et permettra de distinguer plus facilement l'influence du layout

Nous allons d'abord étudier comme les repartition des neuronnes dans les couches de notre layout de test incluence le score

| Layout | répartition des Neuronne | Score |
| :---: | :-----------: | :------: |
| 2 couches | 160 / 80 | 97.89
| 2 couches | 220 / 20 | 97.80
| 2 couches | 120 / 120 | 97.78
| 2 couches | 80 / 160 | 97.25

Nous remarquons qu'il est préférable pour une couche n+1 d'avoir autant ou moins de neuronnes que la couche n

Nous allons mainteant modifier le nombre de couches et la répartions des neuronnes.

| Layout | répartition des Neuronne | Score |
| :---: | :-----------: | :------: |
| 3 couches | 140  / 80 / 20 | 97.61
| 3 couches | 180  / 20 / 20 | 97.2
| 3 couches | 160  / 40 / 20 | 97.64
| 3 couches | 100  / 100 / 40 | 97.5
| 3 couches | 120  / 80 / 60 | 97.37
| 4 couches | 80 / 80 / 40 / 20 | 96.99

On remarque que l'ajout de couches nous est plutôt négatif. Aucune des versions avec 3 couches ou 4 n'arrive à égaler le score obtenu avec 2 couches.

Nous souhaitions finalement verifier si a diminution des scores venait du nombre de couches ou d'un nombre de neurones peut-être trop faible pour prendre avantage de plus de couches. Nous avons donc choisi les meilleurs layouts 3 couches et 2 couches et les avons testé avec ~4400 neurones

| Layout | répartition des Neurones | Score |
| :---: | :-----------: | :------: |
| 2 couches | 2933 / 1466 | 98.45
| 3 couches | 3200  / 800 / 400 | 98.35

On peut conclure que layout avec 2 couches est plus intéressant que des layout avec plus de couches. Cependant, on remarque une diminution des differences avec une augmentation du nombre de neurones

#### 2. Learning rate

Pour tester l'influence du Learning rate nous allons nous baser sur le meilleur layout précédant avec 240 neurones.

| Learning rate | Score |
| :---: | :------: |
| 0.1 | 56.93
| 0.01 | 97.56
| 0.001 | 97.89
| 0.0001 | 94.26

On en déduit que le learning rate ne doit pas être trop bas ou trop haut. Dans notre cas 0.001 donne les meilleurs résultats. Nous supposons que la raison de la baisse de gains pour un learning rate de 0.0001 est du a une combinaison d'un nombre de neurones et d'époques trop faible

Pour verifier notre hypothèse, nous allons effectuer les memes tests avec un nombre de neurones (4399) ou d'époque (50) plus élevé. 

Test avec 4399 neurones :

| Learning rate | Score |
| :---: | :------: |
| 0.1 | 10.07
| 0.01 | 97.37
| 0.001 | 98.21
| 0.0001 | 98.14

Test avec 100 epoques :

| Learning rate | Score |
| :---: | :------: |
| 0.1 | 10.02
| 0.01 | 97.94
| 0.001 | 98.13
| 0.0001 | 97.56

D'après les tests la modification d'un seul des paramètres à modifier dans notre hypothèse entraine dans les deux cas un gain de précision.

Maintenant vérifions si nous mélangeons les deux :

| Learning rate | Score |
| :---: | :------: |
| 0.001 | 98.79
| 0.0001 | 98.60

Suite à cette fusion, nous remarquons une amélioration qui est largement supérieur a l'un ou a l'autre des cas ce qui valide que l'un et l'autre on individuellement ou conjointement un impact positif sur le score.

#### 3. Poids

Dans ces tests, il a été utilisé le layout de base (2 couches 160/80) ainsi que des poids identiques sur chaque couche.


| Poids_max | Score |
| :---: | :------: | 
| 1 | 92.19
| 0.1 | 97.69
| 0.01 | 97.14
| 0.001 | 96.43
| 0.0001 | 96.52

Nous remarquons donc que les poids ont une influence importante sur le score et que, comme le learning rate, un choix de poids judicieux (ni trop haut ou ni trop bas) est nécessaire.

De plus les poids ici initialisés obtiennent tous des scores inférieurs a ceux attribué par défaut par pytorch.

