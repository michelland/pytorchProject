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
Avant d elire la suite il est important de noté que les test ont été fait sut GPU.

Premièrement nous alons chercher qu'elle est l'influence du nombre de neuronnes en gardant le meme layout. Nous aurons donc pour les tests suivant 2 couches cachées avec la couche 1 ayant 2 fois plus de neuronnes que le 2. 

| Neuronnes total | Couche 1 | Couche 2 | Score |
| --- | ----------- | ---------| ------ |
| 30 | 20 | 10 | 94.24
| 60 | 40 | 20 | 96.17
| 120 | 80 | 40 | 97.36
| 240 | 160 | 80 | 97.89
| 240 | 1280 | 640 | 98.10

On remarque donc comme attendu que plus le nombre de neuronnes augmente plus le score augmente. On remarque aussi que le gain de score n'est pas linerairement lié au nombre de neuronnes et diminue progressivement plus ceux-ci augmentent en nombre.


Pour analyser de la forme des couches cachées nous avons décidé de garder un nombre fixe de neuronnes ( à 1 près pour les cas impaire ). Nous testerons ensuite différent layout avec ce nombre de neuronnes.

Ce nombre a decidé d'etre fixé arbitrairement a 240. Il a été choisi pour nous permettre d'augmenté le nombre de couches en evitant que c'elle ci n'est un nombre trop faible de neuronnes et aussi car d'apres nos test précedant il obtient un score raisonablement haut et permetra de distinguer plus facilement l'influence du layout

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

On remarque que l'ajout de couches nous est plutot détrimentale. Aucune des version avec 3 couches ou 4 n'arrive a égaler le score obtenue avec 2 couches.

Nous souhaitions finalement verifier si a diminution des scores vennait du nombre de couches ou d'un nombre de neuronnes peut etre trop faible pour prendre avantage de plus couche. Nous avons donc choisi le meilleur layout 3 couches et 2 couches et les avons testé avec 

| Layout | répartition des Neuronne | Score |
| :---: | :-----------: | :------: |
| 2 couches | 2933 / 1466 | 98.45
| 3 couches | 3200  / 800 / 400 | 98.35

On peux conclure que layout avec 2 couches est plus intéressant que des layout avec plus de couches. Cependant on remarque une diminution des difference avec une augmentation du nombres de neuronnes


