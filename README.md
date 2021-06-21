# kalman-filter-gpu

## Compiler le code

Il se peut que la version 10 de g++ soit requise.
```
export CXX=/usr/local/bin/g++-10
```
avant d'appeler cmake.

Sur une machine debian :
```
sudo apt install cmake
mkdir build && cd build
cmake ..
make
```

## Utilisation

Les images données en entrée au programme sont des images .PGM ASCII (P2) de uint8 (0 - 255).

Il est possible de convertir n'importe quel fichier image dans ce format grâce à `ImageMagick` avec la commande :

```
convert <img_path> -colorspace Gray -compress none <out_path>.pgm
```

Le programme s'utilise de la façon suivante :

```
./kalman-gpu [mode] <imagepath>
```

**Modes :**

- `--gpu`     active le mode parallèle GPU
- `--batch` active le mode batch CPU
- `--sequential` active le mode séquentiel CPU

La sortie est une image de labels au format .PGM dans le fichier `out.pgm`.\
Pour colorer chaque label d'une couleur différente, on peut utiliser le script fourni :

```
python3 output_to_rgb.py <out_img>
```
