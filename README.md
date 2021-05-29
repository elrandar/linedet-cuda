# kalman-filter-gpu

Les images données en entrée au programme sont des images .PGM ASCII (P2) de uint8 (0 - 255).

Il est possible de convertir n'importe quel fichier image dans ce format grâce à `ImageMagick` avec la commande :

```
convert <img_path> -colorspace Gray -compress none <out_path>.pgm
```