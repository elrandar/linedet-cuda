# kalman-filter-gpu

## Compiling

This code requires `g++` version `10`.
```
export CXX=/usr/local/bin/g++-10
```
Before calling `cmake`.

On a debian computer:
```
sudo apt install cmake
mkdir build && cd build
cmake ..
make
```

## Usage

Input images are `uint8` .PGM ASCII (P2), Range is (0 - 255). 

Using `ImageMagick`, it is possible to convert an image in this format using the following command:

```
convert <img_path> -colorspace Gray -compress none <out_path>.pgm
```

The tool is used in the following manner:

```
./kalman-gpu [mode] <imagepath>
```

**Modes :**

- `--gpu`     activate GPU mode
- `--batch` activate batch CPU mode (CPU mode emulating GPU)
- `--sequential` activate sequential CPU mode (Original implementation)

The output is a label image in the `.PGM` format. It is stored in the `out.pgm` file.\
To display the output using python, the following script can be used :
```
python3 output_to_rgb.py <out_img>
```
