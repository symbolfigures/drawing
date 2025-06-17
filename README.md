# Drawing

Scripts that 
- prepare scanned line drawings as a dataset for deep learning.
- generate immitation images from a trained model.
- make animations by calculating transformations through the model's latent space.
- fill generated images with color to create a new synthetic dataset.

Two compositional styles are supported:
- **Spec**: as demonstrated by my Second Study, each page has specimens roughly arranged on a grid, and have vertical and horizontal axes of symmetry. <sup>[1](https://symbolfigures.io/drawing/ex/1_secondstudy.png)</sup>
- **Blob**: as demonstrated by my Third Study, each page is a continuous and asymmetrical blob of shapes with no apparent orientation. <sup>[2](https://symbolfigures.io/drawing/ex/2_thirdstudy.png)</sup>

The GAN in [train/](train/) is based on an implementation by Brad Klingensmith available in his excellent [Udemy course](https://www.udemy.com/course/high-resolution-generative-adversarial-networks), which is in turn based on [ProGAN](https://arxiv.org/abs/1710.10196) with improvements from [StyleGAN2](https://arxiv.org/abs/1912.04958).

## Setup

Clone and enter the repository using your preferred method. Create a virtual environment and install the required packages.

```
git clone git@github.com:symbolfigures/drawing.git
cd drawing
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, freeze.txt has all the dependencies and versions explicit.

## [data](data/)

`cd data`

The scans should be prepared as
- PNG format
- RGB mode  
- 300, 600, or 1200 DPI

#### [spec_tile.py](data/spec_tile.py)  
Dissect each spec drawing into a set of **tiles** <sup>[3](https://symbolfigures.io/drawing/ex/3_spec_tile.png)</sup> or images the model will train on. An approximate grid formation allows the program to automatically capture each specimen. <sup>[4](https://symbolfigures.io/drawing/ex/4_spec_grid.png)</sup>
```
python spec_tile.py \
    <dir_in> \
	<resolution> \
    --dir_out=... \
    --dir_out_grid=...
```

#### [spec_rotateflip.py](data/spec_rotateflip.py)  
Rotate and flip each tile, multiplying the volume by eight. <sup>[5](https://symbolfigures.io/drawing/ex/5_spec_rotateflip.png)</sup>
```
python spec_rotateflip.py \
    <dir_in>
```

#### [blob_grid.py](data/blob_grid.py)  
Draw a grid on a copy of each blob drawing to show the margin within which tiles are cut, and how large the tiles will be. <sup>[6](https://symbolfigures.io/drawing/ex/6_blob_grid.png)</sup> It generates [adjustment.json](data/adjustment.json) to manually adjust the margin of each drawing.
```
python blob_grid.py \
    <dir_in> \
    <dpi> \
    --dir_out=... \
    --index=... \
    --rows=... \
    --columns=...
```

#### [blob_tile.py](data/blob_tile.py)  
Cut tiles from the blob. Each tile is cut at a random angle, and perchance flipped. `--steps` measures the overlap of adjacent tiles. <sup>[7](https://symbolfigures.io/drawing/ex/7_blob_tile.png)</sup>
```
python blob_tile.py \
    <dir_in> \
    <dpi> \
    <resolution> \
    --dir_out=... \
    --rows=... \
    --cols=... \
    --steps=...
```

#### [tfrecord.py](data/tfrecord.py)  
Convert to .tfrecord format. Output is separated into 300MB files. For small datasets, use `--size` to set a lower size to ensure there are multiple files. <sup>[8](https://symbolfigures.io/drawing/ex/8.tfrecord)</sup>
```
python tfrecord.py \
    <dir_in> \
    --dir_out=... \
    --size=...
```

#### [tfrecord_reverse.py](data/tfrecord_reverse.py)  
Convert back to .png just to make sure everything went okay.
```
python tfrecord_reverse.py \
    <dir_in> \
    --dir_out=...
```

`cd ..`

## [train](train/)

`cd train`

#### [main.py](train/main.py)  
Train the model. Provide an output folder. That folder must contain an [options.json](train/options.json) file.
```
export TF_USE_LEGACY_KERAS=1
python main.py <dir_out>
```

#### Notes:

Code:
- A few fixes were necessary to work with a newer version of tensorflow. Legacy Keras avoids the error described [here](https://github.com/keras-team/keras/issues/19246).
- The implementation by B.K. includes a lot of features not included here, such as a visualization generator.
- Part of it is copied to [gen/gan](gen/gan) in order to load the generator.

Training data:
- To fit 1024x1024 pixel images on two GPUs with 24GB memory each, the generator and discriminator are each given a dedicated GPU in [training_loop.py](train/training_loop.py).
- RGB images produce better results than grayscale or RGBA. Despite having one channel instead of three, grayscale does not improve speed or save memory.
- At 14,400 images the Second Study dataset is too small to train images larger than 256x256 pixels, regardless of DPI. At 1024x1024 resolution, the latent space is divided into homogeneous chunks, so the animation is mostly still with sudden movements.

Hyperparameters:
- Transformations through the latent space (animation) continue to improve well after the loss metrics reach their minima.
- A lower learning rate makes the animation slower, as if the latent space were less dense.
- beta_1=0.5 optimizes generator loss, but the images are best at beta_1=0.0 
- From looking at images or animations, it's hard to tell any difference between various latent sizes or weight decay rates.

`cd ..`

## [gen](gen/)

`cd gen`

#### [path.py](gen/path.py)  
Construct a path through the model's latent space, according to a selected style and parameters. All paths form a loop, except **random**. For models with up to 64 dimensions, **bitloop**, **period**, and **phase** come with a plot showing the modulation of each dimension.
- **random**: The anti-path has no continuity or direction. Use to make a synthetic dataset. <sup>[11](https://symbolfigures.io/drawing/ex/11_random.png)</sup>
- **bezier**: A set of points are randomly chosen in the latent space. Adjacent points are connected by a bezier curve with three anchors, also randomly chosen. Apparent speed varies due to the varying distance between points. <sup>[12](https://symbolfigures.io/drawing/ex/12_bezier.mp4)</sup>
- **bitloop**: Takes a bitstring and maps each bit to a dimension in the latent space, looping until it covers every dimension. If the bit is a 1, then the dimension's value is modulated over a sine wave. If the bit is a 0, the value is held constant. <sup>[13](https://symbolfigures.io/drawing/ex/13_bitloop.mp4)</sup> <sup>[14](https://symbolfigures.io/drawing/ex/14_bitloop_plot.png)</sup>
- **period**: Each dimension is modulated over a sine wave lasting 60 seconds at 30 fps. They have randomly assigned periods. The period is always a factor of 60, allowing the path to loop. <sup>[15](https://symbolfigures.io/drawing/ex/15_period.mp4)</sup> <sup>[16](https://symbolfigures.io/drawing/ex/16_period_plot.png)</sup>
- **phase**: Similar to **period**, but they all have the same period of 60 seconds, and instead vary by phase. The result is a steady tumbling motion. <sup>[17](https://symbolfigures.io/drawing/ex/17_phase.mp4)</sup> <sup>[18](https://symbolfigures.io/drawing/ex/18_phase_plot.png)</sup>
```
python path.py \
	<model> \
	<style> \
	--dir_out=... \
	--count=... \
	--segments=... \
	--frames=... \
	--bitloop=...
```

#### [rgba.py](gen/rgba.py)  
Convert a folder full of images from RGB to RGBA. White turns clear, and black turns opaque. The opaque color can also be customized.
```
python rgba.py \
    <dir_in> \
    --color=...
```

#### [mp4.py](gen/mp4.py)  
Convert a folder full of images to an mp4 video. If the images are at all transparent, include a background image or video.
```
python rgba.py \
    <dir_in> \
    --background=...
```

`cd ..`

## [fill](fill/)

`cd fill`

#### [blob_fill.py](fill/blob_fill.py)  
Given a set of random generated images,
- fill the shapes with color <sup>[19](https://symbolfigures.io/drawing/ex/19_shape.png)</sup>
- fill the lines in between <sup>[20](https://symbolfigures.io/drawing/ex/20_line.png)</sup>
- optionally blend the edges <sup>[21](https://symbolfigures.io/drawing/ex/21_blend.png)</sup>

By default, colors are picked at random from an input image via `--palette`. The `--overlay` flag picks according to location. To pick the color of a pixel (x, y) in the drawing, it takes the color of the pixel (x, y) in the palette.
```
python blob_fill.py \
    <dir_in> \
    <palette> \
    --dir_out=... \
    --overlay \
    --blend
```

#### [spec_fill.py](fill/spec_fill.py)
Similar to [blob_fill.py](fill/blob_fill.py) without overlay. Pairs of symmetric shapes are colored the same. <sup>[22](https://symbolfigures.io/drawing/ex/22_spec_fill.png)</sup>
```
python spec_fill.py \
    <dir_in> \
    --dir_out=... \
    --blend
```

#### [blend.py](fill/blend.py)
For larger datasets, blend with this instead. It requires a GPU. First, compile [blend.cu](fill/blend.cu).
```
nvcc -shared -o blend.so blend.cu --compiler-options '-fPIC'
python blend.py \
    <dir_in> \
    --dir_out=...
```

`cd ..`























