# GradeADreamer

[[arXiv](https://arxiv.org/abs/2406.09850)] [Project Page Coming Soon...]

A high-quality text-to-3D generation with low occurrence rate of the Multi-face Janus Problem and fast generation time.

## Result Examples

A car made out of sushi |  A blue jay standing on a large basket of rainbow macarons
:-------------------------:|:-------------------------:
![car_sushi](eval/images/car_sushi/gad.gif) |  ![blue_jay_macaron](eval/images/blue_jay_macaron/gad.gif)

A DSLR photo of an ice cream sundae |  A DSLR photo of a plate of fried chicken and waffles with maple syrup on them
:-------------------------:|:-------------------------:
![icecream](eval/images/icecream/gad.gif) |  ![waffle](eval/images/waffle/gad.gif)

## Resource Requirements

The model was tested on a single RTX 3090 GPU, achieving a generation time of around 30 minutes per asset. We measured the memory peak, and it requires at least 16 GB of GPU VRAM to execute the entire pipeline.

## Environment Setup
```bash
# from yml file
conda env create --file=environment.yml
conda activate GradeADreamer

# Gaussian Splatting
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization

# simple-knn
pip install git+https://github.com/camenduru/simple-knn/

# MVDream
pip install git+https://github.com/bytedance/MVDream

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit
```

## Run (Entire pipeline)
```bash
./run.sh -opt astro
```

## Run each stage separately
```bash
# Stage 1 : Create Prior Point Clouds [MVDream + SDS]
python main_prior.py --config configs/astro/prior.yaml

# Stage 2 : Gaussian Splatting Optimization [Stable Diffusion + SDS]
python main_gs.py --config configs/astro/gs.yaml

# Stage 3 : Texture Optimization [Stable Diffusion + SDS]
python main_appearance.py --config configs/astro/appearance.json
```

## Export to VDO
```bash
kire logs/astro/astro_appearance/dmtet_mesh/mesh.obj --save_video logs/astro/astro_output_vdo.mp4 --wogui
```

## Evaluation

You will just need to move the needed gif representations of models to evaluate (360° around the 3D model) in the `eval/images` folder and arrange them by folders inside like `eval/images/astro`, also make sure to respect the name of the associated config for automatic search.

```
python eval.py
```

## Acknowledgement

This repository is built on top of [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) and [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) repositories. We would like to thank for their incredible works ❤️.

## Footnote

This work is the final project of the Deep Learning 2024 Spring course at Tsinghua University 🟣. We would like to express our sincere gratitude to this course !
