# GradeADreamer

[[ArXiv](https://arxiv.org/abs/2406.09850)] [[Project Page](https://trapoom555.github.io/GradeADreamer_Project_Page/)] [ThreeStudio Integration Coming Soon...]

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

# pip indexed packages
pip install -r requirements.txt

# pip github packages
pip install -r requirements_external.txt

## Installing GradeADreamer as a pip package (optional)

This step might especially be interesting for you if you plan to use [threestudio](https://github.com/threestudio-project/threestudio).

pip install -e .

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
python main_appearance.py --config configs/astro/appearance.yaml
```

## Export to VDO
```bash
kire logs/astro/astro_appearance/dmtet_mesh/mesh.obj --save_video logs/astro/astro_output_vdo.mp4 --wogui
```

## Evaluation

You will just need to move the needed gif representations of models to evaluate (360° around the 3D model) in the `eval/images` folder and arrange them by folders inside like `eval/images/astro`, also make sure to respect the name of the associated config for automatic search.

```
python main_eval.py
```

## Acknowledgement

This repository is built on top of [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) and [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) repositories. We would like to thank for their incredible works ❤️.

## Footnote

This work is the final project of the Deep Learning 2024 Spring course at Tsinghua University 🟣. We would like to express our sincere gratitude to this course !
