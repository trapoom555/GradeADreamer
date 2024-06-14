# GradeADreamer
Text to 3D generation

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

This is the final project of the Deep Learning 2024 Spring course at Tsinghua University. 🟣 We would like to express our sincere gratitude to this course for providing us with a robust foundational knowledge of deep learning and offering us the opportunity to undertake a project aligned with our interests. We are particularly grateful to Professor Xiaolin Hu and Professor Jun Zhu, the course lecturers, for their invaluable guidance. Additionally, we extend our appreciation to Professor Hualiu Ping and Professor Liu Yongjin for supplying the necessary computational resources. Finally, we would like to thank the 54 participants in our user study for their valuable contributions.
