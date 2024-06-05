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

## \[Experimental\] Geometry Optimization

After Stage 2, you can run the following command for geometry optimization.

```bash
python3  main_appearance.py --config configs/astro/geometry.json
```