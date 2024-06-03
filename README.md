# GradeADreamer
Text to 3D generation

## Environment Setup
```bash
# from yml file
conda env create --file=environment.yml
conda activate gd

# Gaussian Splatting
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization

# simple-knn
pip install git+https://github.com/camenduru/simple-knn/

# MVDream
pip install git+https://github.com/bytedance/MVDream

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

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
```

## Export to VDO
```bash
kire logs/astro/astro_refined_mesh.obj --save_video logs/astro_output_vdo.mp4 --wogui
```