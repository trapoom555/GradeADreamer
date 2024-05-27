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
./run.sh -opt icecream
```

## Run each stage separately
```bash
# Stage 1 : Gaussian Splatting with VSD
python main.py --config configs/icecream.yaml

# Stage 2 : Texture Optimization with VSD
python main2.py --config configs/icecream.yaml
```

## Export to VDO
```bash
kire logs/icecream/icecream_refined_mesh.obj --save_video logs/icecream_output_vdo.mp4 --wogui
```