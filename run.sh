export CUDA_VISIBLE_DEVICES=1

python main.py --config configs/text_mv.yaml prompt="a DSLR photo of a corgi" save_path=corgi --wogui
