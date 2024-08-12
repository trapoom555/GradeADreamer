from gradeadreamer.appearance import appeareance_pass

import torch
import numpy as np

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="./configs/appearance.yaml", help="path to the yaml config file")
    parser.add_argument("--gpu", required=False, default="0")
    parser.add_argument("--prompt", required=True, help="prompt")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    opt.gpu_id = args.gpu
    opt.text = args.prompt + " DSLR, realistic, 4K"
    appeareance_pass(opt)