from gradeadreamer.appearance import appeareance_pass

import torch
import numpy as np

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--gpu", required=False, default="0")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    if "gpu_id" not in opt:
        opt.gpu_id = args.gpu
    appeareance_pass(opt)