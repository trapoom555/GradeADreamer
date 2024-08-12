from gradeadreamer.gs import Trainer

import torch
import numpy as np

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="./configs/gs.yaml", help="path to the yaml config file")
    parser.add_argument("--gpu", required=False, default="0")
    parser.add_argument("--prompt", required=True, help="prompt")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    if "gpu_id" not in opt:
        opt.gpu_id = args.gpu
    opt.prompt = args.prompt + " DSLR, realistic, 4K"

    # seed
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # train
    trainer = Trainer(opt)
    trainer.train(opt.iters)