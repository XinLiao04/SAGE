import os
import yaml
from Runner import run_hatemm, run_mhclip_bl, run_mhclip_yt

if __name__ == "__main__":
    with open("./config/config.yaml", "r") as f:                          # To Do
        config = yaml.safe_load(f)

    dataset = "hatemm"                                                    # To Do
    if dataset == 'hatemm':
        run_hatemm(config)
    elif dataset == 'mhclip-bl': 
        num_classes = 2
        run_mhclip_bl(config, num_classes)
    elif dataset == 'mhclip-yt':
        num_classes = 2
        run_mhclip_yt(config, num_classes)
    else:
        raise Exception('Dataset Not Defined!! Please check the config file.')