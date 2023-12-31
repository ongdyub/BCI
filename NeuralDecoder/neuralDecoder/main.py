import os
import tensorflow as tf
import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch
# from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
# from neuralDecoder._brain_to_sen import BrainToSen
from neuralDecoder._pho_to_sen import PhonemeToSen

@hydra.main(config_path='configs', config_name='config')
def app(config):
    #print(OmegaConf.to_yaml(config))
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    #set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
    if 'gpuNumber' in config:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f'Setting CUDA_VISIBLE_DEVICES to {config["gpuNumber"]}')
        os.environ["CUDA_VISIBLE_DEVICES"] = config['gpuNumber']

    if 'Slurm' in HydraConfig.get().launcher._target_:
        # TF train saver doesn't support file name with '[' or ']'. So we'll use relative path here.
        config.outputDir = './'
    print(f'Output dir {config.outputDir}')
    os.makedirs(config.outputDir, exist_ok=True)

    if 'wandb' in config and config.wandb.enabled:
        run = wandb.init(**config.wandb.setup,
                         config=OmegaConf.to_container(config, resolve=True),
                         sync_tensorboard=True,
                         resume=True)

    #instantiate the train model
    # nsd = BrainToSen(args=config)
    nsd = PhonemeToSen(args=config)

    #train or infer
    if config['mode'] == 'train':
        cer = nsd.train()
        return cer
    elif config['mode'] == 'inference':
        nsd.inference()

if __name__ == "__main__":
    app()
