#%%
from notebook_helpers import run, load_model_from_config
from main import instantiate_from_config
from omegaconf import OmegaConf
import torch
import numpy as np
import IPython.display as d
from PIL import Image
#%%

steps = 100 #@param
config = OmegaConf.load('models/ldm/bsr_sr/config.yaml')
model, step = load_model_from_config(config, "models/ldm/bsr_sr/model.ckpt")

input_path = "data/super_resolution_sample/1.png" #@param

logs = run(model["model"], input_path, "superresolution", steps)

sample = logs["sample"]
sample = sample.detach().cpu()
sample = torch.clamp(sample, -1., 1.)
sample = (sample + 1.) / 2. * 255
sample = sample.numpy().astype(np.uint8)
sample = np.transpose(sample, (0, 2, 3, 1))
print(sample.shape)
a = Image.fromarray(sample[0])
display(a)
     
# %%

print(a.shape)

# %%

a.save('ship1.png',"JPEG")

# %%
