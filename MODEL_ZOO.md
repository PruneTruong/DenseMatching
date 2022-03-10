
# Dense Matching Model Zoo

| Model        | Pre-trained model type | Paper | Description | Link [all](https://drive.google.com/drive/folders/1LVrwAHDvVxsqzaGtd409wHv8cSPeLj1i?usp=sharing) |
|--------------|------------------------|-------------------------------------|------|------|
| PWarpCSFNet_WS | pfpascal / spair  | [6] | weakly-supervised | [pfpascal](https://drive.google.com/file/d/1wtY_Lt5WD5GDQ5f-y8IxYgFfwLvIsM3w/view?usp=sharing) / [spair](https://drive.google.com/file/d/1TK4CDi0X3nlkGRv6TVTfFtRLS9w3LQIP/view?usp=sharing) 
| PWarpCSFNet_SS | pfpascal / spair  | [6] | strongly-supervised |  [pfpascal](https://drive.google.com/file/d/1EYonyRBYl5RND2LSXYT24v2vaEaABOe0/view?usp=sharing) /  [spair](https://drive.google.com/file/d/1zLi1PztBu_QvHrHbObo3evSMTZnqctnw/view?usp=sharing)
| PWarpCNCNet_WS |  pfpascal / spair  | [6] | weakly-supervised | [pfpascal](https://drive.google.com/file/d/1xwS7qnjngO9mfeRUkwFJAdlz-GBLDxXz/view?usp=sharing) / [spair](https://drive.google.com/file/d/1lzMXLmeTROi2BEpIhdWA-wVXdoi88-Sg/view?usp=sharing)
|PWarpCNCNet_SS |pfpascal / spair  | [6] | strongly-supervised |  [pfpascal](https://drive.google.com/file/d/1hv8oS5L5ncO_z2_I3T7a558PqRCLo7SO/view?usp=sharing) / [spair](https://drive.google.com/file/d/1dQqBiVpT-ZiyMdy7JmMCZXBjXrBC661r/view?usp=sharing)
|PWarpCCATs_ft_features_SS |pfpascal / spair  | [6] | strongly-supervised | [pfpascal](https://drive.google.com/file/d/1VmUook2CCFHexRh1u5WkUBs-G0dloMBa/view?usp=sharing) / 
| ---- | -------- | -------| ---  | --- |
| PDCNet_plus      | megadepth  |  [3], [5] |   PDC-Net+       |  [model](https://drive.google.com/file/d/151X9ovbOG35tbPjioV5CYk_5GKQ8FErw/view?usp=sharing) 
| ---- | -------- | -------| ---  | --- |
| WarpCSemanticGLUNet       | spair  |  [4] |   Original SemanticGLU-Net is finetuned using our warp consistency objective      |  [model](https://drive.google.com/file/d/1aLZ8MoV_fHFScx__WWmqxZMm3k3MuEpr/view?usp=sharing)
| WarpCSemanticGLUNet       | pfpascal  |  [4] |    Original SemanticGLU-Net is finetuned using our warp consistency objective    |  [model](https://drive.google.com/file/d/1m_1dSa3cmUOmDWL4A1PBLEm6VW8O7x2x/view?usp=sharing)
| SemanticGLUNet      | pfpascal  |   [4] |   Original SemanticGLU-Net is finetuned using warp supervision                             |  [model](https://drive.google.com/file/d/1rhOXoYjO5QPnvcmHX45NCyevqCUx2YmH/view?usp=sharing)
| WarpCRANSACFlow       | megadepth   |        [4]                             | | [model](https://drive.google.com/file/d/1bKiwQ9tLIPi5KvQJHAT43d-zP5HCtOJW/view?usp=sharing)
| WarpCGLUNet      | megadepth  /    megadepth_stage1  |  [4] |                                     |  [megadepth](https://drive.google.com/file/d/1ztQL04eSxleXAIRmInFjHY3tqK6n_iyA/view?usp=sharing) / [megadepth_stage1](https://drive.google.com/file/d/1vnYpYoqBNWg1EcBSkQm65en_IdsEbbX2/view?usp=sharing)
| GLUNet_star       | megadepth /    megadepth_stage1  |  [4] |        Baseline for WarpCGLU-Net, trained with warp-supervision loss only   |  [megadepth](https://drive.google.com/file/d/1udUBzDkHoe6AggpZ8tRjYrljt3au0-rh/view?usp=sharing)  / [megadepth_stage1](https://drive.google.com/file/d/1PtLuTtO9kOCM_IO7WtW8xqbQDzQ9xixi/view?usp=sharing)
| ---- | -------- | -------| ---  | --- |
| PDCNet       | megadepth              |  [3] |                                   |  [model](https://drive.google.com/file/d/1nOpC0MFWNV8N6ue0csed4I2K_ffX64BL/view?usp=sharing)    |
| GLUNet_GOCor_star | megadepth              | [3] |corresponds to GLU-Net-GOCor* in [PDCNet](https://arxiv.org/abs/2101.01710) |    [model](https://drive.google.com/file/d/1bU6ZPMGsyzZJdAE5gmuxYjgxyVzwcLPj/view?usp=sharing)  |
| ---- | -------- | -------| ---  | --- |
| GLUNet_GOCor | dynamic                |  [2] |                                   | [model](https://drive.google.com/file/d/1j8lUIRf39wECSNMHJqnu42VoBi6mWd-v/view?usp=sharing)     |
| GLUNet_GOCor | static                 | [2] |                                    |  [model](https://drive.google.com/file/d/1f-XOVJlMUmmFsQojB7KuiBfX3nXJA_Er/view?usp=sharing)    |
| PWCNet_GOCor | chairs_things_ft_sintel   |            [2]          |                                     |  [model](https://drive.google.com/file/d/1oL07Fv5qz_H3EzZE2NmmZRR06x8fK3Jn/view?usp=sharing)    |
| PWCNet_GOCor | chairs_things          | [2] |                                    | [model](https://drive.google.com/file/d/1ofkmCZR7xyUgzreyL7B5QXSl5ZljkMLo/view?usp=sharing)     |
| GLUNet       | dynamic     |  [2] |                                   |   [model](https://drive.google.com/file/d/1SoCEg0IKfbkTu7aD5HnxIRKirjn3EJte/view?usp=sharing)   |
| ---- | -------- | -------| ---  | --- |
| GLUNet       | static (CityScape-DPED-ADE)     | [1] |                                    |  [model](https://drive.google.com/file/d/1cu_8lwhuqeNsIxEsuB6ihDBzz-yLW_L5/view?usp=sharing)    |
| SemanticGLUNet   | static (CityScape-DPED-ADE) |  [1] |     |  [model](https://drive.google.com/file/d/15cDS1tyySMn-SHBUIa-pS1VY8-zbp0hO/view?usp=sharing)


To download all of them, run the command ```bash assets/download_pre_trained_models.sh```. 

All networks are created in 'model_selection.py'. Weights should be put in pre_trained_models/


<br />
<br />

**Evaluation of WarpCRANSACFlow:**

The pre-trained weights can directly be used in the [RANSAC-Flow repo](https://github.com/XiSHEN0220/RANSAC-Flow). 