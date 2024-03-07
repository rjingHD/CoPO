# Code for Coordinated Policy Optimization

[**Webpage**](https://decisionforce.github.io/CoPO) | [**Code**](https://github.com/decisionforce/CoPO) |  [**Paper**](https://arxiv.org/pdf/2110.13827.pdf) | [**Talk (English)**](https://youtu.be/sOw43l8lwxE) | [**Talk (Chinese)**](https://www.bilibili.com/video/BV1gr4y1C7Ab)
| [**Results&Models**](benchmarks/MetaDrive-0.2.5)



```diff
Changelog:
+ Jan 29, 2024: Fix dependencies issue.
+ Feb 19, 2023: 🎉 Upload torch implementation of CoPO, compatible with ray=2.2.0.
+ Oct 22, 2022: Update latest experiments results, curves and models!
+ June 22, 2022: Update README to include FAQ, update evaluate population script
+ June 23, 2022: Update a demo script to draw population evaluation results (See FAQ section)
+ July 7, 2022: Remove redundant files and use `train_all_*` scripts
```



Hi there! This is the source code of the paper “Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization”. 


Please following the tutorial below to kickoff the reproduction of our results.


## 🎉 Results, curves and models

Please refer to this link for latest training and evaluation results, learning curves, scripts to draw figures and models:

* [benchmarks/MetaDrive-0.2.5](benchmarks/MetaDrive-0.2.5)

## My Installation - Ran
This requires you to download and install ANACONDA (https://www.anaconda.com/download) on your machine.

**1. Create the virtual environment**
```bash
# Install CoPO repo
git clone git@github.com:rjingHD/CoPO.git
# make sure you are in the conda base environment before the command below - this will quick the virtual environment named "copo-auto"
cd CoPO
python copo-env-setup.py
```
If the installation fails, check step 2 and 3 for potential solutions.

**2. Loose the package requirement (can be skipped since I already changed the file, just FYI.)**

Since the ```ray[rllib]``` somehow requires gym version 0.23.1 that conflicts with the metadrive 0.2.5 required gym version of 0.19.0. Let pip attempt to solve the dependency conflict by commenting out the version requirement in ray (rllib). To do so, open the file ```~/CoPO/copo_code/setup.py``` comment out the two lines with "ray" such that the code looks like this: (not sure if this causes any issues yet.)
```bash
setup(
    name="copo",
    install_requires=[
        "yapf==0.30.0",
        # "ray==2.2.0",
        # "ray[rllib]==2.2.0",
        "tensorflow==2.3.1",
        "torch",
        "tensorflow-probability==0.11.1",
        "tensorboardX",
        "gym==0.19.0"
    ],
    license="Apache 2.0",
)
```

**3. maybe optional? - correct opencv & wheel issue.**

If during the installation, you face an error like this
```
wheel.vendored.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
opencv-python>=3.
~~~^
[end of output]

note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed building wheel for gym
```
A ugly solution towards this can be:
```bash
cd /home/$YOUR_USER_NAME$/anaconda3/envs/copo-auto/lib/python3.7/site-packages/wheel/vendored/packaging/
gedit requirements.py
```
In `requirements.py`,
change line 33-37 from:
```
    def __init__(self, requirement_string: str) -> None:
        try:
            parsed = _parse_requirement(requirement_string)
        except ParserSyntaxError as e:
            raise InvalidRequirement(str(e)) from e
```
to:
```
    def __init__(self, requirement_string: str) -> None:
        try:
            if requirement_string.find('opencv-python>=3.')>=0:
                requirement_string += "0"    # opencv-python>=3.0
            parsed = _parse_requirement(requirement_string)
        except ParserSyntaxError as e:
            raise InvalidRequirement(str(e)) from e
```
This solution comes from: (https://github.com/openai/gym/issues/3202)

**4. install CoPO in the virtual environment (copo-auto)**
```bash
# activate the conda virtual env 
conda activate copo-auto
# install CoPO now
cd ~/CoPO  # Go to repo root.
cd copo_code
pip install -e .
# now you can test by executing the command in training/visualization/evaluation sections in the README. 
# for example
cd ~/CoPO  # Go to repo root.
python ./copo_code/copo/torch_copo/train_ccppo.py
```

## Official Installation - not used for now - buggy

```bash
# Create virtual environment
conda create -n copo python=3.7
conda activate copo

# Install MetaDrive version 0.2.5
pip install git+https://github.com/metadriverse/metadrive.git@releases/0.2.5

# Install dependency
pip install torch  # Make sure your torch is successfully installed! Especially when using GPU!

# Install CoPO repo
git clone https://github.com/decisionforce/CoPO
cd CoPO/copo_code
pip install -e .

# For **running torch implementation**, install torch and update ray:
# (If you are using TF implementation, you might need to follow ray==1.2.0.
pip install -U ray==2.2.0 "ray[rllib]==2.2.0"
pip install -U "numpy<1.24.0"
pip uninstall opencv-python
pip uninstall opencv-python-headless
pip install opencv-python==4.5.5.64
pip install pydantic==1.9.0
```

Please install latest [MetaDrive](https://github.com/decisionforce/metadrive).
Note that we do not support `gym>=0.20.0`. In `setup.py` we specify we are using `gym==0.19.0`.


## Training

```diff
+ 🎉 We update torch version of our algorithms compatible with ray=2.2.0!!
```

Please take a look on the scripts at:

```bash
cd ~/CoPO  # Go to repo root.
python ./copo_code/copo/torch_copo/train_ccppo.py
python ./copo_code/copo/torch_copo/train_ippo.py
python ./copo_code/copo/torch_copo/train_copo.py
```

Note that you can even kickoff training at Macbook with this code!

---

(The below is old guideline for running TF version)

As a quick start, you can start training CoPO in Intersection environment immediately after installation by running:

```
cd copo_code/copo/
python train_all_copo_dist.py --exp-name copo 
```

Please visit each training script to adjust the hyper-parameters. 
The general way to run training is following:

```
cd copo_code/copo/
python train_all_ALGO.py --exp-name EXPNAME
```

Here `EXPNAME` is arbitrary name to represent the experiment. One experiment contains multiple concurrent trials with different random seeds or sweeps of hyper-parameter. By default the exp name is `TEST`.
`ALGO` is the shorthand for algorithms:

```
ippo  # Individual Policy Optimization
ccppo  # Mean Field Policy Optimization
cl  # Curriculum Learning
copo_dist  # Coordinated Policy Optimiztion (Ours)
```

You can also specify to use GPU via `python train_all_ALGO.py --exp-name EXPNAME --num-gpus 4`.
By default, we will run 8 trails with different seeds for one environment and one algorithm.
If this overwhelms your computing resource, please take a look on the training scripts and modify it by yourself.
If you want to verify the code in details, e.g. through setting breakpoints, you can to learn how to run CoPO in the local mode in [FAQ section](#faq). 

## Visualization

We provide the trained models for all algorithms in all environments. A simple command can bring you the visualization of the behaviors of the populations!

```
cd copo_code/copo

# Download and unzip this file:
wget https://github.com/metadriverse/metadrive-benchmark/releases/download/asset-marl/new_best_checkpoints.zip
unzip new_best_checkpoints.zip

python new_vis.py 

# In default, we provide you the CoPO population in Intersection environment. 
# If you want to see others, try:
python new_vis.py --env round --algo ippo

# Or you can use the native renderer for 3D rendering:
#  Press H to show helper message
#  Press Q to switch to third-person view
python new_vis.py --env tollgate --algo cl --use_native_render
```

We hope you enjoy the interesting behaviors learned in this work! 
Please feel free to contact us if you have any questions, thanks! 

There are two legacy scripts for visualization [vis_from_checkpoint.py](copo_code/copo/vis_from_checkpoint.py) and [vis.py](copo_code/copo/vis.py).
However, the performance of the agents varies largely due to the changes in MetaDrive environment.
The [new_vis.py](copo_code/copo/new_vis.py) instead runs the trained models from latest benchmark:
https://github.com/metadriverse/metadrive-benchmark/tree/main/MARL


## Evaluation

Evaluation is important to collect the test time performance of your agents. 
We will use evaluation results to draw the Radar figure with three metrics: safety, completeness and efficiency.


You can easily evaluate your trained agents via our provided script: `copo_code/copo/eval.py`. 
Suppose you use train script `python train_all_ippo.py --exp-name my_ippo`, then you can run the evaluation via:

```bash
cd copo_code/copo/

# Training
python train_all_ippo.py --exp-name my_ippo

# Evaluating
python eval.py --root my_ippo
``` 

The evaluation results will be saved to `copo_code/copo/evaluate_results`. 
Now, please refer to this link for script to draw figures:  https://github.com/metadriverse/metadrive-benchmark/tree/main/MARL


## FAQ

### How to run CoPO in the local mode?

If you want to dive into the code and try to understand what is happening, you can try to set local mode of Ray to True, 
in which case all code will run in a single process so that you can easily set breakpoints to step the code.

However, this will raise problem in native CoPO scripts, since MetaDrive has a strict 
requirement of singleton. Each process can only host one MetaDrive instance, imposed by the simulation engine.

To solve this issue, we need to make several light modifications to the training scripts:
Here is the procedure to setup local mode:

1. Set `config["num_workers] = 1`, indicating that you ask RLLib to only setup one process.
2. Remove `config["evaluation_config]` if any. In CoPO we don't test agents in different environments apart from the training environments, so we don't need to do this step.
3. Remove all `tune.grid_search([...])` code by setting each config with only one value.
4. Set `train(..., local_mode=True)`.


Here is the exemplar code for training IPPO in roundabout environment, provided natively in CoPO codebase:

```python
...
config = dict(
    env=get_rllib_compatible_env(MultiAgentRoundaboutEnv),
    env_config=dict(start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]), ),
    num_gpus=0.25 if args.num_gpus != 0 else 0,
)
train(
    IPPOTrainer,
    exp_name=exp_name,
    keep_checkpoints_num=5,
    stop=stop,
    config=get_ippo_config(config),
    num_gpus=args.num_gpus,
    num_seeds=1,
    test_mode=args.test,
    custom_callback=MultiAgentDrivingCallbacks,
    # local_mode=True
)
```

After the aforementioned modifications, the code becomes:

```python
config = dict(
    env=get_rllib_compatible_env(MultiAgentRoundaboutEnv),
    env_config=dict(start_seed=5000),  # <<<=== Modifified!
    num_gpus=0.25 if args.num_gpus != 0 else 0,
    num_workers=1,  # <<<=== Modifified!
)
train(
    IPPOTrainer,
    exp_name=exp_name,
    keep_checkpoints_num=5,
    stop=stop,
    config=get_ippo_config(config),
    num_gpus=args.num_gpus,
    num_seeds=1,
    test_mode=args.test,
    custom_callback=MultiAgentDrivingCallbacks,
    local_mode=True  # <<<=== Modifified!
)
```

Now you can run the training script with debugger! Please make sure to reset those changes if you want to deploy the script in production. Thanks!

### Can I use GPU for training?

Yes. Apart from specifying `python train_all_ALGO.py --num-gpus 4` to tell RLLib "I have 4 gpus in this computer!",
you can also modify the `num_gpus` config WITHIN the config dict.
The `num_gpus` within config dict specifies the number of GPU each trial will consume.
By default, `config["num_gpus"]=0.5` means each trial will use 0.5 GPU. If your computer has 4 gpus and sufficient cpus, then RLLib will
launch 8 concurrent trials. Note that those specifications does not mean true resource consumption.


### Opencv-python error

```bash
AttributeError: module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
```

Try:

```bash
pip uninstall opencv-python
pip uninstall opencv-python-headless
pip install opencv-python==4.5.5.64
```


### Ray dashboard error

```bash
TypeError: __init_subclass__() takes no keyword arguments
```

Try:

```bash
pydantic==1.9.0
```




## Citation

```latex
@article{peng2021learning,
  title={Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization},
  author={Peng, Zhenghao and Hui, Ka Ming and Liu, Chunxiao and Zhou, Bolei},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

