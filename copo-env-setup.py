#!/usr/bin/python3

# Setup script for CoPO. Creates a conda environment and installs require prerequisite packages via conda.
# To generate the yml file for all packages required in a conda environment, when the environment is activated, run:
# "conda env export > environment.yml" and the file named "environment.yml" will be generated in the current directory.     
# (C) Ran Jing, Andrew Sabelhaus and Kiyn Chin, 2020-2024 

## pip install wheel==0.38.4

# REQUIRES:
#   - Python3 installed (so this file can run)
#   - anaconda / miniconda / one of its variants

import subprocess
import os

welcome = """\ncopo-auto: Setup script. Starting... Check conda installation...\n"""
print(welcome, flush=True)

# subprocess behavior with "shell" is different between linux and windows, so we have to check the platform.
is_windows = False
if os.name == 'nt':
    is_windows = True

# dependencies are slightly different if windows vs. linux/macOS, specifically with visual c++
env_spec_file = 'setup/env_copo_022924.yml'
if is_windows:
    env_spec_file = 'setup/env_copo_022924.yml'

# Try to set up conda and channels but stop if not found
try:
    subprocess.run(['conda', 'init', 'bash'], shell=is_windows, check=True)
    print('copo-auto: Conda found and is initialized.', flush=True)
    subprocess.run(['conda', 'config', '--add', 'channels', 'conda-forge'], shell=is_windows, check=True)
    print('copo-auto: Conda channels set up for conda-forge.', flush=True)
except subprocess.CalledProcessError:
    print('Conda not found. Please install before continuing with ezloophw-py: https://docs.conda.io/en/latest/miniconda.html')
    quit()

# create our conda environment with its dependencies. TO-DO: Force re-create each time this script is run.
try:
    print('\ncopo-auto: Attempting to create environment...', flush=True)
    subprocess.run(['conda', 'env', 'create', '-f', env_spec_file], shell=is_windows, check=True)
except subprocess.CalledProcessError:
    print('\ncopo-auto: Conda issue with environment creation. Possible fixes:\n')
    print('If the copo-auto environment already exists, you will need to run before trying again: conda env remove -n copo-auto')
    print('If creation was successful but you get an init error, you may need to close this terminal and re-open.\n', flush=True)
    quit()

try: 
    print('copo-auto: Conda found and environment created or already present. Now activating just to check...', flush=True)
    subprocess.run(['conda', 'activate', 'ezloophw-cv'], shell=is_windows, check=True)
except subprocess.CalledProcessError:
    print('\ncopo-auto: Conda issue with environment activation. If all else was successful, ignore this error, you are ready to go.\n')

cmesg = """\nSetup Complete. To run an example ezloophw-py script: 
        conda activate copo-auto
        python copo-auto/example.py\n"""
print(cmesg)

print("Final note! The dependencies in setup/env_copo_022924.yml may be unnecessary/outdated! Update as needed!\n")
