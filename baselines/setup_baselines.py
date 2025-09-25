#!/usr/bin/env python3
"""
Setup script for downloading and configuring baseline models.
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
import torch

BASELINE_URLS = {
    'monodepth2': {
        'weights': 'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip',
        'repo': 'https://github.com/nianticlabs/monodepth2.git'
    },
    'dorn': {
        'weights': 'http://www.cs.cornell.edu/~dchen/dorn_models/DORN_nyu.zip',
        'repo': 'https://github.com/huangkun1993/DORN.git'
    }
}

def download_file(url, filename):
    """Download a file showing progress"""
    print(f"Downloading {url} to {filename}")
    urllib.request.urlretrieve(url, filename)

def setup_monodepth2():
    """Setup MonoDepth2 baseline"""
    weights_dir = 'baselines/weights/monodepth2'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Clone repository if needed
    if not os.path.exists('baselines/monodepth2'):
        print("Cloning MonoDepth2 repository...")
        subprocess.run(['git', 'clone', BASELINE_URLS['monodepth2']['repo'], 'baselines/monodepth2'])
    
    # Download weights if needed
    zip_path = os.path.join(weights_dir, 'mono_1024x320.zip')
    if not os.path.exists(zip_path):
        download_file(BASELINE_URLS['monodepth2']['weights'], zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)

def setup_dorn():
    """Setup DORN baseline"""
    weights_dir = 'baselines/weights/dorn'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Clone repository if needed
    if not os.path.exists('baselines/dorn'):
        print("Cloning DORN repository...")
        subprocess.run(['git', 'clone', BASELINE_URLS['dorn']['repo'], 'baselines/dorn'])
    
    # Download weights if needed
    zip_path = os.path.join(weights_dir, 'DORN_nyu.zip')
    if not os.path.exists(zip_path):
        download_file(BASELINE_URLS['dorn']['weights'], zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)

def main():
    """Setup all baseline models"""
    print("Setting up baseline models...")
    
    # Create weights directory
    os.makedirs('baselines/weights', exist_ok=True)
    
    # Setup each baseline
    setup_monodepth2()
    setup_dorn()
    
    print("\nInstalling requirements...")
    subprocess.run(['pip', 'install', '-r', 'baselines/requirements.txt'])
    
    print("\nSetup complete! You can now run baseline comparisons.")

if __name__ == '__main__':
    main()