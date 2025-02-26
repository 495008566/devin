import os
import sys
import torch
import platform

def collect_env():
    """Collect environment information."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')
    env_info['CUDA available'] = torch.cuda.is_available()
    env_info['GPU devices'] = torch.cuda.device_count()
    if torch.cuda.is_available():
        env_info['CUDA_HOME'] = os.environ.get('CUDA_HOME', 'None')
        env_info['NVCC'] = 'None'
        env_info['GPU 0'] = torch.cuda.get_device_name(0)
    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()
    env_info['OS'] = platform.system()
    env_info['OS release'] = platform.release()
    env_info['CPU'] = platform.processor()
    return env_info
