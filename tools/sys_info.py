import platform
import sys
import subprocess
import os
import tempfile
import importlib.metadata
import re

class Logger(object):
    def __init__(self, filename="cuda_env_report.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def print_header(title):
    print(f"\n{'='*20} {title} {'='*20}")

def run_command(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None

def get_linux_host_info():
    cpu_model = "Unknown CPU"
    ram_gb = 0.0
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":")[1].strip()
                    break
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    # MemTotal: 65809312 kB
                    kb = int(re.sub("[^0-9]", "", line))
                    ram_gb = kb / (1024.0 * 1024.0)
                    break
    except Exception:
        pass
    return cpu_model, ram_gb

def get_cuda_info_via_c():
    """Compiles and runs a small CUDA program to get detailed device properties."""
    cuda_code = r'''
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found or driver error.\n");
        return 1;
    }

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("--- [Hardware Identification] ---\n");
    printf("Device Name: %s\n", props.name);
    printf("Compute Capability: %d.%d (sm_%d%d)\n", props.major, props.minor, props.major, props.minor);
    printf("Number of SMs: %d\n", props.multiProcessorCount);
    printf("GPU Clock Rate: %.2f GHz\n", props.clockRate / 1e6f);

    printf("\n--- [Memory & Bandwidth] ---\n");
    printf("Total Global Memory: %.2f GB\n", props.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("L2 Cache Size: %.2f MB\n", props.l2CacheSize / (1024.0f * 1024.0f));
    printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
    printf("Memory Clock Rate: %.2f GHz\n", props.memoryClockRate / 1e6f);
    
    // Peak memory bandwidth calculation (2 * memoryClockRate * (busWidth / 8))
    float mem_bw = 2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6f;
    printf("Peak Memory Bandwidth: %.2f GB/s\n", mem_bw);
    printf("ECC Enabled: %s\n", props.ECCEnabled ? "Yes" : "No");

    printf("\n--- [Execution Limits] ---\n");
    printf("Warp Size: %d\n", props.warpSize);
    printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("Max Threads per SM:    %d\n", props.maxThreadsPerMultiProcessor);
    
    printf("\n--- [Shared Memory & Registers] ---\n");
    printf("Max Shared Memory per Block (Default): %.2f KB\n", props.sharedMemPerBlock / 1024.0f);
    printf("Max Shared Memory per Block (Opt-in):  %.2f KB\n", props.sharedMemPerBlockOptin / 1024.0f);
    printf("Max Shared Memory per SM:              %.2f KB\n", props.sharedMemPerMultiprocessor / 1024.0f);
    printf("Max Registers per Block: %d\n", props.regsPerBlock);
    printf("Max Registers per SM:    %d\n", props.regsPerMultiprocessor);

    printf("\n--- [Grid/Block Dimensions] ---\n");
    printf("Max Block Dim: [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("Max Grid Dim:  [%d, %d, %d]\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    
    return 0;
}
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        cu_file = os.path.join(tmpdir, "dev_info.cu")
        out_file = os.path.join(tmpdir, "dev_info")
        with open(cu_file, "w") as f:
            f.write(cuda_code)
        
        compile_cmd = ["nvcc", cu_file, "-o", out_file]
        try:
            subprocess.check_call(compile_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result = subprocess.check_output([out_file], text=True)
            return result
        except Exception:
            return "Failed to compile or run CUDA info snippet. Ensure nvcc is in PATH and drivers are loaded."

def main():

    sys.stdout = Logger("cuda_env_report.txt")
    print_header("System & Host Environment")
    print(f"OS             : {platform.system()} {platform.release()}")
    
    if platform.system() == "Linux":
        cpu, ram = get_linux_host_info()
        print(f"Host CPU       : {cpu}")
        print(f"System RAM     : {ram:.2f} GB")
        
    print(f"Python Version : {sys.version.split()[0]}")

    print_header("C++/CUDA Build Toolchain")
    tools = {
        "GCC": ["gcc", "--version"],
        "NVCC": ["nvcc", "--version"],
        "CMake": ["cmake", "--version"],
        "Ninja": ["ninja", "--version"]
    }
    
    for name, cmd in tools.items():
        out = run_command(cmd)
        if out:
            if name == "NVCC":
                version_line = out.splitlines()[-1]
            else:
                version_line = out.splitlines()[0]
            print(f"{name.ljust(15)} : {version_line}")
        else:
            print(f"{name.ljust(15)} : NOT FOUND")

    print_header("Python Packages (AI Infra)")
    packages = [
        "torch", "torchvision", "triton", "flash-attn", 
        "vllm", "xformers", "ninja", "pybind11"
    ]
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            print(f"{pkg.ljust(15)} : {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg.ljust(15)} : NOT INSTALLED")

    print_header("PyTorch CUDA Environment")
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        print(f"CUDA Available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA Ver  : {torch.version.cuda}")
            print(f"cuDNN Version     : {torch.backends.cudnn.version()}")
            print(f"Current Device    : {torch.cuda.get_device_name()}")
            cap = torch.cuda.get_device_capability()
            print(f"Device Capability : {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")
    except ImportError:
        print("PyTorch not installed.")

    print_header("Detailed GPU Hardware Info (via Native CUDA API)")
    print(get_cuda_info_via_c())

if __name__ == "__main__":
    main()