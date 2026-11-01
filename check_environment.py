from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def command_output(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return f"{command[0]} not found"
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30)
    except subprocess.CalledProcessError as error:
        return error.output


report: dict[str, object] = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
    "nvidia_smi": command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    ).strip(),
}

try:
    import torch
    import torchvision

    report.update(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    )
except Exception as error:
    report["pytorch_import"] = f"failed: {error!r}"

print(json.dumps(report, indent=2, ensure_ascii=False))
