

import subprocess


class GPUMonitor:
    """Utility class for monitoring GPU memory usage."""
    
    @staticmethod
    def get_memory_usage() -> list[int]:
        """Get GPU memory usage for all GPUs in MB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                text=True,
                check=True
            )
            return [int(x) for x in result.stdout.strip().split('\n')]
        except (subprocess.SubprocessError, ValueError) as e:
            print(f"Warning: Could not get GPU memory usage: {e}")
            return [0]
