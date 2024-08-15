import subprocess
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# 定义 Pushgateway 地址
pushgateway_address = 'localhost:9091'

# 创建一个新的 registry
registry = CollectorRegistry()

# 定义 Prometheus 指标
gpu_utilization = Gauge('jetson_gpu_utilization', 'GPU utilization in %', registry=registry)

def parse_tegrastats():
    result = subprocess.run(['tegrastats'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    gpu_util = 0
    for line in output.splitlines():
        if 'GR3D_FREQ' in line:
            parts = line.split()
            for part in parts:
                if 'GR3D_FREQ' in part:
                    gpu_util = int(part.split('@')[1].replace('%', ''))
            break
    return gpu_util

def collect_and_push_metrics():
    while True:
        gpu_util = parse_tegrastats()
        gpu_utilization.set(gpu_util)

        # 将指标推送到 Pushgateway
        push_to_gateway(pushgateway_address, job='jetson_tegrastats', registry=registry)

        time.sleep(5)

if __name__ == '__main__':
    collect_and_push_metrics()

