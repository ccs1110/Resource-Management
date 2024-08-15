from kubernetes import client, config
from util.deployments import deploy
import requests
import time
from datetime import datetime, timedelta
import scaling
import sys
import subprocess
import jaeger
from util.MS_trace import MS_trace
import scaling
import math

# 伸缩方法是以基于jaeger的分布式追踪的端到端时延，并定义时延阈值来决定微服务的整体伸缩程度
# 在每轮伸缩中，pod总数量满足下等式:
#         pod_num_demand=math.ceil(target_lantency*pod_num_current/average_end_duration)
# 每个微服务能够获得pod数量是:
#         r_n=pod_num_demand*ms.weight
# 其中每个微服务的weight计算方式是
#         ms.weight=ms.duration/duration_total
#  ms.duration是所有追踪的中，微服务的时延之和，
#  duration_total是上述ms.duration的不同微服务之和，weight总和为1






config.load_kube_config()
v1 = client.AppsV1Api()
api_client = client.ApiClient()
#test namespace
namespace = 'ccs'
max_pod=6
min_pod=1
target_lantency=5000   #目标延迟
scaling_interval=60


def run_locust():
    locust_command = [
        'locust',
        '-f', 'locustfile.py',
        '--headless',
        '-t', '30m'
    ]

    try:
        # Run the locust command
        result = subprocess.run(locust_command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running locust: {e.stderr}")

def calibrate(pod_num):
    pod_num=max(pod_num,min_pod)
    pod_num=min(pod_num,max_pod)
    pod_num=math.ceil(pod_num)
    return pod_num


Microservice_name = ["user-mention-service", "home-timeline-service", "text-service",
                     "unique-id-service", "media-service", "nginx-web-server", "url-shorten-service",
                     "post-storage-service", "user-service", "social-graph-service", "user-timeline-service",
                     "compose-post-service"]
#只有无状态服务参与伸缩



Microservices_trace = {
}
Microservices_deploy = {
}
deploys = scaling.get_deploy(v1, namespace, Microservice_name)
for name in Microservice_name:
    Mt = MS_trace(name=name)
    Microservices_trace[name] = Mt
    for dp in deploys:
        if(dp.name==name):
            Microservices_deploy[name]= dp
#初始化两个字典，方便查询微服务对象
#获取


if __name__ == "__main__":
    while(True):
        # run_locust()
        # time.sleep(30) #等待locust运行后的30s
        ########################
        deploys = scaling.get_deploy(v1, namespace,Microservice_name)
        pod_num_current=sum([deploy.replicas for deploy in deploys])
        print(f"current_pod_num={pod_num_current}")

        Microservices_trace,average_end_duration=jaeger.get_weight(Microservices_trace)
        pod_num_demand=math.ceil(target_lantency*pod_num_current/average_end_duration)
        print(f"pod_num_demand={pod_num_demand}")

        for dp in deploys:
            r_n=pod_num_demand*Microservices_trace[dp.name].weight  #期望的副本数量
            r_n=calibrate(r_n)
            dp.replicas=r_n
            print(f"{dp.name} get new replicas ={r_n}")
        scaling.scaling(v1,namespace,deploys)  #执行伸缩操作，用k8s API进行伸缩

        for ms in Microservices_trace.values():
            ms.clear_duration()
        time.sleep(scaling_interval)










