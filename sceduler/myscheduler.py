import kubernetes as k8s
import algorithm
import argparse
import re
import json
from kubernetes import client, watch
from kubernetes.client.rest import ApiException
from collections import defaultdict
from pint import UnitRegistry
import requests


edge_gpu_limit_thread=40
# k8s.config.load_kube_config()
k8s.config.load_incluster_config()
v1 = client.CoreV1Api()
api_client = client.ApiClient()


ureg = UnitRegistry()
ureg.load_definitions('kubernetes_units.txt')
Q_ = ureg.Quantity

parser = argparse.ArgumentParser(description='My scheduler name has been initialized')
parser.add_argument('--scheduler-name', type=str, default=None)  # 添加变量
args = parser.parse_args()
pods = v1.list_pod_for_all_namespaces()

node_ip_dic={
        "kctd":"10.112.184.214",
        "t2":"10.112.196.200",
        "t3":"10.112.188.4",
        "nvidia-6-28":"10.112.254.159",
}
node_type={
        "kctd":"master",
        "t2":"t2",
        "t3":"t3",
        "nvidia-6-28":"edge",
}


class Pod(object):

    def __init__(self):
        self.name = 'podName'
        self.cpuReqs = 0.0
        self.memReqs = 0.0
        self.cpuLmt = 0.0
        self.memLmt = 0.0
        self.gpuReqs=0.0
        self.gpuLmt=0.0
        self.appointedge=False
        self.needgpu=False
    pass


class Node(object):

    def __init__(self):
        self.ipadress=None
        self.name = 'nodeName'
        self.cpuAlloc = 0.0
        self.memAlloc = 0.0
        self.cpuFree = 0.0
        self.memFree = 0.0
        self.maxPodNum = 0.0
        self.gpuAlloc =0.0
        self.gpuFree=0.0
        self.nodetype=None
        self.gpu_util=0.0
    pass



def get_edge_cpu_gpu():
    metrics_url = "http://localhost:9098/metrics"
    response = requests.get(metrics_url)
    response.raise_for_status()
    metrics = {}
    for line in response.text.splitlines():
        if line.startswith('#') or line.strip() == '':
            continue
        key, value = line.split()[:2]
        metrics[key] = float(value)



def get_pod_resource(pod):

    PodReqs = {'cpu_reqs_sum': 0, 'mem_reqs_sum': 0,'gpu_reqs_sum':0}
    PodLmt = {'cpu_lmt_sum': 0, 'mem_lmt_sum': 0,'gpu_lmt_sum':0}
    for container in pod['spec']['containers']:
        res = container['resources']
        reqs = defaultdict(lambda: 0, res['requests'] or {})
        lmts = defaultdict(lambda: 0, res['limits'] or {})
        PodReqs['cpu_reqs_sum'] = Q_(reqs['cpu']) + PodReqs['cpu_reqs_sum']
        PodReqs['mem_reqs_sum'] = Q_(reqs['memory']) + PodReqs['mem_reqs_sum']
        PodReqs['gpu_reqs_sum']=  Q_(reqs['aliyun.com/gpu-mem'])+PodReqs['gpu_reqs_sum']

        PodLmt['cpu_lmt_sum'] =   Q_(lmts['cpu']) + PodLmt['cpu_lmt_sum']
        PodLmt['mem_lmt_sum'] =   Q_(lmts['memory']) + PodLmt['mem_lmt_sum']
        PodLmt['gpu_lmt_sum'] =   Q_(lmts['aliyun.com/gpu-mem']) + PodLmt['gpu_lmt_sum']
    return PodReqs, PodLmt


def get_node(pod_has_gpu,quene_pod_appoint_edge):  #这里传递一个pod_has_gpu的信号来筛选具有gpu的node ，值为true的时候只筛选有gpu的节点
                            #注：有gpu的节点应该手动打上containsGPU=true的指标
    # 存放所有节点对象
    nodeList = []
    # 获取所有节点
    nodeInstance = v1.list_node()
    # 存放可用节点的节点名称
    useNodeName = []
    # 存放API获取的资源信息
    NodeApilist = []
    for i in nodeInstance.items:
        # print(i.metadata.name)
        if (i.status.conditions[-1].status == "True" and
                i.status.conditions[-1].type == "Ready" and
                i.metadata.name not in ["master", "kctd"]):
            # print(i.metadata.name)
            if(not quene_pod_appoint_edge):
                if(pod_has_gpu and i.metadata.labels.get('containsGPU') == 'true' and i.metadata.labels.get('nodetype')!='edge'):
                    #这里筛查有gpu的节点
                        useNodeName.append(i.metadata.name)
                elif(pod_has_gpu==False and i.metadata.labels.get('nodetype')!='edge'):
                    useNodeName.append(i.metadata.name)
            else:
                if(i.metadata.labels.get('nodetype')=='edge'):
                    if (pod_has_gpu and i.metadata.labels.get('containsGPU') == 'true' and is_edge_gpu_u_suit(i)):
                        useNodeName.append(i.metadata.name)
                    elif (pod_has_gpu == False ):
                        useNodeName.append(i.metadata.name)

    useNodeName.sort(key=None, reverse=False)
    # 此处通过metrics API获得每个node已经使用的cpu和mem
    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/nodes', 'GET', auth_settings=['BearerToken'], response_type='json', _preload_content=False)
    response = json.loads(ret_metrics[0].data.decode('utf-8'))


    for item in response['items']:
        NodeApilist.append(item['metadata']['name'])
    # 创建node对象并对其元素初始化
    node_ip="."
    for i in range(len(useNodeName)):
        locals()['node_' + str(i)] = Node()
        locals()['node_' + str(i)].name = useNodeName[i]
        for addr in v1.read_node_status(locals()['node_' + str(i)].name).status.addresses:
            if addr.type == "InternalIP":
                node_ip = addr.address



        locals()['node_' + str(i)].ipadress = node_ip
        locals()['node_' + str(i)].cpuAlloc = Q_('0') + Q_(v1.read_node_status(locals()['node_' + str(i)].name).status.allocatable['cpu'])
        locals()['node_' + str(i)].maxPodNum = int(int(v1.read_node_status(locals()['node_' + str(i)].name).status.allocatable['pods']) * 1.5)
        locals()['node_' + str(i)].memAlloc = Q_('0 Gi') + Q_(v1.read_node_status(locals()['node_' + str(i)].name).status.allocatable['memory'])
        field_selector = ("status.phase!=Succeeded,status.phase!=Failed," + "spec.nodeName=" + locals()['node_' + str(i)].name)
        if(locals()['node_' + str(i)].name in NodeApilist):
            index = NodeApilist.index(locals()['node_' + str(i)].name)
            locals()['node_' + str(i)].cpuFree = Q_('0') + Q_(locals()['node_' + str(i)].cpuAlloc) - Q_(response['items'][index]['usage']['cpu'])
            locals()['node_' + str(i)].memFree = Q_('0 Gi') + Q_(locals()['node_' + str(i)].memAlloc) - Q_(response['items'][index]['usage']['memory'])
        else:
            cpu_util,mem_util=get_node_metrics_from_pushgateway()
            locals()['node_' + str(i)].cpuFree = Q_(locals()['node_' + str(i)].cpuAlloc) * (1 - cpu_util)  # 计算空闲CPU
            locals()['node_' + str(i)].memFree = Q_(locals()['node_' + str(i)].memAlloc) * (1 - mem_util)  # 计算空闲内存


        if(node_ip!="." and pod_has_gpu==True and node_type[locals()['node_' + str(i)].name]!='edge'):#校验IP是否为空，
            url="http://"+node_ip+":9400/metrics"
            gpufree,gpuse,gpu_util=fetch_gpu_memory_and_utilization_metrics(url)
            locals()['node_' + str(i)].gpuAlloc = gpufree+gpuse
            locals()['node_' + str(i)].gpuFree = gpufree
            locals()['node_' + str(i)].gpu_util=gpu_util


        else:
            print("ip null")
            locals()['node_' + str(i)].gpuAlloc = 0.0
            locals()['node_' + str(i)].gpuFree = 0.0
            locals()['node_' + str(i)].gpu_util = 0.0



        nodeList.append(locals()['node_' + str(i)])
    return nodeList


def get_schedule_queue():
    i = 0
    queue = []
    queuePodName = []
    try:
        pods = v1.list_pod_for_all_namespaces()
        print("Successfully retrieved all pods.")
    except ApiException as e:
        print(f"Exception when calling CoreV1Api->list_pod_for_all_namespaces: {e}")
        print(f"Status: {e.status}")
        print(f"Reason: {e.reason}")
        print(f"Body: {e.body}")
        return queue  # Return an empty queue in case of exception

    try:
        # 获取需要调度的pod，也就是处于pending状态的pod
        for pod in pods.items:
            if (pod.status.phase == 'Pending' and
                pod.spec.node_name is None and
                pod.spec.scheduler_name == args.scheduler_name):

                queuePodName.append(pod.metadata.name)
                locals()['schedulingPod' + str(i)] = Pod()
                locals()['schedulingPod' + str(i)].name = queuePodName[-1]
                PodReqs, PodLmt = get_pod_resource(pod.to_dict())
                locals()['schedulingPod' + str(i)].cpuReqs = Q_('0') + PodReqs['cpu_reqs_sum']
                locals()['schedulingPod' + str(i)].memReqs = Q_('0 Gi') + PodReqs['mem_reqs_sum']
                locals()['schedulingPod' + str(i)].cpuLmt = Q_('0') + PodLmt['cpu_lmt_sum']
                locals()['schedulingPod' + str(i)].memLmt = Q_('0 Gi') + PodLmt['mem_lmt_sum']
                locals()['schedulingPod' + str(i)].gpuReqs = Q_('0') + PodReqs['gpu_reqs_sum']
                locals()['schedulingPod' + str(i)].gpuLmt = Q_('0') + PodLmt['gpu_lmt_sum']
                if ('needgpu' in pod.metadata.labels and pod.metadata.labels['needgpu'] == 'true'):
                    locals()['schedulingPod' + str(i)].needgpu=True
                else:
                    locals()['schedulingPod' + str(i)].needgpu = False
                if ('appointedge' in pod.metadata.labels and pod.metadata.labels['appointedge'] == 'true'):
                    locals()['schedulingPod' + str(i)].appointedge=True
                    locals()['schedulingPod' + str(i)].gpuReqs = Q_('0') + 0
                    locals()['schedulingPod' + str(i)].gpuLmt = Q_('0') + 0
                queue.append(locals()['schedulingPod' + str(i)])
                i += 1
    except Exception as e:
        print(f"Exception while processing pods: {e}")

    return queue



# Binding
def binding(NodeName, PodName, namespace='default'):
    target = client.V1ObjectReference(api_version='v1', kind="Node", name=NodeName)
    meta = client.V1ObjectMeta()
    meta.name = PodName
    body = client.V1Binding(metadata=meta, target=target)
    return v1.create_namespaced_binding(namespace, body, _preload_content=False)



def fetch_gpu_memory_and_utilization_metrics(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            raw_data = response.text
            print("200 to get data")
            fb_free = None
            fb_used = None
            gpu_util = None

            fb_free_match = re.search(r'DCGM_FI_DEV_FB_FREE{[^}]+} (\d+)', raw_data)
            if fb_free_match:
                fb_free = int(fb_free_match.group(1))

            fb_used_match = re.search(r'DCGM_FI_DEV_FB_USED{[^}]+} (\d+)', raw_data)
            if fb_used_match:
                fb_used = int(fb_used_match.group(1))

            gpu_util_match = re.search(r'DCGM_FI_DEV_GPU_UTIL{[^}]+} (\d+)', raw_data)
            if gpu_util_match:
                gpu_util = int(gpu_util_match.group(1))

            if fb_free is not None and fb_used is not None and gpu_util is not None:
                return fb_free, fb_used, gpu_util
            else:
                print("Failed to find required metrics")
        else:
            print(f"Request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching metrics: {str(e)}")

def fetch_gpu_util_from_pushgateway(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            raw_data = response.text
            print("200 to get data")
            gpu_util = None

            gpu_util_match = re.search(r'jetson_gpu_utilization{[^}]+} (\d+)', raw_data)
            if gpu_util_match:
                gpu_util = int(gpu_util_match.group(1))

            if  gpu_util is not None:
                return  gpu_util
            else:
                print("Failed to find required metrics")
        else:
            print(f"Request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching metrics: {str(e)}")





def is_edge_gpu_u_suit(node):#筛选edge节点gpu使用率低于阈值
    nodename=node.metadata.name
    nodeip=node_ip_dic[node.metadata.name]
    now_gpu = 0;
    if(node_type[nodename]!='edge'):
        return True
    else:
        url="http://10.112.184.214"+':9098/metrics'
        now_gpu = fetch_gpu_util_from_pushgateway(url);
    if now_gpu>edge_gpu_limit_thread:
        return False
    else :
        return True
def get_node_metrics_from_pushgateway():
    response = requests.get("http://10.112.184.214:9098/metrics")
    if response.status_code != 200:
        raise RuntimeError("Failed to get metrics from node-exporter")

    metrics = response.text
    total_idle_cpu = 0.0
    total_user_cpu = 0.0
    total_system_cpu = 0.0
    cpu_count = 0
    mem_total, mem_available = None, None

    for line in metrics.splitlines():
        if(line.startswith('#')):
            continue
        if 'node_cpu_seconds_total_cpu' in line and '_mode__idle_' in line:
            total_idle_cpu += float(line.split()[-1])
            cpu_count += 1
        elif 'node_cpu_seconds_total_cpu' in line and '_mode__user_' in line:
            total_user_cpu += float(line.split()[-1])
        elif 'node_cpu_seconds_total_cpu' in line and '_mode__system_' in line:
            total_system_cpu += float(line.split()[-1])
        elif line.startswith("node_memory_MemTotal_bytes"):
            mem_total = float(line.split()[-1])
        elif line.startswith("node_memory_MemAvailable_bytes"):
            mem_available = float(line.split()[-1])
    if cpu_count > 0:
        total_cpu_time = total_idle_cpu + total_user_cpu + total_system_cpu
        cpu_usage = (total_user_cpu + total_system_cpu) / total_cpu_time
    else:
        cpu_usage = None

    if mem_total is not None and mem_available is not None:
        mem_usage = 1 - (mem_available / mem_total)
    else:
        mem_usage = None

    return cpu_usage, mem_usage


def main():
    w = watch.Watch()
    for event in w.stream(v1.list_pod_for_all_namespaces):
        try:
            quene_pod_has_gpu = False
            quene_pod_appoint_edge = False
            Queue = get_schedule_queue()
            # print("Queue:", Queue)
            if len(Queue):
                for pod in Queue:
                    try:
                        pod.cpuReqs = float(re.split(' ', str(pod.cpuReqs))[0])
                        pod.memReqs = float(re.split(' ', str(pod.memReqs))[0])
                        pod.cpuLmt = float(re.split(' ', str(pod.cpuLmt))[0])
                        pod.memLmt = float(re.split(' ', str(pod.memLmt))[0])
                        pod.gpuReqs =float(re.split(' ', str(pod.gpuReqs))[0])
                        pod.gpuLmt =float(re.split(' ', str(pod.gpuLmt))[0])
                        if(pod.gpuReqs!=0.0 or pod.gpuLmt!=0.0 or pod.needgpu==True):
                            quene_pod_has_gpu=True
                            print(f"pod_need_gpu:{quene_pod_has_gpu}")
                        if(pod.appointedge==True):
                            quene_pod_appoint_edge=True
                            print(f"apointedge:{quene_pod_appoint_edge}")
                    except Exception as e:
                        print(f"Error parsing pod resources: {e}")
                NodeList = get_node(quene_pod_has_gpu,quene_pod_appoint_edge)   #筛选节点
                print("NodeList:", NodeList)
                for node in NodeList:
                    try:
                        node.cpuAlloc = float(re.split(' ', str(node.cpuAlloc))[0])
                        node.memAlloc = float(re.split(' ', str(node.memAlloc))[0])
                        node.cpuFree = float(re.split(' ', str(node.cpuFree))[0])
                        node.memFree = float(re.split(' ', str(node.memFree))[0])
                        node.gpuAlloc = float(node.gpuAlloc)/1000
                        node.gpuFree = float(node.gpuFree)/1000
                    except Exception as e:
                        print(f"Error parsing node resources: {e}")

                pod_num = len(Queue)
                node_num = len(NodeList)
                pso = algorithm.PSO(2, 2, 0.5, pod_num, 60, 120, node_num, node_num + 1, -10000000000000, 10000, 0.99,
                                    NodeList, Queue)
                try:
                    global_best_fit_list, best_pos = pso.update()
                    print("global_best_fit_list:", global_best_fit_list)
                    print("best_pos:", best_pos)
                except Exception as e:
                    print(f"Error running PSO algorithm: {e}")

                try:
                    best_pos = best_pos.astype(int)
                    for i in range(len(Queue)):
                        binding(NodeList[best_pos[0][i]].name, Queue[i].name)
                        print(f"Pod {Queue[i].name} bound to Node {NodeList[best_pos[0][i]].name}")


                    print("Scheduling has completed!")
                except Exception as e:
                    print(f"Error during binding: {e}")

        except ApiException as e:
            print("Error while scheduling:", e)


if __name__ == '__main__':
    main()


