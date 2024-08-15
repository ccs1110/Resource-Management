from kubernetes import client, config
from util.deployments import deploy
import tool

import  numpy
# 加载配置
config.load_kube_config()
# 创建 API 实例
v1 = client.AppsV1Api()
api_client = client.ApiClient()
#test namespace
namespace = 'ccs'
# 获取当前的 Deployment 对象

def get_deploy():
    deploymentlist = v1.list_namespaced_deployment(namespace=namespace) #返回类型是v1deployment_list
    deploys=[]
    # 初始化
    for de in deploymentlist.items:
        name=de.metadata.name
        replicas = de.spec.replicas
        cpu=de.spec.template.spec.containers[0].resources.limits.get('cpu')
        mem=de.spec.template.spec.containers[0].resources.limits.get('memory')
        deploy_instance=deploy(name=name,replicas=replicas,cpu=tool.convert_resource_value(cpu),mem=tool.convert_resource_value(mem))  #这里有多个可选项
        deploys.append(deploy_instance)
    return deploys

def scaling():
    deploys = get_deploy()  # 加载
    de = deploys[0]
    print(de.name)

    body = {
        "spec": {
            "replicas": 2
        }
    }
    response = v1.patch_namespaced_deployment_scale(name=de.name, namespace=namespace, body=body)
    print(f"Deployment {de.name} scale updated. New replicas: {response.spec.replicas}")




if __name__ == "__main__":
    deploys=get_deploy()#加载
    de=deploys[0]
    print (de.name)
    body = {
        "spec": {
            "replicas": 2
        }
    }
    response = v1.patch_namespaced_deployment_scale(name=de.name, namespace=namespace, body=body)
    print(f"Deployment {de.name} scale updated. New replicas: {response.spec.replicas}")















# # 更新 replicas 数量
# deployment.spec.replicas = 5  # 例如，将副本数量设置为 5
#
# # 更新 Deployment
# response = v1.patch_namespaced_deployment(
#     name=name,
#     namespace=namespace,
#     body=deployment
# )
#
# print(f"Deployment {name} updated. Replicas: {response.spec.replicas}")