from kubernetes import client, config
from util.deployments import deploy
import tool

namespace="ccs"
config.load_kube_config()
# 创建 API 实例
v1 = client.AppsV1Api()
api_client = client.ApiClient()
Microservice_name = ["user-mention-service", "home-timeline-service", "text-service",
                     "unique-id-service", "media-service", "nginx-web-server", "url-shorten-service",
                     "post-storage-service", "user-service", "social-graph-service", "user-timeline-service",
                     "compose-post-service"]
def get_deploy(v1,namespace,Microservice_name):
    deploymentlist = v1.list_namespaced_deployment(namespace=namespace) #返回类型是v1deployment_list
    deploys=[]
    # 初始化
    for de in deploymentlist.items:
        name=de.metadata.name
        if (name not in Microservice_name):
            continue
        replicas = de.spec.replicas
        # cpu=de.spec.template.spec.containers[0].resources.limits.get('cpu')
        # mem=de.spec.template.spec.containers[0].resources.limits.get('memory')
        # deploy_instance=deploy(name=name,replicas=replicas,cpu=tool.convert_resource_value(cpu),mem=tool.convert_resource_value(mem))  #这里有多个可选项
        deploy_instance = deploy(name=name, replicas=replicas)
        deploys.append(deploy_instance)
    return deploys

def scaling(v1,namespace,deploys):
    try:
        for de in deploys:
            body = {
                "spec": {
                    "replicas": de.replicas
                }
            }
            response = v1.patch_namespaced_deployment_scale(name=de.name, namespace=namespace, body=body)
            print(f"Deployment {de.name} scale updated. New replicas: {response.spec.replicas}")
    except Exception as e:
        print(f"An error occurred while updating the deployment scale: {e}")




if __name__ == "__main__":
    deploys=get_deploy(v1,namespace,Microservice_name)#加载
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