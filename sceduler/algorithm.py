import numpy as np
import math


# 测试代码部分
class Pod(object):

    def __init__(self,name,cr,mr,cl,ml,gr,gl):
        self.name = name
        self.cpuReqs = cr
        self.memReqs = mr
        self.cpuLmt = cl
        self.memLmt =  ml
        self.gpuReqs=gr
        self.gpuLmt=gl
    pass


class Node(object):

    def __init__(self,name,ca,ma,cf,mf,ga,gf):
        self.name = name
        self.cpuAlloc = ca
        self.memAlloc = ma
        self.cpuFree = cf
        self.memFree = mf
        self.maxPodNum = 0.0
        self.gpuAlloc =ga
        self.gpuFree=gf
    pass


node0 = Node('node0', 2.0, 2.0, 1.957555882, 1.4630508422851561,8.0,4.0)
node1 = Node('node1', 1.0, 1.0, 0.966974067, 0.6905937194824219,8.0,5.0)
node2 = Node('node2', 1.0, 1.5, 0.976784023, 1.1852989196777344,8.0,4.0)
node3 = Node('node3', 2.0, 1.0, 1.954301959, 0.6926765441894531,8.0,7.0)
node4 = Node('node4', 2.0, 1.5, 1.956475688, 1.1220016479492188,8.0,2.0)
node5 = Node('node5', 1.0, 2.0, 0.974918764, 1.1879463195800781,0.0,0.0)
nodelist = [node0, node1, node2, node3, node4, node5]
pod0 = Pod('pod0', 0.160, 0.156, 0.3, 0.3,1.0,2.0)
pod1 = Pod('pod1', 0.160, 0.156, 0.4, 0.4,0.0,0.0)
pod2 = Pod('pod2', 0.160, 0.156, 0.4, 0.4,1.0,1.0)
pod3 = Pod('pod3', 0.160, 0.156, 0.4, 0.4,1.0,2.0)
pod4 = Pod('pod4', 0.160, 0.156, 0.4, 0.4,1.0,1.0)
podqueue = [pod0, pod1, pod2, pod3, pod4]

# 测试代码截止


def fit_func(position, nodeList, podQueue):
    value = fr(position, nodeList, podQueue)
    return value


def fr(position, nodeList, podQueue):
    minCpu = float('Inf')
    minMem = float('Inf')
    minGpu =float('Inf')
    value = 0
    node_cpu_scheduled_list = []
    node_mem_scheduled_list = []
    node_gpu_scheduled_list = []

    for nodeNum in range(len(nodeList)):
        node_cpu_scheduled = nodeList[nodeNum].cpuFree
        node_mem_scheduled = nodeList[nodeNum].memFree
        node_gpu_scheduled = nodeList[nodeNum].gpuFree

        for i in range(len(podQueue)):
            if position[0][i] == nodeNum:  # 筛选此pod是否选择当前nodeNum所对应的node
                node_cpu_scheduled = node_cpu_scheduled - podQueue[i].cpuReqs  # 后cpu
                node_mem_scheduled = node_mem_scheduled - podQueue[i].memReqs  # 后mem
                node_gpu_scheduled = node_gpu_scheduled - podQueue[i].gpuReqs#后mem
                if node_cpu_scheduled < 0 or node_mem_scheduled < 0 or node_gpu_scheduled < 0:
                    return 100000000000000
        node_cpu_scheduled_list.append(node_cpu_scheduled)
        node_mem_scheduled_list.append(node_mem_scheduled)
        node_gpu_scheduled_list.append(node_gpu_scheduled)
        # 求取所有node中，cpu和mem最小值
        if node_cpu_scheduled < minCpu:
            minCpu = node_cpu_scheduled
        if node_mem_scheduled < minMem:
            minMem = node_mem_scheduled
        if node_gpu_scheduled < minGpu:
            minGpu = node_gpu_scheduled

    for nodeNum in range(len(nodeList)):
        value = node_cpu_scheduled_list[nodeNum] - minCpu + node_mem_scheduled_list[nodeNum] - minMem + value \
            +node_gpu_scheduled_list[nodeNum] - minGpu
    return value


class Particle(object):
    def __init__(self, x_max, v_max, p_dim, nodeList, podQueue):
        self.__position = np.random.randint(0, x_max, (1, p_dim), 'int')  # 粒子当前位置 原来是x_max+1 忘了为啥，感觉没必要
        self.__v = np.random.uniform(v_max, v_max, (1, p_dim))  # 粒子的速度
        self.__pbest = np.zeros((1, p_dim))  # 粒子最好的位置
        self.__nodeList = nodeList  # 节点信息
        self.__podQueue = podQueue  # pod待调度队列
        self.__fitnessValue = fit_func(self.__position, self.__nodeList, self.__podQueue)  # 粒子当前适应度函数值
        self.__pbestFitValue = fit_func(self.__pbest, self.__nodeList, self.__podQueue)

    def set_pos(self, value):
        self.__position = value

    def get_pos(self):
        return self.__position

    def set_best_pos(self, value):
        self.__pbest = value

    def get_best_pos(self):
        return self.__pbest

    def set_vel(self, value):
        self.__v = value

    def get_vel(self):
        return self.__v

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue

    def get_pbest_fitvalue(self):
        return self.__pbestFitValue


class PSO(object):
    def __init__(self, c1, c2, w, p_dim, p_num, iter_num, x_max, v_max, eps, T, alpha, nodeList, podQueue,
                 global_best_fit=float('Inf')):
        self.alpha = alpha  # 退火因子
        self.T = T  # 模拟退火初始温度
        self.c1 = c1
        self.c2 = c2
        self.w = w  # 惯性权重
        self.p_dim = p_dim  # 维度
        self.p_num = p_num  # 种群规模
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max  # 飞行范围(可调度node个数)
        self.v_max = v_max  # 速度上限
        self.eps = eps  # 精度（截止条件）
        self.global_best_fit = global_best_fit  # 全局最优fitness‘ value
        self.global_best_fit_list = []  # 每次迭代后历史最优适应值列表
        self.global_best_position = np.zeros((1, p_dim), dtype='int')  # 全局最优位置
        self.nodeList = nodeList  # 节点信息
        self.podQueue = podQueue  # pod待调度队列
        self.Particle_list = [Particle(self.x_max, self.v_max, self.p_dim, self.nodeList, self.podQueue) for i in
                              range(self.p_num)]

    def update_vel_pos(self, particle, T):
        # 新速度计算
        vel_value = self.w * particle.get_vel() + self.c1 * np.random.rand() * (
                    particle.get_best_pos() - particle.get_pos()) \
                    + self.c2 * np.random.rand() * (self.global_best_position - particle.get_pos())
        vel_value[vel_value > self.v_max] = self.v_max
        vel_value[vel_value < -self.v_max] = -self.v_max
        # 新位置计算
        pos_value = np.floor(particle.get_pos() + vel_value)  # floor，取整
        pos_value[pos_value > (self.x_max - 1)] = (self.x_max - 1)
        pos_value[pos_value < 0] = 0

        # 概率容忍阶段
        before_value = fit_func(particle.get_pos(), self.nodeList, self.podQueue)  # 更新前fit_value
        after_value = fit_func(pos_value, self.nodeList, self.podQueue)  # 更新后fit_value
        if after_value <= before_value:
            particle.set_vel(vel_value)  # vel更新位置
            particle.set_pos(pos_value)  # position更新位置
        elif math.exp((before_value - after_value) / T) > np.random.uniform(low=0, high=1):
            particle.set_vel(vel_value)  # vel更新位置
            particle.set_pos(pos_value)  # position更新位置
        pbFlag = 0  # 个体最优解更新标志
        gbFlag = 0  # 群体最优解更新标志
        value = fit_func(particle.get_pos(), self.nodeList, self.podQueue)
        if value < particle.get_fitness_value():
            pbFlag = 1
            particle.set_fitness_value(value)
            particle.set_best_pos(pos_value)
        if value < self.global_best_fit:
            gbFlag = 1
            self.global_best_fit = value
            self.global_best_position = pos_value
        return value, pbFlag, gbFlag

    def update(self):
        for i in range(self.iter_num):
            # 更新速度和位置
            SwTatol = 0
            for part in self.Particle_list:
                partFitValue, pbflag, gbflag = self.update_vel_pos(part, self.T)
                if gbflag:
                    Sw = 60
                elif pbflag:
                    Sw = 1
                else:
                    Sw = 0
                SwTatol = SwTatol + Sw
            # 退火
            if self.T > 0.000001:
                self.T = self.alpha * self.T
            else:
                self.T = 0.000001
           # print("温度是%f" % self.T)  # #
           # print(self.global_best_position)
            self.global_best_fit_list.append(self.global_best_fit)  # 每次迭代完把当前的最优适应度存到列表
           # print(self.global_best_fit_list[-1])  # #
            # 自适应权重w更新
            weight = SwTatol / self.p_num

            self.w = (0.3 + 0.3 * self.w - 0.10 * i / self.iter_num)
           # print("本轮w%f" % self.w)
            if self.global_best_fit < self.eps:
                break
        return self.global_best_fit_list, self.global_best_position


if __name__ == "__main__":
    node0 = Node('node0', 2.0, 2.0, 1.957555882, 1.4630508422851561,8.0,1.0)
    node1 = Node('node1', 1.0, 1.0, 0.966974067, 0.6905937194824219,8.0,5.0)
    node2 = Node('node2', 1.0, 1.5, 0.976784023, 1.1852989196777344,8.0,4.0)
    node3 = Node('node3', 2.0, 1.0, 1.954301959, 0.6926765441894531,8.0,7.0)
    node4 = Node('node4', 2.0, 1.5, 1.956475688, 1.1220016479492188,8.0,2.0)
    node5 = Node('node5', 1.0, 2.0, 0.974918764, 1.1879463195800781,0.0,0.0)
    nodelist = [node0, node1, node2, node3, node4, node5]
    pod0 = Pod('pod0', 0.160, 0.156, 0.3, 0.3,1.0,2.0)
    pod1 = Pod('pod1', 0.160, 0.156, 0.4, 0.4,0.0,0.0)
    pod2 = Pod('pod2', 0.160, 0.156, 0.4, 0.4,1.0,1.0)
    pod3 = Pod('pod3', 0.160, 0.156, 0.4, 0.4,1.0,2.0)
    pod4 = Pod('pod4', 0.160, 0.156, 0.4, 0.4,1.0,1.0)
    podqueue = [pod0, pod1, pod2, pod3, pod4]
    pod_num = len(podqueue)
    node_num = len(nodelist)
    print("podnum",pod_num)
    print("node_num",node_num)
    for node in nodelist:
        print(node.cpuFree, node.memFree,node.gpuFree)



    pso = PSO(2, 2, 0.5, pod_num, 60, 1200, node_num, node_num + 1, -10000000000000, 10000, 0.98, nodelist, podqueue)
    fit_var_list, best_pos = pso.update()
    print("最优位置:" + str(best_pos))
    print("最优解:" + str(fit_var_list[-1]))
    print("list" + str(fit_var_list))
    for node in nodelist:
        print(node.cpuFree, node.memFree,node.gpuFree)
    for i in range(len(podqueue)):
        nodelist[int(best_pos[0][i])].cpuFree -= podqueue[i].cpuReqs
        nodelist[int(best_pos[0][i])].memFree -= podqueue[i].memReqs
        nodelist[int(best_pos[0][i])].gpuFree -= podqueue[i].gpuReqs
    for node in nodelist:
        print("节点剩余资源")
        print(node.cpuFree, node.memFree,node.gpuFree)

