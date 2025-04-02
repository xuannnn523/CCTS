import networkx as nx
import pickle
import os
import numpy as np
import random
import time
import math
method=['Louvain','GN','LPA','FN']
num = 1
# names = [ 'karate','football', 'personal', 'polblogs', 'polbooks', 'railways','web-spam','road-minnesota','cit-DBLP']
#            0          1          2           3           4           5         6         7              8
names = [ 'karate','football', 'personal', 'polbooks' ]

index = 0

tag=[False,True]
i=0
# 是否将结果写入文件
if_write = True
# if_write = False
#是否使用社区中全部点
# if_part = True
if_part = tag[i]
# 是否限制阈值范围
# if_limit_threshold = True
if_limit_threshold = tag[i]
# 是否使用阈值剪枝
# if_threshold_pruning = True
if_threshold_pruning = tag[i]
# 是否使用BFS
# if_BFS = True
if_BFS = tag[i]
# 固定随机种子
random.seed(42)
np.random.seed(42)




class CommunityAnalyzer:
    def __init__(self, G, community_nodes, alpha=1, beta=1, seta=1, if_part=True, if_limit_threshold=True, if_threshold_pruning=True, if_BFS=True):
        self.G = G
        self.community_nodes = community_nodes
        self.alpha = alpha
        self.beta = beta
        self.seta = seta
        self.if_part = if_part
        self.if_limit_threshold = if_limit_threshold
        self.if_threshold_pruning = if_threshold_pruning
        self.if_BFS = if_BFS

    def calculate_precision_and_error_rate(self, community_nodes, threshold_nodes):
        """计算精确度和错误率"""
        correct_count = sum(1 for node in community_nodes if node in threshold_nodes)
        precision = correct_count / len(community_nodes) if community_nodes else 0
        # incorrect_nodes = [node for node in threshold_nodes if node not in community_nodes]
        # incorrect_nodes =
        # error_rate = len(incorrect_nodes) / len(threshold_nodes) if threshold_nodes else 0
        error_rate = (len(threshold_nodes)-correct_count) / len(threshold_nodes) if threshold_nodes else 0
        # return precision, error_rate, incorrect_nodes
        return precision, error_rate


    def objective_function(self, precision, error_rate):
        """目标函数：优选较小的阈值"""
        return self.alpha * precision - self.beta * error_rate

    def find_best_center_and_threshold(self):
        best_center_node = None
        best_threshold = None
        final_threshold_nodes = None
        best_score = -float('inf')
        center_threshold = -1
        # 获取枚举节点和划分比例
        top_nodes, proportion = self.get_top_nodes(len(self.community_nodes))
        for candidate_node in top_nodes:
            shortest_path_lengths = nx.single_source_shortest_path_length(G, candidate_node)
            max_distance = max(shortest_path_lengths[node] for node in community_nodes if node in shortest_path_lengths)
            left, right = self.determine_threshold_range(max_distance, center_threshold, len(self.community_nodes))
            temp_best_score = 0
            if self.if_BFS:
                threshold_nodes = set([candidate_node])
                # 遍历原距离字典，按照距离分层存储节点
                distance_layered_dict = {}
                flag = True
                for target_node, distance in shortest_path_lengths.items():
                    if distance not in distance_layered_dict:
                        distance_layered_dict[distance] = []
                    distance_layered_dict[distance].append(target_node)

            for threshold in range(left, right):
                # 记忆化BFS，动态规划
                if self.if_BFS:
                    if flag:
                        new_nodes = []
                        for dist in range(1, threshold + 1):
                            if dist in distance_layered_dict:
                                new_nodes.extend(distance_layered_dict[dist])
                        flag = False
                    else:
                        new_nodes = distance_layered_dict[threshold]
                    threshold_nodes.update(new_nodes)
                else:
                    threshold_nodes = [node for node, distance in shortest_path_lengths.items() if distance <= threshold]

                precision, error_rate = self.calculate_precision_and_error_rate(self.community_nodes, threshold_nodes)
                current_score = self.objective_function(precision, error_rate)

                if current_score > best_score or (current_score == best_score and threshold < best_threshold and best_threshold is None):
                    best_precision = precision
                    final_error_rate = error_rate
                    final_threshold_nodes = threshold_nodes
                    best_score = current_score
                    best_center_node = candidate_node
                    best_threshold = threshold
                    # 使用阈值限制
                    if if_limit_threshold == True:
                        center_threshold = threshold
                # 阈值剪枝
                if if_threshold_pruning == True:
                    if current_score > temp_best_score:
                        temp_best_score = current_score
                    else:
                            break
                if precision == 1.0:
                    break

        return best_score, best_precision, final_error_rate, best_center_node, best_threshold, proportion, final_threshold_nodes

    def get_top_nodes(self, num_nodes):
        """根据社区大小确定比例并筛选符合要求的点"""
        if self.if_part and num_nodes > 100:
            # 这里可以添加代码来选择社区中的重要节点
            return self.find_top_nodes()
        else:
            return self.community_nodes, 1

    # 根据社区大小确定比例
    def determine_proportion(self, community_size, max_community_size, min_proportion=0.05):
        # 使用Sigmoid函数调整比例，使得小社区有较高的比例，大社区有较低的比例
        # 这里我们调整Sigmoid函数，使其在社区大小接近max_community_size一半时开始快速下降
        k = 5 / max_community_size  # 控制曲线的陡峭程度
        x0 = max_community_size / 2  # 控制曲线的横向移动，这里设置为最大社区大小的一半
        proportion = min(1.0, min_proportion + (1 - 1 / (1 + math.exp(-k * (community_size - x0)))))

        return proportion


    # 按比例筛选符合要求的点
    def find_top_nodes(self):
        # 根据节点的度数进行排序
        degree_dict = {node: self.G.degree(node) for node in community_nodes}
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)  # 按度数从大到小排序
        # 计算取点比例
        proportion = self.determine_proportion(len(community_nodes), max_community_size)
        # 计算要取的节点数量
        num_top_nodes = max(100, int(len(sorted_nodes) * proportion))  # 确保至少有一个节点
        # 只取度数较大的点
        top_nodes = sorted_nodes[:num_top_nodes]
        # print(f'总数：{len(community_nodes)}，取节点数：{len(top_nodes)}，划分比例：{proportion * 100:.2f}%')
        return top_nodes, proportion

    def determine_threshold_range(self, max_distance, center_threshold, num_nodes):
        """确定阈值搜索范围"""
        if self.if_limit_threshold and center_threshold != -1 and num_nodes > 100:
            left = max(1, center_threshold - self.seta)
            right = min(center_threshold + self.seta, max_distance + 1)
        else:
            left = 1
            right = max_distance + 1
        return left, right

if __name__ == '__main__':
    current_method = method[num]
    for index in range(len(names)):
        print(f'当前社区：{names[index]}，社区划分方法：{current_method}')
        dataset_name = f'{names[index]}'
        G = nx.read_edgelist(f'data/graph/{dataset_name}.txt', nodetype=int)
        with open(f'./community_partitions/{current_method}/{dataset_name}_partition.pkl', 'rb') as f:
            partition = pickle.load(f)

        print('数据读取完成')

        # 提取每个社区节点并构建邻接表
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        communities = list(communities.values())
        if tag[i]==True:
            start_time = time.time()
        # 找到最大的社区
        max_community_size = max(len(community) for community in communities)
        # 打印最大社区的节点数量
        print(f"最大社区的节点数量为：{max_community_size}")

        if tag[i]==False:
            start_time = time.time()
        # 存储每个社区的中心节点、阈值、覆盖率等信息
        community_results = []
        community_thresholds = []
        for index, community_nodes in enumerate(communities):
            analyzer = CommunityAnalyzer(G, community_nodes, if_part=if_part, if_limit_threshold=if_limit_threshold, if_threshold_pruning=if_threshold_pruning, if_BFS=if_BFS)
            part_start_time = time.time()
            score, coverage_rate, error_rate, center_node, threshold_distance, proportion, threshold_nodes = analyzer.find_best_center_and_threshold()
            part_end_time = time.time()
            part_total_cost_time = part_end_time - part_start_time
            print(f'社区 {index}，中心节点 {center_node}，阈值 {threshold_distance}，耗时{part_total_cost_time * 1000}ms')
            community_thresholds.append((center_node, threshold_distance))
            community_nodes = list(set(community_nodes))
            community_results.append({
                '社区 ID': len(community_results),
                '节点个数': len(community_nodes),
                '可解释区域内的所有节点': threshold_nodes,
                '原始节点': community_nodes,
                '划分比例': proportion,
                '中心节点': center_node,
                '阈值': threshold_distance,
                '耗时': part_total_cost_time,
                '得分': score ,
                '覆盖率': coverage_rate,
                '错误率': error_rate
            })

        end_time = time.time()
        total_cost_time = end_time - start_time

        # 输出每个社区的结果
        for result in community_results:
            print(f"社区 {result['社区 ID']} --> 节点个数: {result['节点个数']}  得分: [{result['得分']:.2f}]")
            print(f"  中心节点: {result['中心节点']}, 阈值: {result['阈值']}")
            print(f"  原始节点: {result['原始节点']}")
            print(f"  可解释区域内的所有节点: {result['可解释区域内的所有节点']}")
            print(f"  划分比例：{result['划分比例'] * 100:.2f}%, 耗时: {result['耗时'] * 1000}ms")
            print(f"  覆盖率: {result['覆盖率']:.2%}, 错误率: {result['错误率']:.2%}\n")

        print(f"循环总执行时间：{total_cost_time * 1000}ms")
        # 计算社区总个数
        total_communities = len(community_results)

        # 计算平均得分、平均正确率、平均错误率
        average_score = np.mean([result['得分'] for result in community_results])
        average_precision = np.mean([result['覆盖率'] for result in community_results])  # 覆盖率即正确率
        average_error_rate = np.mean([result['错误率'] for result in community_results])

        # 打印结果
        print(f"社区总个数: {total_communities}")
        print(f"社区的平均得分: {average_score:.2f}")
        print(f"社区的平均正确率: {average_precision:.2%}")  # 转为百分比格式
        print(f"社区的平均错误率: {average_error_rate:.2%}")  # 转为百分比格式



        if if_write:
            # 检查result文件夹是否存在，如果不存在则创建
            result_dir = 'result'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # 检查dataset_name文件夹是否存在，如果不存在则创建
            # dataset_dir = os.path.join(result_dir, dataset_name)
            # if not os.path.exists(dataset_dir):
            #     os.makedirs(dataset_dir)
            param = ''
            if if_part:
                param += '1'
            else:
                param += '0'
            if if_limit_threshold:
                param += '_1'
            else:
                param += '_0'
            if if_threshold_pruning:
                param += '_1'
            else:
                param += '_0'
            # 打开一个文件用于写入，如果文件不存在则创建
            with open(f'result/{current_method}/{dataset_name}_{param}.txt', 'w', encoding='utf-8') as file:
                # 输出每个社区的结果
                for result in community_results:
                    file.write(f"社区 {result['社区 ID']} --> 节点个数: {result['节点个数']}  得分: [{result['得分']:.2f}]\n")
                    file.write(f"  中心节点: {result['中心节点']}, 阈值: {result['阈值']}\n")
                    file.write(f"  划分比例：{result['划分比例'] * 100:.2f}%, 耗时: {result['耗时'] * 1000}ms\n")
                    file.write(f"  覆盖率: {result['覆盖率']:.2%}, 错误率: {result['错误率']:.2%}\n\n")

                # 输出循环总执行时间
                file.write(f"循环总执行时间：{total_cost_time * 1000}ms\n")
                file.write("\n===================\n")
                file.write(f"社区总个数: {total_communities}\n")
                file.write(f"社区的平均得分: {average_score:.2f}\n")
                file.write(f"社区的平均正确率: {average_precision:.2%}\n")
                file.write(f"社区的平均错误率: {average_error_rate:.2%}\n")
        print('#############################################################\n\n')