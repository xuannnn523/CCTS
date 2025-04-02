import networkx as nx
import pickle
import os
import numpy as np
import random
import time
import math

# 方法列表
method = ['Louvain', 'GN', 'LPA', 'FN']
num = 2
names = ['karate', 'football', 'personal', 'polblogs', 'polbooks', 'railways', 'cit-DBLP', 'web-spam',
         'road-minnesota']
index = 8

tag = [ False,True]
i = 0

# 是否将结果写入文件
if_write = True
if_part = tag[i]
if_limit_threshold = tag[i]
if_threshold_pruning = tag[i]
if_BFS = tag[i]
random.seed(42)
np.random.seed(42)


class CommunityAnalyzer:
    def __init__(self, G, community_nodes, alpha=1, beta=1, seta=1, if_part=True, if_limit_threshold=True,
                 if_threshold_pruning=True, if_BFS=True):
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
        error_rate = (len(threshold_nodes) - correct_count) / len(threshold_nodes) if threshold_nodes else 0
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

        # 使用DC来选择社区的中心节点
        centrality = nx.degree_centrality(self.G)  # 计算所有节点的度中心性
        sorted_nodes_by_dc = sorted(self.community_nodes, key=lambda node: centrality[node], reverse=True)  # 按照DC排序

        # 选择度中心性最高的节点作为中心节点
        best_center_node = sorted_nodes_by_dc[0]
        print(f"选定的中心节点为: {best_center_node}")

        # 获取社区的最短路径
        shortest_path_lengths = nx.single_source_shortest_path_length(self.G, best_center_node)
        max_distance = max(
            shortest_path_lengths[node] for node in self.community_nodes if node in shortest_path_lengths)

        # 动态调整阈值范围
        left, right = self.determine_threshold_range(max_distance, center_threshold, len(self.community_nodes))

        temp_best_score = 0
        flag = True

        for threshold in range(left, right):
            threshold_nodes = set([best_center_node])

            if self.if_BFS:
                distance_layered_dict = {}
                for target_node, distance in shortest_path_lengths.items():
                    if distance not in distance_layered_dict:
                        distance_layered_dict[distance] = []
                    distance_layered_dict[distance].append(target_node)

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

            if current_score > best_score or (
                    current_score == best_score and threshold < best_threshold and best_threshold is None):
                best_precision = precision
                final_error_rate = error_rate
                final_threshold_nodes = threshold_nodes
                best_score = current_score
                best_threshold = threshold
                center_threshold = threshold

            if self.if_threshold_pruning and current_score < temp_best_score:
                break
            temp_best_score = current_score

        return best_score, best_precision, final_error_rate, best_center_node, best_threshold, final_threshold_nodes

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
        G = nx.read_edgelist(f'data/{dataset_name}.txt', nodetype=int)
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

        # # 找到最大的社区
        # max_community_size = max(len(community) for community in communities)
        # print(f"最大社区的节点数量为：{max_community_size}")

        # 记录开始时间
        start_time = time.time()

        # 存储每个社区的中心节点、阈值、覆盖率等信息
        community_results = []
        for index, community_nodes in enumerate(communities):
            analyzer = CommunityAnalyzer(G, community_nodes, if_part=if_part, if_limit_threshold=if_limit_threshold,
                                         if_threshold_pruning=if_threshold_pruning, if_BFS=if_BFS)
            part_start_time = time.time()
            score, coverage_rate, error_rate, center_node, threshold_distance, threshold_nodes = analyzer.find_best_center_and_threshold()
            part_end_time = time.time()
            part_total_cost_time = part_end_time - part_start_time
            print(f'社区 {index}，中心节点 {center_node}，阈值 {threshold_distance}，耗时{part_total_cost_time * 1000}ms')

            community_results.append({
                '社区 ID': len(community_results),
                '节点个数': len(community_nodes),
                '原始节点': community_nodes,
                '中心节点': center_node,
                '阈值': threshold_distance,
                '覆盖率': coverage_rate,
                '错误率': error_rate,
                '得分': score,
                '耗时': part_total_cost_time
            })

        end_time = time.time()
        total_cost_time = end_time - start_time

        # 输出每个社区的结果
        for result in community_results:
            print(f"社区 {result['社区 ID']} --> 节点个数: {result['节点个数']}  得分: [{result['得分']:.2f}]")
            print(f"  中心节点: {result['中心节点']}, 阈值: {result['阈值']}")
            print(f"  原始节点: {result['原始节点']}")
            print(f"  覆盖率: {result['覆盖率']:.2%}, 错误率: {result['错误率']:.2%}\n")

        print(f"总耗时: {total_cost_time * 1000:.2f}毫秒")

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
            result_dir = 'result'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            param = ''
            if if_part:
                param += '1'
            else:
                param += '2'
            if if_limit_threshold:
                param += '_1'
            else:
                param += '_2'
            if if_threshold_pruning:
                param += '_1'
            else:
                param += '_2'

            # 保存结果到文件
            with open(f'result/{current_method}/{dataset_name}_{param}.txt', 'w', encoding='utf-8') as file:
                for result in community_results:
                    file.write(
                        f"社区 {result['社区 ID']} --> 节点个数: {result['节点个数']} 得分: [{result['得分']:.2f}]\n")
                    file.write(f"  中心节点: {result['中心节点']}, 阈值: {result['阈值']}\n")
                    file.write(f"  原始节点: {result['原始节点']}\n")
                    file.write(f"  覆盖率: {result['覆盖率']:.2%}, 错误率: {result['错误率']:.2%}\n\n")

                # 输出循环总执行时间
                file.write(f"循环总执行时间：{total_cost_time * 1000}ms\n")
                file.write("\n===================\n")
                file.write(f"社区总个数: {total_communities}\n")
                file.write(f"社区的平均得分: {average_score:.2f}\n")
                file.write(f"社区的平均正确率: {average_precision:.2%}\n")
                file.write(f"社区的平均错误率: {average_error_rate:.2%}\n")
        print('#############################################################\n\n')