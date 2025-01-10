import networkx as nx
import pickle
import os
import random
import numpy as np
from collections import defaultdict
from networkx.algorithms.community import label_propagation_communities

# 固定随机种子
random.seed(42)
np.random.seed(42)


def save_community_partition(dataset_name, output_folder, num_runs=10):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载图数据
    G = nx.read_edgelist(f'data/graph/{dataset_name}.txt', nodetype=int)

    # 保存所有运行结果的分区
    all_partitions = []

    print(f"Performing LPA community detection for {dataset_name} with {num_runs} runs...")
    for run in range(num_runs):
        # 每次运行需要一个新的随机种子
        random.seed(42 + run)
        np.random.seed(42 + run)

        communities = label_propagation_communities(G)

        # 创建一个字典来存储当前运行的节点到社区的映射
        partition = {}
        community_index = 0
        for community in communities:
            for node in community:
                partition[node] = community_index
            community_index += 1

        all_partitions.append(partition)

    # 统计每个节点的社区编号的众数（多次运行后最常出现的社区）
    final_partition = {}
    for node in G.nodes():
        community_votes = [partition[node] for partition in all_partitions if node in partition]
        final_partition[node] = max(set(community_votes), key=community_votes.count)  # 众数

    # 创建社区列表
    final_community_list = defaultdict(list)
    for node, community in final_partition.items():
        final_community_list[community].append(node)
    final_community_list = [sorted(nodes) for nodes in final_community_list.values()]

    # 保存最终的社区划分结果到 .pkl 文件
    partition_filename = f'{dataset_name}_partition.pkl'
    with open(os.path.join(output_folder, partition_filename), 'wb') as f:
        pickle.dump(final_partition, f)
    print(f"Final community partition for {dataset_name} has been saved in {output_folder} as a .pkl file.")

    # 保存最终的社区划分结果到 .txt 文件
    txt_filename = f'{dataset_name}_communities.txt'
    with open(os.path.join(output_folder, txt_filename), 'w') as f:
        for community in final_community_list:
            f.write(" ".join(map(str, community)) + "\n")
    print(f"Final community partition for {dataset_name} has been saved in {output_folder} as a .txt file.")


if __name__ == '__main__':
    names = ['karate', 'football', 'personal', 'polblogs', 'polbooks', 'railways', 'cit-DBLP', 'web-edu', 'web-spam', 'road-minnesota']

    output_folder = 'community_partitions/LPA'
    for name in names:
        save_community_partition(name, output_folder)
