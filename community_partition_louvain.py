import networkx as nx
import pickle
import os
import random
import numpy as np
from community import community_louvain  # 需要安装 python-louvain 库

# 固定随机种子
random.seed(42)
np.random.seed(42)


def save_community_partition(dataset_name, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载图数据
    G = nx.read_edgelist(f'data/graph/{dataset_name}.txt', nodetype=int)

    # 执行 Louvain 社区发现算法
    print(f"Performing Louvain community detection for {dataset_name}...")
    partition = community_louvain.best_partition(G)

    # 创建一个空列表来存储社区的节点列表
    community_list = {}
    for node, community_id in partition.items():
        if community_id not in community_list:
            community_list[community_id] = []
        community_list[community_id].append(node)

    # 将社区列表按社区编号排序，并保存每个社区的节点
    sorted_communities = []
    for community_id in sorted(community_list.keys()):
        sorted_communities.append(sorted(community_list[community_id]))

    # 保存社区划分结果到 .pkl 文件
    partition_filename = f'{dataset_name}_partition.pkl'
    with open(os.path.join(output_folder, partition_filename), 'wb') as f:
        pickle.dump(partition, f)
    print(f"Community partition for {dataset_name} has been saved in {output_folder} as a .pkl file.")

    # 保存社区划分结果到 .txt 文件
    txt_filename = f'{dataset_name}_communities.txt'
    with open(os.path.join(output_folder, txt_filename), 'w') as f:
        for community in sorted_communities:
            f.write(" ".join(map(str, community)) + "\n")
    print(f"Community partition for {dataset_name} has been saved in {output_folder} as a .txt file.")


if __name__ == '__main__':
    # names = ['karate', 'football', 'personal', 'polblogs', 'polbooks', 'railways', 'cit-DBLP', 'web-edu', 'web-spam', 'road-minnesota']
    names = ['cit-DBLP']
    output_folder = 'community_partitions/Louvain'
    for name in names:
        save_community_partition(name, output_folder)