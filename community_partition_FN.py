import networkx as nx
import pickle
import os
import random
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities  # 导入 FN 算法模块

# 固定随机种子 不受随机种子的影响
random.seed(42)
np.random.seed(42)


def save_community_partition(dataset_name, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载图数据
    G = nx.read_edgelist(f'data/graph/{dataset_name}.txt', nodetype=int)

    # 执行 Fast Newman (FN) 社区发现算法
    print(f"Performing Fast Newman (FN) community detection for {dataset_name}...")
    communities = greedy_modularity_communities(G)  # 使用 FN 算法获取社区
    communities = [sorted(list(community)) for community in communities]  # 将社区结果转为列表并排序

    # 创建一个字典来存储节点和社区编号的映射
    partition = {}
    for community_index, community in enumerate(communities):
        for node in community:
            partition[node] = community_index

    # 保存社区划分结果到 .pkl 文件
    partition_filename = f'{dataset_name}_partition.pkl'
    with open(os.path.join(output_folder, partition_filename), 'wb') as f:
        pickle.dump(partition, f)
    print(f"Community partition for {dataset_name} has been saved in {output_folder} as a .pkl file.")

    # 保存社区划分结果到 .txt 文件
    txt_filename = f'{dataset_name}_communities.txt'
    with open(os.path.join(output_folder, txt_filename), 'w') as f:
        for community in communities:
            f.write(" ".join(map(str, community)) + "\n")
    print(f"Community partition for {dataset_name} has been saved in {output_folder} as a .txt file.")


if __name__ == '__main__':
    # 数据集名称
    # names = ['web-EPA', 'web-edu', 'web-spam', 'road-minnesota']  # 数据集列表
    names = ['karate', 'football', 'personal', 'polblogs', 'polbooks', 'railways']

    # names = ['cit-DBLP']
    # 输出文件夹
    output_folder = 'community_partitions/FN'
    for name in names:
        save_community_partition(name, output_folder)
