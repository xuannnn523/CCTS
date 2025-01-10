import networkx as nx
import pickle
import os
import random
import numpy as np
from networkx.algorithms.community import girvan_newman

# 固定随机种子 不受随机种子的影响
random.seed(42)
np.random.seed(42)


def save_community_partition(dataset_name, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载图数据
    G = nx.read_edgelist(f'data/graph/{dataset_name}.txt', nodetype=int)

    # 执行 GN-Newman 社区发现算法
    print(f"Performing GN-Newman community detection for {dataset_name}...")
    communities_generator = girvan_newman(G)
    # 选择前 k 个分裂，生成的社区（这里选择第一层分裂结果）
    top_level_communities = next(communities_generator)
    communities = [sorted(list(community)) for community in top_level_communities]

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
    # names = ['football', 'karate', 'personal', 'polblogs', 'railways']
    # names = ['karate', 'football', 'personal', 'polblogs', 'polbooks', 'railways', 'cit-DBLP', 'web-edu', 'web-spam', 'road-minnesota']

    # names = ['flickr']
    # names = ['web-EPA']
    # names = ['web-spam','road-minnesota']
    names = ['cit-DBLP']

    output_folder = 'community_partitions/GN'
    for name in names:
        save_community_partition(name, output_folder)