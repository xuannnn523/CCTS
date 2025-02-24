import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置
plt.rcParams.update({
    'font.size': 18,  # 全局字体大小
    'axes.titlesize': 20,  # 子图标题字体大小
    'axes.labelsize': 18,  # 坐标轴标签字体大小
    'xtick.labelsize': 16,  # x轴刻度字体大小
    'ytick.labelsize': 16,  # y轴刻度字体大小
    'legend.fontsize': 18,  # 图例字体大小
    'figure.titlesize': 22  # 整体标题字体大小
})

# 数据集名称（按显示名称）
datasets = ['karate','football', 'polbooks', 'railways', 'personal', 'polblogs', 'road-minnesota','web-spam', 'cit-DBLP']
titles = ['Karate','Football', 'Polbooks', 'Railways', 'Personal', 'Polblogs', 'Minnesota', 'Spam', 'DBLP']  # 对应数据集的标题

# 算法名称
methods = ["Louvain", "GN", "FN", "LPA"]

# 设置路径
data_base_dir = 'boxplot/data'  # 数据读取路径

# 从 .txt 文件中提取执行时间
def parse_execution_times(file_path):
    execution_time = 0

    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist.")
        return execution_time

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 提取循环总执行时间
        execution_time_match = re.search(r"循环总执行时间：([0-9.]+)ms", content)
        if execution_time_match:
            execution_time = float(execution_time_match.group(1))

    return execution_time


# 为每个方法绘制图表
def plot_method_chart(ax, method, titles, naive_times, fast_times, baseline_times):
    x_positions = np.arange(len(titles))
    width = 0.25  # 每个柱子的宽度

    # Baseline、Naive和Fast的位置
    baseline_pos = x_positions - width
    naive_pos = x_positions
    fast_pos = x_positions + width

    # 绘制柱状图
    ax.bar(baseline_pos, baseline_times[method], width, color="#2CA02C", alpha=0.7, label=f'Baseline ({method})')
    ax.bar(naive_pos, naive_times[method], width,color="#4C72B0", alpha=1, label=f'Naive ({method})')
    ax.bar(fast_pos, fast_times[method], width, color="#E34A33", alpha=0.7,  label=f'Fast ({method})')

    # 设置标题和标签
    ax.set_ylabel("Running Time (ms)", fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(titles, rotation=45, ha="right", fontsize=16)  # 设置标题显示为数据集标题
    ax.set_yscale('log')  # 使用对数坐标轴
    ax.set_ylim(10 ** -1, 10 ** 8)  # 设置y轴的范围
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 修改图例，使其按一行显示，并确保顺序为 Baseline, Naive, Fast
    ax.legend(fontsize=10, ncol=3, loc="upper left", frameon=False)  # 图例字体变小，且在一行显示
    ax.set_facecolor('#f7f7f7')  # 干净的背景色
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 增大y轴刻度字体
    ax.tick_params(axis='y', labelsize=16)


# 读取并解析所有文件的执行时间
def get_execution_times(methods, datasets, data_base_dir):
    naive_times = {method: [] for method in methods}
    fast_times = {method: [] for method in methods}
    baseline_times = {method: [] for method in methods}

    for method in methods:
        for dataset in datasets:
            # 构建文件路径
            data_dir = os.path.join(data_base_dir, method)
            baseline_file = os.path.join(data_dir, f"{dataset}_2_2_2.txt")  # Baseline 文件
            naive_file = os.path.join(data_dir, f"{dataset}_0_0_0.txt")  # Naive 文件
            fast_file = os.path.join(data_dir, f"{dataset}_1_1_1.txt")  # Fast 文件

            # 提取执行时间
            baseline_times[method].append(parse_execution_times(baseline_file))
            naive_times[method].append(parse_execution_times(naive_file))
            fast_times[method].append(parse_execution_times(fast_file))

    return naive_times, fast_times, baseline_times


# 创建图形，包含2x2子图
def plot_all_methods(datasets, methods, data_base_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 调整图形大小，字体更大

    # 拉平子图数组以便迭代
    axes = axes.flatten()

    # 获取每个方法的执行时间
    naive_times, fast_times, baseline_times = get_execution_times(methods, datasets, data_base_dir)

    # 为每个方法绘制单独的图表
    for i, method in enumerate(methods):
        plot_method_chart(axes[i], method, titles, naive_times, fast_times, baseline_times)

    # 调整布局，避免重叠
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 保存图形为PDF文件
    plt.savefig("community_detection_execution_times.pdf", format='pdf', bbox_inches="tight")
    plt.show()


# 主流程
if __name__ == '__main__':
    plot_all_methods(datasets, methods, data_base_dir)
