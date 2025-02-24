import os
import re
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 12,           # 默认字体大小
    'axes.titlesize': 14,      # 子图标题字体大小
    'axes.labelsize': 12,      # 坐标轴标签字体大小
    'xtick.labelsize': 12,     # x轴刻度标签字体大小
    'ytick.labelsize': 12,     # y轴刻度标签字体大小
    'legend.fontsize': 12,     # 图例字体大小
    'figure.titlesize': 16     # 图形标题字体大小
})

# 当前的社区发现算法和数据集名称
methods = ['Louvain', 'GN', 'FN', 'LPA']
names = ['karate', 'football', 'railways', 'personal', 'polbooks', 'polblogs', 'road-minnesota', 'web-spam', 'cit-DBLP']
titles = ['Karate', 'Football', 'Railways', 'Personal', 'Polbooks', 'Polblogs', 'Minnesota', 'Spam', 'DBLP']  # 对应数据集的标题
data_base_dir = 'boxplot/data'  # 数据读取路径

# 从 .txt 文件中提取阈值数据
def extract_threshold_data(file_path):
    thresholds = []

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return thresholds

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # 提取阈值
        matches = re.findall(r'阈值: (\d+)', content)
        thresholds = [int(match) for match in matches]

    return thresholds

# 绘制所有数据集的阈值箱线图
def plot_thresholds_for_all_datasets(method, names, titles, data_base_dir):
    # 动态设置输出路径到 method 文件夹
    output_dir = os.path.join('boxplot/result', method)
    os.makedirs(output_dir, exist_ok=True)  # 创建路径（如果不存在）

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # 设置为 3x3 格式，调整图整体尺寸
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # 调整子图之间的间距

    for i, (dataset, title) in enumerate(zip(names, titles)):
        data_dir = os.path.join(data_base_dir, method)  # 当前 method 的数据路径
        naive_file = os.path.join(data_dir, f"{dataset}_0_0_0.txt")  # Naive 文件
        fast_file = os.path.join(data_dir, f"{dataset}_1_1_1.txt")  # Fast 文件
        baseline_file = os.path.join(data_dir, f"{dataset}_2_2_2.txt")  # Baseline 文件

        # 确定子图位置
        row, col = divmod(i, 3)

        # 检查文件是否存在
        if not os.path.exists(naive_file) or not os.path.exists(fast_file) or not os.path.exists(baseline_file):
            print(f"Files for {method} in dataset {dataset} not found. Skipping...")
            axs[row, col].set_title(f"{title} (No Data)")
            axs[row, col].axis('off')  # 隐藏没有数据的子图
            continue

        # 提取数据
        naive_thresholds = extract_threshold_data(naive_file)
        fast_thresholds = extract_threshold_data(fast_file)
        baseline_thresholds = extract_threshold_data(baseline_file)

        # 准备箱线图数据
        results = [baseline_thresholds, naive_thresholds, fast_thresholds]  # 顺序调整为 baseline, naive, fast

        # 创建子图
        labels = ["Baseline", "Naive", "Fast"]
        box = axs[row, col].boxplot(
            results,
            vert=True,
            patch_artist=True,
            showmeans=True,
            labels=labels
        )

        # 设置箱线图颜色为蓝色
        for patch in box['boxes']:
            patch.set_facecolor('#4C72B0')  # 蓝色

        # 修复y轴显示问题
        if baseline_thresholds and naive_thresholds and fast_thresholds:
            y_min = min(min(baseline_thresholds), min(naive_thresholds), min(fast_thresholds))
            y_max = max(max(baseline_thresholds), max(naive_thresholds), max(fast_thresholds))
            axs[row, col].set_ylim(y_min - 1, y_max + 1)  # 确保显示完整范围
            axs[row, col].set_yticks(range(y_min, y_max + 1, max(1, (y_max - y_min) // 5)))  # 调整y轴刻度间隔
        else:
            axs[row, col].set_ylim(0, 1)  # 如果没有数据，设定一个默认范围

        # 设置子图标题为数据集名称（直接放在x轴标签下）
        axs[row, col].set_xticklabels(labels, fontsize=10)  # 显示 Baseline, Naive 和 Fast
        axs[row, col].set_xlabel(title, fontsize=12)  # 数据集名称显示在横轴

        # 移除横轴标题
        axs[row, col].set_title("")  # 删除子图标题

        # 添加纵轴标签
        axs[row, col].set_ylabel('Threshold', fontsize=10)

    # 隐藏空白子图
    for i in range(len(names), 9):
        row, col = divmod(i, 3)
        axs[row, col].axis('off')

    # 添加全局标题
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整全局标题和子图间距

    # 保存图表为 PDF 文件
    output_file = os.path.join(output_dir, f"{method}_threshold_boxplot.pdf")
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Boxplot saved to: {output_file}")

    # 显示图表
    plt.show()


# 主函数
if __name__ == '__main__':
    method_index = 3  # 指定要展示的算法索引
    method = methods[method_index]  # 当前选择的算法
    plot_thresholds_for_all_datasets(method, names, titles, data_base_dir)
