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
names = ['karate','football', 'railways', 'personal', 'polbooks', 'polblogs', 'road-minnesota','web-spam', 'cit-DBLP']
titles = ['Karate','Football', 'Railways', 'Personal', 'Polbooks', 'Polblogs', 'Minnesota', 'Spam', 'DBLP']  # 对应数据集的标题
data_base_dir = 'boxplot/data'  # 数据读取路径
result_base_dir = 'boxplot/result'  # 数据存储路径

# 从 .txt 文件中提取社区结果
def parse_community_results(file_path):
    scores = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 匹配得分
            score_match = re.search(r'得分: \[(.*?)\]', line)
            if score_match:
                score = float(score_match.group(1))
                if score > 1:  # 检查得分是否大于 1
                    score /= 100  # 如果大于 1，除以 100
                scores.append(score)

    return scores

# 绘制所有数据集在指定算法上的得分箱线图
# 绘制所有数据集在指定算法上的得分箱线图
def plot_scores_for_all_datasets(method, names, titles, data_base_dir, result_base_dir):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # 设置为 3x3 格式，调整图整体尺寸
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # 调整子图之间的间距

    for i, (dataset, title) in enumerate(zip(names, titles)):
        data_dir = os.path.join(data_base_dir, method)  # 当前 method 的数据路径
        naive_file = os.path.join(data_dir, f"{dataset}_0_0_0.txt")  # Naive 文件
        fast_file = os.path.join(data_dir, f"{dataset}_1_1_1.txt")  # Fast 文件

        # 确定子图位置
        row, col = divmod(i, 3)

        # 检查文件是否存在
        if not os.path.exists(naive_file) or not os.path.exists(fast_file):
            print(f"Files for {method} in dataset {dataset} not found. Skipping...")
            axs[row, col].set_title(f"{title} (No Data)")
            axs[row, col].axis('off')  # 隐藏没有数据的子图
            continue

        # 从文件中解析数据
        naive_scores = parse_community_results(naive_file)
        fast_scores = parse_community_results(fast_file)

        # 准备箱线图数据
        results = {
            "Objective Score": [naive_scores, fast_scores]
        }

        # 创建子图
        labels = ["Naive", "Fast"]
        axs[row, col].boxplot(results["Objective Score"], vert=True, patch_artist=True, showmeans=True, labels=labels)
        axs[row, col].set_ylim(0, 1.05)  # 设置纵坐标范围稍微超过 1
        axs[row, col].set_xticks([1, 2])  # 确保x刻度正常显示
        axs[row, col].set_xlabel(title, fontsize=12)  # 数据集名称放在图下面
        axs[row, col].set_ylabel("Score", fontsize=12)  # 设置纵坐标名称
        axs[row, col].tick_params(axis='both', which='major', labelsize=12)

    # 移除空白子图
    for i in range(len(names), 9):
        row, col = divmod(i, 3)
        axs[row, col].axis('off')

    plt.tight_layout()

    # 确保存储路径存在
    result_dir = os.path.join(result_base_dir, method)
    os.makedirs(result_dir, exist_ok=True)

    # 保存结果
    output_path = os.path.join(result_dir, f"{method}_all_datasets_score_boxplot.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Combined boxplot for {method} saved at {output_path}")
    plt.show()

# 主流程
if __name__ == '__main__':
    method_index = 0 # 指定要展示的算法索引
    method = methods[method_index]  # 当前选择的算法
    plot_scores_for_all_datasets(method, names, titles, data_base_dir, result_base_dir)
