import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from scipy.stats import norm
# plt.style.use(['science', 'ieee'])

def plot_scatter_with_lines(data, labels, x_labels, save_path=None):
    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    for i, points in enumerate(data):
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        fig_i = i // int(len(data) / 2)
        inner_i = i % int(len(data) / 2)
        if fig_i == 0:
            axs[0].plot(x_values, y_values, label=labels[inner_i], marker='o')
            axs[0].set_xlabel(x_labels[fig_i])
            axs[0].set_ylabel('ImageNet Top1 Accuracy')
            # axs[0].set_title('Scatter Plot with Lines (BitOps)')
        elif fig_i == 1:
            axs[1].plot(x_values, y_values, label=labels[inner_i], marker='o')
            axs[1].set_xlabel(x_labels[fig_i])
            # axs[1].set_ylabel('ImageNet Top1 Accuracy')
            # axs[1].set_title('Scatter Plot with Lines (Params)')

        # # 添加注释
        # for j, point in enumerate(points):
        #     axs[i].annotate(f'{labels[i]}_{j+1}', (point[0], point[1]), textcoords="offset points", xytext=(0,5), ha='center')

    axs[0].legend()
    axs[1].legend()
    # axs[0].grid(True)
    # axs[1].grid(True)
    plt.subplots_adjust(hspace=0.2)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.cla()

def plot_ofa_data():
    # 4g,   #w4a4,   #w5a5,   #w6a6
    data_points = [
        # BitOps
        [(3.5, 47.66), (4, 53.09), (4.5, 57.02), (5.394, 65.67), (7.991, 67.18), (11.167, 67.21)],
        [(3.5, 63.23), (4, 65.35), (4.5, 66.57), (5.394, 67.36), (7.991, 68.66), (11.167, 68.88)],
        [(5.394, 62.01), (7.991, 67.65), (11.167, 67.69)],
        [(5.394, 48.72), (7.991, 60.29), (11.167, 63.99)],
        # Params
        [(16.822, 54.31), (18, 62.4), (19.011, 63.1), (20, 66.47), (21.199, 67.18), (23.388, 67.21)],
        [(16.822, 65.7), (18, 67.78), (19.011, 68.29), (20, 68.52), (21.199, 68.66), (23.388, 68.88)],
        [(16.822, 50), (19.011, 60), (21.199, 65), (23.388, 70)],
        [(19.011, 48.72), (21.199, 60.29), (23.388, 67.69)]
    ]

    group_labels = ['Cobits_S1', 'Cobits_S2', 'LSQ_S1', 'LSQ_S2']

    # with plt.style.context(['science', 'ieee']):
    plot_scatter_with_lines(data_points, group_labels, x_labels=['BitOps(G)', 'Params(M)'], save_path='ofa_results.png')

# plot_ofa_data()

def plot_curve_with_fill(split=2, save_path=None):
    # 生成正态分布的数据
    assert split in [2, 4], f'Unsupported split: {split}'
    mu = 0
    sigma = 0.2 if split == 2 else 0.2
    x = np.linspace(mu - split*sigma, mu + split*sigma, 100)
    y = norm.pdf(x, mu, sigma)

    # 绘制正态分布曲线
    plt.plot(x, y, '#B0A2C3', linewidth=2)

    # 标注均值和标准差
    # plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label='Mean')
    # plt.axvline(mu + sigma, color='green', linestyle='dashed', linewidth=2, label='Mean + Std Dev')
    # plt.axvline(mu - sigma, color='green', linestyle='dashed', linewidth=2, label='Mean - Std Dev')

    if split == 2:
        fill_x = np.linspace(mu - 2*sigma, mu, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#00C9A7', alpha=1.0)
        fill_x = np.linspace(mu, mu + 2*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#CC6222', alpha=1.0)
    else:        
        fill_x = np.linspace(mu - 4*sigma, mu - 2*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#80E8FD', alpha=1.0)
        fill_x = np.linspace(mu - 2*sigma, mu, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#80E4D3', alpha=1.0) 
        fill_x = np.linspace(mu, mu + 2*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#E1A49A', alpha=1.0)               
        fill_x = np.linspace(mu + 2*sigma, mu + 4*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#F5EB88', alpha=1.0)
    # 填充曲线下不同的面积部分，并标注面积
    # for i in range(split):
    #     fill_x = np.linspace(mu - i*sigma, mu + i*sigma, 100)
    #     fill_y = norm.pdf(fill_x, mu, sigma)
    #     plt.fill_between(fill_x, fill_y, color='yellow', alpha=0.3, label='68% Area')

    #     fill_x_2std = np.linspace(mu - 2*sigma, mu + 2*sigma, 100)
    #     fill_y_2std = norm.pdf(fill_x_2std, mu, sigma)
    #     plt.fill_between(fill_x_2std, fill_y_2std, color='orange', alpha=0.3, label='95% Area')

    #     fill_x_3std = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    #     fill_y_3std = norm.pdf(fill_x_3std, mu, sigma)
    #     plt.fill_between(fill_x_3std, fill_y_3std, color='red', alpha=0.3, label='99.7% Area')

    plt.xlim(-1.0, 1.0)
    plt.xticks([-1.0, -0.5, 0, 0.5, 1.0], size=15)
    plt.yticks([])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_color('none')
    # plt.legend()

    # 显示图形
    plt.show()

    plt.savefig(save_path, bbox_inches='tight')
    plt.cla()

plot_curve_with_fill(split=2, save_path='demo_split2.png')
plot_curve_with_fill(split=4, save_path='demo_split4.png')