import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

data = pd.DataFrame(
    [
        [0.5, 0, 1],
        [0.5, 0.3, 2],
        [0.5, 0.6, 3],
        [0.5, 0.9, 2],
        [0.5, 1.2, 3],
        [0.5, 1.5, 2],
        [0.5, 1.8, 3],
        [0.5, 2.1, 50],
        [0.8, 0, 4],
        [0.8, 0.3, 5],
        [0.8, 0.6, 6],
        [0.8, 0.9, 4],
        [0.8, 1.2, 5],
        [0.8, 1.5, 6],
        [0.8, 1.8, 4],
        [0.8, 2.1, 50],
        [1.1, 0, 7],
        [1.1, 0.3, 50],
        [1.1, 0.6, 50],
        [1.1, 0.9, 50],
        [1.1, 1.2, 8],
        [1.1, 1.5, 9],
        [1.1, 1.8, 7],
        [1.1, 2.1, 50],
        [1.4, 0, 10],
        [1.4, 0.3, 11],
        [1.4, 0.6, 50],
        [1.4, 0.9, 10],
        [1.4, 1.2, 11],
        [1.4, 1.5, 50],
        [1.4, 1.8, 10],
        [1.4, 2.1, 11],
        [1.7, 0, 10],
        [1.7, 0.3, 11],
        [1.7, 0.6, 50],
        [1.7, 0.9, 10],
        [1.7, 1.2, 11],
        [1.7, 1.5, 50],
        [1.7, 1.8, 10],
        [1.7, 2.1, 50],
    ],
    columns=['Y', 'X', 'W'],
)

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(6.6, 4), frameon=False)
    ax.set_title('Sciplex4', pad=10, fontsize=20, fontweight='bold')
    ax.scatter(
        x=data.X,
        y=data.Y,
        s=data.W * 17,
        c=data.W * 17,
        cmap='Blues',
        edgecolors=['gray'],
        linewidths=0.8,
    )
    ax.set_ylabel('Mean gene expression', fontsize=19, labelpad=10)
    ax.yaxis.set_label_position('right')

    ax.set_ylim(0.3, 1.9)
    ax.set_yticks([0.5, 0.8, 1.1, 1.4, 1.7])
    ax.set_yticklabels(
        ['sfees', 'esfse', 'efs', 'sef', 'sfse'], fontsize=12, fontweight='bold'
    )
    ax.set_xlim(-0.3, 2.4)
    ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1])
    ax.set_xticklabels(
        ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Gene6', 'Gene7', 'Gene8'],
        rotation=87,
        fontsize=12,
        fontweight='bold',
    )
    # ax

    # 设置刻度
    ax.tick_params(axis='x', length=7, width=2, which='major', pad=5)
    ax.tick_params(axis='y', length=0, width=0, which='major', pad=5)

    # 设置 colorbar
    norm = colors.Normalize(vmin=-0.2, vmax=5)
    im = cm.ScalarMappable(norm=norm, cmap='Blues')
    # 设置 colorbar 的位置
    cax = fig.add_axes([0.9, 0.23, 0.025, 0.54])
    cbar = fig.colorbar(im, cax=cax, pad=0.09)
    # 设置 colorbar 的刻度范围和刻度值
    cbar.set_ticks([0, 2, 4])
    cbar.update_ticks()
    # 设置 colorbar 的刻度字体和刻度粗细
    cbar.ax.tick_params(labelsize=12, length=7, width=2, pad=5, labelfontfamily='bold')

    # 调整图像边距
    plt.subplots_adjust(bottom=0.2, right=0.82, top=0.87)

    # plt.rcParams['font.sans-serif'] = 'Bradley Hand ITC'
    plt.show()
