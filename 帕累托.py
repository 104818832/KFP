from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. 输出文件夹
# ============================================================
SAVE_DIR = Path(
    r"D:\2026wenjian\agriculture-4429543\KFP-YOLO-source\pareto"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 主流模型对比数据
# 数据格式：
# 模型名称、GFLOPs、mAP@0.5:0.95均值、标准差
# ============================================================
MAINSTREAM_MODELS = [
    ("YOLOv5n", 7.19, 0.6970, 0.0053),
    ("YOLOv8n", 8.20, 0.7516, 0.0038),
    ("YOLO11n", 6.45, 0.7550, 0.0049),
    ("YOLO12n", 6.49, 0.7808, 0.0027),
    ("YOLOv13n", 6.30, 0.8065, 0.0038),
    ("YOLOv13s", 22.30, 0.8715, 0.0046),
    ("YOLO26n", 5.79, 0.7579, 0.0044),
    ("YOLO26s", 22.54, 0.8362, 0.0035),
]


# ============================================================
# 3. 消融实验数据
# 数据格式：
# 模型名称、GFLOPs、mAP@0.5:0.95均值、标准差
# ============================================================
ABLATION_MODELS = [
    ("YOLO26n", 5.79, 0.7579, 0.0044),
    ("+ADown", 5.09, 0.7511, 0.0042),
    ("+C3-PD", 5.12, 0.7134, 0.0048),
    ("+CFA", 5.35, 0.7410, 0.0031),
    ("+ADown +C3-PD", 4.41, 0.7352, 0.0045),
    ("+ADown +CFA", 5.03, 0.7582, 0.0030),
    ("+C3-PD +CFA", 5.13, 0.7168, 0.0037),
    ("KFP-YOLO", 4.42, 0.7434, 0.0043),
]


# ============================================================
# 4. 标签位置偏移
# 单位为屏幕点，防止模型名称互相遮挡
# ============================================================
MAINSTREAM_LABEL_OFFSETS = {
    "YOLOv5n": (10, -22),
    "YOLOv8n": (10, 8),
    "YOLO11n": (-75, -18),
    "YOLO12n": (10, 8),
    "YOLOv13n": (10, 8),
    "YOLOv13s": (-78, 8),
    "YOLO26n": (-75, 8),
    "YOLO26s": (-78, -20),
}

ABLATION_LABEL_OFFSETS = {
    "YOLO26n": (10, 8),
    "+ADown": (10, 8),
    "+C3-PD": (10, -22),
    "+CFA": (10, 8),
    "+ADown +C3-PD": (-20, -25),
    "+ADown +CFA": (10, 10),
    "+C3-PD +CFA": (10, -22),
    "KFP-YOLO": (10, 8),
}


# ============================================================
# 5. 计算Pareto前沿
# 横坐标GFLOPs越低越好
# 纵坐标mAP@0.5:0.95越高越好
# ============================================================
def calculate_pareto_frontier(points):
    """
    计算Pareto前沿。

    Parameters
    ----------
    points : list
        每个元素格式为：
        (模型名称, GFLOPs, mAP@0.5:0.95, 标准差)

    Returns
    -------
    frontier_indices : numpy.ndarray
        Pareto前沿点在原始列表中的索引。
    dominated_flags : numpy.ndarray
        每个点是否被其他点支配。
    """

    number_of_points = len(points)
    dominated_flags = np.zeros(number_of_points, dtype=bool)

    for i in range(number_of_points):
        current_x = points[i][1]
        current_y = points[i][2]

        for j in range(number_of_points):
            if i == j:
                continue

            compared_x = points[j][1]
            compared_y = points[j][2]

            # j点支配i点的条件：
            # GFLOPs不高于i，mAP不低于i，
            # 并且至少有一个指标严格优于i
            is_better_or_equal = (
                compared_x <= current_x
                and compared_y >= current_y
            )

            is_strictly_better = (
                compared_x < current_x
                or compared_y > current_y
            )

            if is_better_or_equal and is_strictly_better:
                dominated_flags[i] = True
                break

    frontier_indices = np.where(~dominated_flags)[0]

    # 按照GFLOPs从小到大排序，方便连接前沿线
    frontier_indices = frontier_indices[
        np.argsort(
            [points[index][1] for index in frontier_indices]
        )
    ]

    return frontier_indices, dominated_flags


# ============================================================
# 6. 绘制Pareto图
# ============================================================
def plot_pareto_figure(
    points,
    figure_title,
    output_path,
    label_offsets,
    highlight_model=None,
):
    """
    绘制并保存Pareto前沿图。

    Parameters
    ----------
    points : list
        模型数据。
    figure_title : str
        图标题。
    output_path : pathlib.Path
        JPG输出路径。
    label_offsets : dict
        模型标签偏移位置。
    highlight_model : str or None
        需要突出显示的模型名称。
    """

    frontier_indices, dominated_flags = calculate_pareto_frontier(
        points
    )

    model_names = [item[0] for item in points]
    gflops_values = np.array(
        [item[1] for item in points],
        dtype=float,
    )
    map_values = np.array(
        [item[2] for item in points],
        dtype=float,
    )
    map_errors = np.array(
        [item[3] for item in points],
        dtype=float,
    )

    # --------------------------------------------------------
    # 全局字体设置
    # --------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.unicode_minus": False,
        }
    )

    # --------------------------------------------------------
    # 创建画布
    # --------------------------------------------------------
    figure, axis = plt.subplots(
        figsize=(10.5, 7.5),
        dpi=300,
    )

    # --------------------------------------------------------
    # 绘制全部模型散点和误差棒
    # --------------------------------------------------------
    axis.errorbar(
        gflops_values,
        map_values,
        yerr=map_errors,
        fmt="o",
        markersize=9,
        markeredgewidth=1.0,
        capsize=5,
        capthick=1.4,
        elinewidth=1.4,
        linestyle="none",
        label="Compared models",
        zorder=3,
    )

    # --------------------------------------------------------
    # 绘制Pareto前沿
    # --------------------------------------------------------
    frontier_x = np.array(
        [points[index][1] for index in frontier_indices]
    )
    frontier_y = np.array(
        [points[index][2] for index in frontier_indices]
    )

    axis.plot(
        frontier_x,
        frontier_y,
        marker="s",
        markersize=8,
        markeredgewidth=1.0,
        linewidth=2.2,
        label="Pareto frontier",
        zorder=4,
    )

    # --------------------------------------------------------
    # 突出显示KFP-YOLO或其他指定模型
    # --------------------------------------------------------
    if highlight_model is not None:
        for model_name, gflops, map_value, _ in points:
            if model_name == highlight_model:
                axis.plot(
                    gflops,
                    map_value,
                    marker="*",
                    markersize=21,
                    markeredgewidth=1.1,
                    linestyle="none",
                    label=highlight_model,
                    zorder=6,
                )
                break

    # --------------------------------------------------------
    # 添加模型名称
    # --------------------------------------------------------
    for model_name, gflops, map_value, _ in points:
        offset_x, offset_y = label_offsets.get(
            model_name,
            (10, 10),
        )

        font_weight = (
            "bold"
            if model_name == highlight_model
            else "normal"
        )

        axis.annotate(
            model_name,
            xy=(gflops, map_value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=13,
            fontweight=font_weight,
            ha="left",
            va="center",
            annotation_clip=False,
            zorder=7,
        )

    # --------------------------------------------------------
    # 坐标轴和标题
    # --------------------------------------------------------
    axis.set_xlabel(
        "GFLOPs",
        fontsize=16,
        labelpad=10,
    )

    axis.set_ylabel(
        "mAP@0.5:0.95",
        fontsize=16,
        labelpad=10,
    )

    axis.set_title(
        figure_title,
        fontsize=17,
        pad=15,
    )

    # --------------------------------------------------------
    # 根据数据范围自动设置坐标轴边界
    # --------------------------------------------------------
    x_range = gflops_values.max() - gflops_values.min()
    y_range = map_values.max() - map_values.min()

    if x_range == 0:
        x_range = 1.0

    if y_range == 0:
        y_range = 0.01

    axis.set_xlim(
        gflops_values.min() - x_range * 0.10,
        gflops_values.max() + x_range * 0.13,
    )

    axis.set_ylim(
        map_values.min() - y_range * 0.13,
        map_values.max() + y_range * 0.13,
    )

    # --------------------------------------------------------
    # 网格、刻度和边框
    # --------------------------------------------------------
    axis.grid(
        True,
        linestyle="--",
        linewidth=0.9,
        alpha=0.35,
        zorder=0,
    )

    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1.1,
        labelsize=13,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(1.1)

    # --------------------------------------------------------
    # 图例
    # --------------------------------------------------------
    axis.legend(
        loc="best",
        fontsize=12,
        frameon=True,
        borderpad=0.8,
        handlelength=2.2,
    )

    # --------------------------------------------------------
    # 保存JPG
    # --------------------------------------------------------
    figure.tight_layout()

    figure.savefig(
        output_path,
        format="jpg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        pil_kwargs={
            "quality": 95,
            "subsampling": 0,
        },
    )

    plt.close(figure)

    return frontier_indices, dominated_flags


# ============================================================
# 7. 主程序
# ============================================================
def main():
    mainstream_output = (
        SAVE_DIR / "pareto_mainstream_models.jpg"
    )

    ablation_output = (
        SAVE_DIR / "pareto_ablation_models.jpg"
    )

    # 生成主流模型帕累托图
    mainstream_frontier, _ = plot_pareto_figure(
        points=MAINSTREAM_MODELS,
        figure_title="Pareto Frontier of Mainstream Models",
        output_path=mainstream_output,
        label_offsets=MAINSTREAM_LABEL_OFFSETS,
        highlight_model=None,
    )

    # 生成消融实验帕累托图
    ablation_frontier, _ = plot_pareto_figure(
        points=ABLATION_MODELS,
        figure_title="Pareto Frontier of Ablation Models",
        output_path=ablation_output,
        label_offsets=ABLATION_LABEL_OFFSETS,
        highlight_model="KFP-YOLO",
    )

    # 输出Pareto前沿模型
    print("=" * 65)
    print("主流模型Pareto前沿：")

    for index in mainstream_frontier:
        print(
            f"  {MAINSTREAM_MODELS[index][0]}"
        )

    print("\n消融实验Pareto前沿：")

    for index in ablation_frontier:
        print(
            f"  {ABLATION_MODELS[index][0]}"
        )

    print("\n图片生成完成：")
    print(f"  {mainstream_output}")
    print(f"  {ablation_output}")
    print("=" * 65)


if __name__ == "__main__":
    main()