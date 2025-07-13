import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ! 基尼系数 ! #

def gini_coefficient(k_field):
    """
    计算一个2D渗透场的基尼系数。
    """
    # 确保输入为正值，并展平为1D数组
    x = k_field[k_field > 0].flatten()
    if x.size == 0:
        return 0.0
        
    # 按从小到大排序
    x_sorted = np.sort(x)
    n = len(x_sorted)
    
    # 计算累积值
    cumx = np.cumsum(x_sorted, dtype=float)
    
    # 核心公式
    # Gini = ( (n + 1) / n - 2 * sum(cumx) / (n * sum(x)) )
    sum_x = cumx[-1]
    if sum_x == 0: # 避免除以零
        return 0.0
        
    gini_val = (n + 1 - 2 * np.sum(cumx) / sum_x) / n
    return gini_val

def calculate_gini_timeseries(K_timeseries):
    """
    计算渗透场时间序列的基尼系数。
    """
    num_times = K_timeseries.shape[0]
    gini_series = np.zeros(num_times)
    for i in range(num_times):
        gini_series[i] = gini_coefficient(K_timeseries[i, :, :])
    return gini_series

# ! 欧拉系数法, 已放弃

from skimage.measure import euler_number
def calc_connectivity_euler(K_field, percentile=50):
    """
    使用欧拉示性数评估高渗透区的连通性.
    
    参数:
        K_field (np.ndarray): 2D渗透系数场.
        percentile (int): 用于二值化的渗透率百分位阈值 (0-100).
                         例如，75代表分析渗透率最高的25%区域.
        
    返回:
        int: 欧拉示性数.
    """
    threshold = np.percentile(K_field, percentile)
    binary_image = K_field > threshold
    # connectivity=2 表示8连通
    return euler_number(binary_image, connectivity=2)

# ! 