import numpy as np

def map_values_to_domain(
    values: np.ndarray,
    idomain: np.ndarray,
    fill_value: float | int
):
    # 确保输入是 NumPy 数组，以获得最佳性能和功能
    idomain = np.asarray(idomain)
    values = np.asarray(values)

    # 1. 计算 idomain 中活跃单元格的总数
    num_active_cells = np.sum(idomain == 1)

    # 2. 验证 values 数组的维度是否正确
    if values.shape[-1] != num_active_cells:
        raise ValueError(
            f"值数组(values)的最后一个维度 ({values.shape[-1]}) "
            f"与 idomain 中的活跃单元格数量 ({num_active_cells}) 不匹配。"
        )

    # 3. 创建一个布尔掩码 (boolean mask)，True 代表活跃位置
    # 这个掩码的形状与 idomain 完全相同
    active_mask = (idomain == 1)

    # 4. 处理两种情况：values 是单个时间步（1D）还是多个时间步（>=2D）
    
    if values.ndim == 1:
        # --- 情况一：values 是一维数组 (单个结果集) ---
        
        # 创建一个与 idomain 形状相同、用 fill_value 填充的新数组
        # 使用 values.dtype 来确保结果数组的数据类型可以容纳 values 的值
        # (例如，如果 values 是浮点数，即使 fill_value 是整数，结果也应是浮点数)
        result = np.full(idomain.shape, fill_value, dtype=values.dtype)
        
        # 使用布尔掩码将 values 的值赋给 result 中的对应位置
        result[active_mask] = values
        
    else: # values.ndim > 1
        # --- 情况二：values 是多维数组 (例如，带时间步) ---
        
        # 计算输出数组的形状：(values 的前缀维度..., idomain 的维度...)
        # 例如: values.shape=(121, 23569), idomain.shape=(100, 400)
        # -> output_shape=(121, 100, 400)
        output_shape = (*values.shape[:-1], *idomain.shape)
        
        # 创建一个目标形状的、用 fill_value 填充的新数组
        result = np.full(output_shape, fill_value, dtype=values.dtype)
        
        # 使用布尔掩码进行高级索引赋值
        # NumPy 会自动地将掩码应用到 result 的最后几个维度上
        # result[..., active_mask] 的形状会是 (121, 23569)
        # 这正好与 values 的形状匹配，可以直接赋值
        result[..., active_mask] = values

    return result