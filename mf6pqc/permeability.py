# updater/permeability.py
import numpy as np
from abc import ABC, abstractmethod


class BasePermeabilityUpdater(ABC):
    """渗透率更新的抽象基类"""
    @abstractmethod
    def update(self, K_old: np.ndarray, porosity_old: np.ndarray, porosity_new: np.ndarray) -> np.ndarray:
        """计算新的渗透率分布"""
        raise NotImplementedError

class KozenyCarmanUpdater(BasePermeabilityUpdater):
    """
    基于 Kozeny-Carman 公式的渗透率更新方法
    K_new = K_old * (phi_new^3/(1-phi_new)^2) / (phi_old^3/(1-phi_old)^2)
    """

    def update(self, K_old: np.ndarray, porosity_old: np.ndarray, porosity_new: np.ndarray) -> np.ndarray:
        eps = 1e-20 # 避免除零
        porosity_old = np.maximum(porosity_old, eps)
        porosity_new = np.maximum(porosity_new, eps)

        term_new = porosity_new**3 / ((1.0 - porosity_new)**2 + eps)
        term_old = porosity_old**3 / ((1.0 - porosity_old)**2 + eps)
        K_new = K_old * (term_new / (term_old + eps))

        return K_new

class PowerLawUpdater(BasePermeabilityUpdater):
    """幂律模型：K_new = K_old * (phi_new / phi_old)**n"""
    def __init__(self, n: float = 2.0):
        self.n = n

    def update(self, K_old: np.ndarray, porosity_old: np.ndarray, porosity_new: np.ndarray) -> np.ndarray:
        eps = 1e-20
        ratio = (porosity_new + eps) / (porosity_old + eps)
        return K_old * np.power(ratio, self.n)

# todo 
class MLPermeabilityUpdater(BasePermeabilityUpdater):
    """示例：用机器学习模型预测渗透率"""
    def __init__(self, model):
        self.model = model

    def update(self, K_old: np.ndarray, porosity_old: np.ndarray, porosity_new: np.ndarray) -> np.ndarray:
        # 这里假设 model.predict 返回与 porosity_new 相同形状的渗透率数组
        return self.model.predict(porosity_new)