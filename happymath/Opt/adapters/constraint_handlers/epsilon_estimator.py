"""
Epsilon估计器

用于估计严格不等式约束的epsilon值。
从OptBase._estimate_epsilon迁移而来。
"""

import numpy as np


class EpsilonEstimator:
    """Epsilon估计器"""

    @staticmethod
    def estimate(con_lambda, con_symbols, symbol_to_index, xl, xu,
                 discrete_vars=None, discrete_value_mapping=None, default_epsilon=1e-6):
        """
        估计约束表达式的数值范围，返回合适的epsilon值

        Args:
            con_lambda: 约束的lambda函数
            con_symbols: 约束涉及的符号列表
            symbol_to_index: 符号到索引的映射
            xl: 变量下界数组
            xu: 变量上界数组
            discrete_vars: 离散变量字典（可选）
            discrete_value_mapping: 离散变量值映射（可选）
            default_epsilon: 默认epsilon值

        Returns:
            float: 适合该约束的epsilon值
        """
        # 获取约束涉及的变量索引
        var_indices = [symbol_to_index[sym] for sym in con_symbols]

        # 生成采样点来估计约束表达式的数值范围
        n_samples = min(100, 5 ** len(con_symbols))  # 避免维度爆炸
        values = []

        try:
            # 生成随机采样点
            for _ in range(n_samples):
                sample = []
                for i, sym in enumerate(con_symbols):
                    idx = var_indices[i]

                    # 处理离散变量
                    if discrete_vars and idx in discrete_vars:
                        # 从离散值中随机选择
                        if discrete_value_mapping and idx in discrete_value_mapping:
                            sample.append(np.random.choice(list(discrete_value_mapping[idx].values())))
                        else:
                            sample.append(np.random.choice(discrete_vars[idx]))
                    else:
                        # 处理连续变量
                        lower = xl[idx] if np.isfinite(xl[idx]) else -10.0
                        upper = xu[idx] if np.isfinite(xu[idx]) else 10.0

                        # 确保边界合理
                        if lower >= upper:
                            lower, upper = -1, 1

                        sample.append(np.random.uniform(lower, upper))

                # 计算约束表达式的值
                try:
                    val = con_lambda(*sample)
                    if np.isfinite(val):
                        values.append(abs(val))
                except:
                    pass

            if values:
                # 基于值的范围计算epsilon
                scale = np.median(values)

                if scale < 1e-10:
                    scale = max(values) if values else 1.0

                # epsilon应该是scale的一个小比例
                epsilon = max(1e-12, min(1e-2, scale * 1e-6))
                return epsilon
            else:
                return default_epsilon

        except Exception:
            return default_epsilon
