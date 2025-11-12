"""
AutoML 基类定义。

该模块包含 HappyMath AutoML 框架的核心基类，实现数据加载、实验初始化、
模型存储与评估等通用能力，后续所有任务类都将继承此基类。
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


DataLike = Union[str, pd.DataFrame, np.ndarray]
TargetLike = Union[str, int, None]


@dataclass
class StoredModel:
    """用于存储模型相关信息的简单数据结构。"""

    model: Any
    metrics: Dict[str, Any]
    name: str
    extra: Dict[str, Any]
    timestamp: pd.Timestamp


class AutoMLBase:
    """
    HappyMath AutoML 基类。

    负责封装 PyCaret 的实验生命周期，提供统一的数据加载、指标判断以及模型管理能力。
    """

    primary_metric: Optional[str] = None

    def __init__(
        self,
        data: DataLike,
        target: TargetLike = None,
        test_data: Optional[DataLike] = None,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        # 数据加载与校验
        self.data, normalized_target = self._load_data(data, target)
        self.target = normalized_target
        self._validate_data(self.data, self.target)

        # 测试数据处理
        if test_data is not None:
            self.test_data, _ = self._load_data(test_data, target=None)
        else:
            self.test_data = None

        # 公共属性初始化
        self.primary_metric = primary_metric or self.primary_metric
        self.setup_kwargs = setup_kwargs
        self.experiment = None
        self.models: Dict[str, StoredModel] = {}
        self.current_model: Optional[Any] = None
        self.is_setup = False
        self.results = None
        self.verbose = getattr(self, "verbose", False)

        # 自动执行实验初始化
        self._setup_experiment(**setup_kwargs)

    # ------------------------------------------------------------------
    # 数据相关工具
    # ------------------------------------------------------------------
    def _load_data(
        self,
        data: DataLike,
        target: TargetLike,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        将多种数据格式统一转换为 DataFrame，并处理目标列。
        """
        if isinstance(data, str):
            if data.lower().endswith(".csv"):
                df = pd.read_csv(data)
            elif data.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(data)
            else:
                raise ValueError(f"不支持的文件格式: {data}")
        elif isinstance(data, pd.Series):
            df = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            # 数组模式下目标列在转换函数中处理
            return self._convert_array_to_frame(data, target)
        else:
            raise TypeError("data 参数必须是文件路径、DataFrame 或 NumPy 数组")

        target_name: Optional[str]
        if target is None:
            target_name = None
        elif isinstance(target, str):
            if target not in df.columns:
                raise ValueError(f"目标列 '{target}' 不存在于数据集中")
            target_name = target
        elif isinstance(target, int):
            try:
                target_name = df.columns[target]
            except IndexError as exc:
                raise ValueError(f"目标列索引 {target} 超出范围") from exc
        else:
            raise TypeError("target 参数必须是列名、列索引或 None")

        return df, target_name

    def _convert_array_to_frame(
        self,
        array: np.ndarray,
        target: TargetLike,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """将 NumPy 数组转换为 DataFrame，并根据索引设置目标列。"""
        if array.ndim != 2:
            raise ValueError("仅支持二维数组作为数据输入")

        columns = [f"feature_{idx}" for idx in range(array.shape[1])]
        df = pd.DataFrame(array, columns=columns)

        if target is None:
            return df, None

        if not isinstance(target, int):
            raise TypeError("NumPy 数组模式下 target 必须为整数索引")

        try:
            target_column = columns[target]
        except IndexError as exc:
            raise ValueError(f"目标列索引 {target} 超出范围") from exc

        df.rename(columns={target_column: "target"}, inplace=True)
        return df, "target"

    def _validate_data(self, data: pd.DataFrame, target: Optional[str]) -> None:
        """基础数据校验，确保目标列存在并且无重复列名。"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("内部数据必须是 pandas.DataFrame 类型")

        if data.columns.duplicated().any():
            raise ValueError("数据集中存在重复的列名，请先进行处理")

        if target is not None and target not in data.columns:
            raise ValueError("指定的目标列未在数据中找到")

    # ------------------------------------------------------------------
    # 指标相关工具
    # ------------------------------------------------------------------
    def _get_metric_direction(self, metric: str) -> str:
        """根据常见指标名称判断优化方向。"""
        lower_better = {
            "MAE",
            "MSE",
            "RMSE",
            "RMSLE",
            "MAPE",
            "MedAE",
            "MASE",
            "RMSSE",
            "SMAPE",
            "Log Loss",
            "FNR",
            "FPR",
        }
        higher_better = {
            "Accuracy",
            "AUC",
            "Recall",
            "Prec.",
            "F1",
            "Kappa",
            "MCC",
            "R2",
            "TPR",
            "TNR",
            "PPV",
            "NPV",
            "Silhouette",
        }

        if metric in lower_better:
            return "lower_better"
        if metric in higher_better:
            return "higher_better"

        print(f"警告: 未知指标 '{metric}'，默认按越大越好处理")
        return "higher_better"

    def _is_better_score(self, new_score: float, current_best: Optional[float]) -> bool:
        """根据指标方向判断新分数是否更优。"""
        if current_best is None:
            return True

        direction = self._get_metric_direction(self.primary_metric)
        if direction == "higher_better":
            return new_score > current_best
        return new_score < current_best

    def _safe_get_model_name(self, model: Any) -> str:
        """获取模型的可读名称，优先使用实验内部提供的方法。"""
        if self.experiment and hasattr(self.experiment, "_get_model_name"):
            try:
                return self.experiment._get_model_name(model)
            except Exception:
                pass
        return getattr(model, "__class__", model).__name__

    def _extract_metrics_from_results(
        self,
        results: Optional[Any],
        model_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """从 PyCaret 的 pull 结果中提取指标信息。"""
        if results is None:
            return {}

        if isinstance(results, dict):
            return dict(results)

        if isinstance(results, pd.Series):
            return results.to_dict()

        if isinstance(results, pd.DataFrame):
            df = results.copy()

            if "Model" in df.columns and model_label:
                matched = df[df["Model"] == model_label]
                if not matched.empty:
                    return matched.iloc[0].to_dict()

            index = df.index
            if isinstance(index, pd.Index):
                for key in ("Mean", "Holdout", "Score"):
                    if key in index:
                        row = df.loc[key]
                        return row.to_dict()

            if df.shape[0] > 0:
                return df.iloc[-1].to_dict()

        return {}

    def _filter_kwargs_for(self, func: Any, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """根据函数签名过滤不被支持的参数。"""
        params = inspect.signature(func).parameters
        return {key: value for key, value in base_kwargs.items() if key in params}

    def _build_plot_customization_context(
        self,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        legend_title: Optional[str],
        legend_labels: Optional[List[str]],
        font_sizes: Optional[Dict[str, Union[int, float]]],
    ):
        """根据自定义文本或字体控制构造绘图上下文，未提供时返回空上下文。"""
        has_custom_text = any([title, xlabel, ylabel, legend_title, legend_labels])
        has_font_customization = font_sizes is not None and len(font_sizes) > 0
        if not (has_custom_text or has_font_customization):
            return nullcontext()
        return self._override_matplotlib_labels(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_title=legend_title,
            legend_labels=legend_labels,
            font_sizes=font_sizes,
        )

    @contextmanager
    def _override_matplotlib_labels(
        self,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        legend_title: Optional[str],
        legend_labels: Optional[List[str]],
        font_sizes: Optional[Dict[str, Union[int, float]]],
    ):
        """临时重写 Matplotlib 的标题、坐标轴与图例设置，确保自定义参数及字体生效。"""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib import rcParams
        import warnings

        patches: List[Tuple[Any, str, Any]] = []
        fonts = font_sizes or {}

        title_size = fonts.get("title")
        xlabel_size = fonts.get("xlabel")
        ylabel_size = fonts.get("ylabel")
        legend_title_size = fonts.get("legend_title")
        legend_label_size = fonts.get("legend_label")
        xtick_size = fonts.get("xtick")
        ytick_size = fonts.get("ytick")

        original_rc: Dict[str, Any] = {}

        def add_patch(target: Any, attr: str, new_func: Any) -> None:
            original = getattr(target, attr)
            setattr(target, attr, new_func)
            patches.append((target, attr, original))

        if title is not None or title_size is not None:
            original_set_title = Axes.set_title

            def set_title_override(self, _text: str, *args: Any, **kwargs: Any):
                kwargs = dict(kwargs)
                if title_size is not None:
                    kwargs["fontsize"] = title_size
                actual = title if title is not None else _text
                return original_set_title(self, actual, *args, **kwargs)

            add_patch(Axes, "set_title", set_title_override)

            original_plt_title = plt.title

            def plt_title_override(_text: str = "", *args: Any, **kwargs: Any):
                kwargs = dict(kwargs)
                if title_size is not None:
                    kwargs["fontsize"] = title_size
                actual = title if title is not None else _text
                return original_plt_title(actual, *args, **kwargs)

            add_patch(plt, "title", plt_title_override)

        if xlabel is not None or xlabel_size is not None:
            original_set_xlabel = Axes.set_xlabel

            def set_xlabel_override(self, _text: str, *args: Any, **kwargs: Any):
                kwargs = dict(kwargs)
                if xlabel_size is not None:
                    kwargs["fontsize"] = xlabel_size
                actual = xlabel if xlabel is not None else _text
                return original_set_xlabel(self, actual, *args, **kwargs)

            add_patch(Axes, "set_xlabel", set_xlabel_override)

            original_plt_xlabel = plt.xlabel

            def plt_xlabel_override(_text: str = "", *args: Any, **kwargs: Any):
                kwargs = dict(kwargs)
                if xlabel_size is not None:
                    kwargs["fontsize"] = xlabel_size
                actual = xlabel if xlabel is not None else _text
                return original_plt_xlabel(actual, *args, **kwargs)

            add_patch(plt, "xlabel", plt_xlabel_override)

        if ylabel is not None or ylabel_size is not None:
            original_set_ylabel = Axes.set_ylabel

            def set_ylabel_override(self, _text: str, *args: Any, **kwargs: Any):
                kwargs = dict(kwargs)
                if ylabel_size is not None:
                    kwargs["fontsize"] = ylabel_size
                actual = ylabel if ylabel is not None else _text
                return original_set_ylabel(self, actual, *args, **kwargs)

            add_patch(Axes, "set_ylabel", set_ylabel_override)

            original_plt_ylabel = plt.ylabel

            def plt_ylabel_override(_text: str = "", *args: Any, **kwargs: Any):
                kwargs = dict(kwargs)
                if ylabel_size is not None:
                    kwargs["fontsize"] = ylabel_size
                actual = ylabel if ylabel is not None else _text
                return original_plt_ylabel(actual, *args, **kwargs)

            add_patch(plt, "ylabel", plt_ylabel_override)

        if any([legend_title, legend_labels, legend_title_size, legend_label_size]):
            original_axes_legend = Axes.legend

            def legend_override(self, *args: Any, **kwargs: Any):
                legend = original_axes_legend(self, *args, **kwargs)
                if legend is not None:
                    if legend_title is not None:
                        legend.set_title(legend_title)
                    if legend_labels is not None:
                        texts = legend.get_texts()
                        if len(legend_labels) != len(texts):
                            warnings.warn(
                                "legend_labels 长度与实际图例数量不匹配，将按最小长度截断处理",
                                UserWarning,
                            )
                        for text_obj, label in zip(texts, legend_labels):
                            text_obj.set_text(label)
                    if legend_title_size is not None and legend.get_title() is not None:
                        legend.get_title().set_fontsize(legend_title_size)
                    if legend_label_size is not None:
                        for text_obj in legend.get_texts():
                            text_obj.set_fontsize(legend_label_size)
                return legend

            add_patch(Axes, "legend", legend_override)

            original_plt_legend = plt.legend

            def plt_legend_override(*args: Any, **kwargs: Any):
                legend = original_plt_legend(*args, **kwargs)
                if legend is not None:
                    if legend_title is not None:
                        legend.set_title(legend_title)
                    if legend_labels is not None:
                        texts = legend.get_texts()
                        if len(legend_labels) != len(texts):
                            warnings.warn(
                                "legend_labels 长度与实际图例数量不匹配，将按最小长度截断处理",
                                UserWarning,
                            )
                        for text_obj, label in zip(texts, legend_labels):
                            text_obj.set_text(label)
                    if legend_title_size is not None and legend.get_title() is not None:
                        legend.get_title().set_fontsize(legend_title_size)
                    if legend_label_size is not None:
                        for text_obj in legend.get_texts():
                            text_obj.set_fontsize(legend_label_size)
                return legend

            add_patch(plt, "legend", plt_legend_override)

        if xtick_size is not None:
            original_rc["xtick.labelsize"] = rcParams.get("xtick.labelsize")
            rcParams["xtick.labelsize"] = xtick_size
        if ytick_size is not None:
            original_rc["ytick.labelsize"] = rcParams.get("ytick.labelsize")
            rcParams["ytick.labelsize"] = ytick_size
        if "tick" in fonts and fonts["tick"] is not None:
            value = fonts["tick"]
            original_rc["xtick.labelsize"] = original_rc.get(
                "xtick.labelsize", rcParams.get("xtick.labelsize")
            )
            original_rc["ytick.labelsize"] = original_rc.get(
                "ytick.labelsize", rcParams.get("ytick.labelsize")
            )
            rcParams["xtick.labelsize"] = value
            rcParams["ytick.labelsize"] = value

        try:
            yield
        finally:
            for target, attr, original in reversed(patches):
                setattr(target, attr, original)
            for key, value in original_rc.items():
                rcParams[key] = value

    # ------------------------------------------------------------------
    # 模型存储与管理
    # ------------------------------------------------------------------
    def _store_model_with_metrics(
        self,
        model: Any,
        model_name: str,
        results_df: Optional[Any] = None,
        model_label: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录模型对象及对应指标，方便后续查询。"""
        metrics: Dict[str, Any] = {}

        if results_df is None:
            try:
                results_df = self.experiment.pull()
            except Exception:
                results_df = None

        if results_df is not None:
            metrics = self._extract_metrics_from_results(results_df, model_label)

        info = StoredModel(
            model=model,
            metrics=metrics,
            name=model_name,
            extra=additional_info or {},
            timestamp=pd.Timestamp.now(),
        )
        self.models[model_name] = info

    def get_best_model(self) -> Tuple[Any, Dict[str, Any]]:
        """基于 primary_metric 返回性能最佳的模型。"""
        if not self.models:
            raise ValueError("没有可比较的模型，请先创建或比较模型")

        best_name = None
        best_info: Optional[StoredModel] = None
        best_score: Optional[float] = None

        for name, info in self.models.items():
            score = info.metrics.get(self.primary_metric)
            if score is None:
                continue
            if self._is_better_score(score, best_score):
                best_score = score
                best_name = name
                best_info = info

        if best_info is None:
            fallback = list(self.models.values())[-1]
            print(
                f"警告: 未找到包含主指标 '{self.primary_metric}' 的模型，将返回最近的模型 '{fallback.name}'"
            )
            self.current_model = fallback.model
            return fallback.model, fallback.metrics

        self.current_model = best_info.model
        return best_info.model, best_info.metrics

    # ------------------------------------------------------------------
    # 训练与评估核心接口
    # ------------------------------------------------------------------
    def compare(
        self,
        include: Optional[List[Any]] = None,
        exclude: Optional[List[str]] = None,
        sort: Optional[str] = None,
        budget_time: Optional[float] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """比较已支持模型并选取表现最优的模型。"""
        self._ensure_setup()
        verbose_flag = self.verbose if verbose is None else verbose
        metric = sort or self.primary_metric

        best_model = self.experiment.compare_models(
            include=include,
            exclude=exclude,
            sort=metric,
            budget_time=budget_time,
            verbose=verbose_flag,
            n_select=1,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(best_model)
        self._store_model_with_metrics(
            best_model,
            model_name="compare_best",
            results_df=results,
            model_label=label,
        )
        self.current_model = best_model
        return best_model

    def create(
        self,
        estimator: Any,
        return_train_score: bool = False,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """基于指定算法创建模型。"""
        self._ensure_setup()
        verbose_flag = self.verbose if verbose is None else verbose

        model = self.experiment.create_model(
            estimator=estimator,
            return_train_score=return_train_score,
            verbose=verbose_flag,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(model)
        self._store_model_with_metrics(
            model,
            model_name=f"create_{label}",
            results_df=results,
            model_label=label,
        )
        if self.current_model is None:
            self.current_model = model
        return model

    def tune(
        self,
        estimator: Optional[Any] = None,
        n_iter: int = 10,
        custom_grid: Optional[Dict[str, List[Any]]] = None,
        optimize: Optional[str] = None,
        verbose: Optional[bool] = None,
        tuner_verbose: Union[int, bool] = True,
        **kwargs: Any,
    ) -> Any:
        """对当前模型或指定模型执行超参数搜索。"""
        self._ensure_setup()

        base_model = estimator or self.current_model
        if base_model is None:
            raise ValueError("没有可调优的模型，请先通过 compare 或 create 生成模型")

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        tuned_model = self.experiment.tune_model(
            estimator=base_model,
            n_iter=n_iter,
            custom_grid=custom_grid,
            optimize=metric,
            verbose=verbose_flag,
            tuner_verbose=tuner_verbose,
            choose_better=True,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(tuned_model)
        self._store_model_with_metrics(
            tuned_model,
            model_name="tuned",
            results_df=results,
            model_label=label,
        )
        self.current_model = tuned_model
        return tuned_model

    def ensemble(
        self,
        estimator: Optional[Any] = None,
        method: str = "Bagging",
        n_estimators: int = 10,
        optimize: Optional[str] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """对模型执行 Bagging 或 Boosting 集成。"""
        self._ensure_setup()
        base_model = estimator or self.current_model
        if base_model is None:
            raise ValueError("没有可用于集成的模型")

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        ensemble_call = {
            "estimator": base_model,
            "method": method,
            "n_estimators": n_estimators,
            "optimize": metric,
            "choose_better": True,
            "verbose": verbose_flag,
        }
        ensemble_call.update(kwargs)
        filtered = self._filter_kwargs_for(self.experiment.ensemble_model, ensemble_call)

        ensemble_model = self.experiment.ensemble_model(**filtered)

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(ensemble_model)
        self._store_model_with_metrics(
            ensemble_model,
            model_name=f"ensemble_{method.lower()}",
            results_df=results,
            model_label=label,
            additional_info={
                "ensemble_method": method,
                "n_estimators": n_estimators,
            },
        )
        self.current_model = ensemble_model
        return ensemble_model

    def blend(
        self,
        estimator_list: Optional[List[Any]] = None,
        optimize: Optional[str] = None,
        method: str = "auto",
        weights: Optional[List[float]] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """对多个模型执行投票/平均混合。"""
        self._ensure_setup()

        if estimator_list is None:
            if len(self.models) < 2:
                raise ValueError("混合模型前至少需要准备两个基础模型")
            estimator_list = [info.model for info in self.models.values()]

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        blend_call = {
            "estimator_list": estimator_list,
            "method": method,
            "weights": weights,
            "optimize": metric,
            "choose_better": True,
            "verbose": verbose_flag,
        }
        blend_call.update(kwargs)
        filtered = self._filter_kwargs_for(self.experiment.blend_models, blend_call)

        blended = self.experiment.blend_models(**filtered)

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(blended)
        self._store_model_with_metrics(
            blended,
            model_name="blended",
            results_df=results,
            model_label=label,
            additional_info={
                "blend_method": method,
                "n_models": len(estimator_list),
            },
        )
        self.current_model = blended
        return blended

    def stack(
        self,
        estimator_list: Optional[List[Any]] = None,
        meta_model: Optional[Any] = None,
        meta_model_fold: Optional[int] = 5,
        method: str = "auto",
        restack: bool = False,
        optimize: Optional[str] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """执行模型堆叠，构建两层结构的集成模型。"""
        self._ensure_setup()

        if estimator_list is None:
            if len(self.models) < 2:
                raise ValueError("堆叠模型前至少需要准备两个基础模型")
            estimator_list = [info.model for info in self.models.values()]

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        stack_call = {
            "estimator_list": estimator_list,
            "meta_model": meta_model,
            "meta_model_fold": meta_model_fold,
            "method": method,
            "restack": restack,
            "optimize": metric,
            "choose_better": True,
            "verbose": verbose_flag,
        }
        stack_call.update(kwargs)
        filtered = self._filter_kwargs_for(self.experiment.stack_models, stack_call)

        stacked = self.experiment.stack_models(**filtered)

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(stacked)
        self._store_model_with_metrics(
            stacked,
            model_name="stacked",
            results_df=results,
            model_label=label,
            additional_info={
                "meta_model": self._safe_get_model_name(meta_model)
                if meta_model
                else "LogisticRegression",
                "n_base_models": len(estimator_list),
            },
        )
        self.current_model = stacked
        return stacked

    def plot(
        self,
        estimator: Optional[Any] = None,
        plot: str = "auc",
        scale: float = 1.0,
        save: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        plot_kwargs: Optional[Dict[str, Any]] = None,
        font_sizes: Optional[Dict[str, Union[int, float]]] = None,
        verbose: Optional[bool] = None,
    ) -> Optional[str]:
        """调用 PyCaret 可视化能力，并提供友好的默认标题。"""
        import warnings

        self._ensure_setup()
        estimator = estimator or self.current_model
        if estimator is None:
            raise ValueError("没有可用于绘图的模型")

        verbose_flag = self.verbose if verbose is None else verbose
        effective_title = title or self._get_default_plot_title(plot, estimator)

        final_kwargs = dict(plot_kwargs or {})
        final_kwargs.setdefault("figsize", figsize)

        if plot in {"auc", "pr", "confusion_matrix", "error", "learning", "vc", "residuals"}:
            if effective_title:
                final_kwargs.setdefault("title", effective_title)
            if xlabel:
                final_kwargs.setdefault("xlabel", xlabel)
            if ylabel:
                final_kwargs.setdefault("ylabel", ylabel)
            if legend_title:
                final_kwargs.setdefault("legend_title", legend_title)
        elif plot in {"feature", "feature_all"}:
            if effective_title:
                final_kwargs.setdefault("title", effective_title)
            if xlabel:
                final_kwargs.setdefault("xaxis_title", xlabel)
            if ylabel:
                final_kwargs.setdefault("yaxis_title", ylabel)
        else:
            if effective_title:
                final_kwargs.setdefault("title", effective_title)

        plot_call = {
            "estimator": estimator,
            "plot": plot,
            "scale": scale,
            "save": save,
            "verbose": verbose_flag,
            "plot_kwargs": final_kwargs,
            "fig_kwargs": final_kwargs,
        }
        filtered_call = self._filter_kwargs_for(self.experiment.plot_model, plot_call)

        customization_context = self._build_plot_customization_context(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_title=legend_title,
            legend_labels=legend_labels,
            font_sizes=font_sizes,
        )

        with customization_context:
            try:
                return self.experiment.plot_model(**filtered_call)
            except Exception as exc:
                warnings.warn(f"使用自定义参数绘图失败，将回退到默认绘图。错误信息: {exc}")
                fallback = {
                    "estimator": estimator,
                    "plot": plot,
                    "scale": scale,
                    "save": save,
                    "verbose": verbose_flag,
                }
                fallback_filtered = self._filter_kwargs_for(
                    self.experiment.plot_model, fallback
                )
                return self.experiment.plot_model(**fallback_filtered)

    def _get_default_plot_title(self, plot: str, estimator: Any) -> str:
        """根据图表类型生成默认标题。"""
        model_name = self._safe_get_model_name(estimator)
        mapping = {
            "auc": f"{model_name} - ROC曲线",
            "pr": f"{model_name} - 精准率-召回率曲线",
            "confusion_matrix": f"{model_name} - 混淆矩阵",
            "error": f"{model_name} - 预测误差",
            "feature": f"{model_name} - 特征重要性",
            "feature_all": f"{model_name} - 全量特征重要性",
            "learning": f"{model_name} - 学习曲线",
            "vc": f"{model_name} - 验证曲线",
            "residuals": f"{model_name} - 残差图",
            "cooks": f"{model_name} - Cook距离图",
        }
        return mapping.get(plot, f"{model_name} - {plot.upper()} 可视化")

    def predict(
        self,
        estimator: Optional[Any] = None,
        data: Optional[pd.DataFrame] = None,
        raw_score: bool = False,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """使用指定模型进行预测，默认利用测试集。"""
        self._ensure_setup()
        predictor = estimator or self.current_model
        if predictor is None:
            raise ValueError("没有可用于预测的模型")

        data_to_use = data if data is not None else self.test_data
        verbose_flag = self.verbose if verbose is None else verbose

        predict_fn = self.experiment.predict_model
        signature = inspect.signature(predict_fn)
        call_kwargs = {
            "estimator": predictor,
            "data": data_to_use,
            "verbose": verbose_flag,
            **kwargs,
        }
        if "raw_score" in signature.parameters:
            call_kwargs["raw_score"] = raw_score
        elif raw_score:
            print("警告: 当前任务不支持 raw_score 参数，该设置将被忽略")

        return predict_fn(**call_kwargs)

    def finalize(self, estimator: Optional[Any] = None) -> Any:
        """在全量数据上重新训练模型。"""
        self._ensure_setup()
        target_model = estimator or self.current_model
        if target_model is None:
            raise ValueError("没有可最终化的模型")

        final_model = self.experiment.finalize_model(target_model)
        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(final_model)
        self._store_model_with_metrics(
            final_model,
            model_name="final",
            results_df=results,
            model_label=label,
        )
        self.current_model = final_model
        return final_model

    def evaluate(self, estimator: Optional[Any] = None) -> None:
        """启动交互式评估界面。"""
        self._ensure_setup()
        target_model = estimator or self.current_model
        if target_model is None:
            raise ValueError("没有可评估的模型")
        self.experiment.evaluate_model(target_model)

    # ------------------------------------------------------------------
    # 对外辅助接口
    # ------------------------------------------------------------------
    def get_models(self) -> Iterable[str]:
        """返回已存储模型的名称列表。"""
        return list(self.models.keys())

    def get_metrics(self) -> Any:
        """返回实验支持的评价指标列表。"""
        self._ensure_setup()
        if hasattr(self.experiment, "get_metrics"):
            return self.experiment.get_metrics()
        if hasattr(self.experiment, "_all_metrics"):
            containers = getattr(self.experiment, "_all_metrics")
            rows = []
            for key, container in containers.items():
                display = getattr(container, "display_name", key)
                rows.append({"ID": key, "Display Name": display})
            return pd.DataFrame(rows)
        raise AttributeError("当前实验不支持 get_metrics 接口")

    def get_results(self) -> pd.DataFrame:
        """获取最近一次操作的结果表。"""
        self._ensure_setup()
        if self.results is not None:
            return self.results
        pulled = self.experiment.pull()
        if pulled is None:
            raise ValueError("当前没有可用的结果表")
        return pulled

    def get_leaderboard(self) -> pd.DataFrame:
        """获取模型排行榜。"""
        self._ensure_setup()
        if hasattr(self.experiment, "get_leaderboard"):
            return self.experiment.get_leaderboard()
        if self.results is not None and not self.results.empty:
            return self.results
        pulled = self.experiment.pull()
        if pulled is not None and not pulled.empty:
            return pulled
        raise ValueError("当前实验暂无排行榜数据")

    def save(self, model_name: str, model: Optional[Any] = None) -> None:
        """保存模型到本地。"""
        self._ensure_setup()
        model_to_save = model or self.current_model
        if model_to_save is None:
            raise ValueError("没有可用模型可以保存")
        self.experiment.save_model(model_to_save, model_name)

    def load(self, model_name: str) -> Any:
        """从本地加载模型。"""
        self._ensure_setup()
        return self.experiment.load_model(model_name)

    def get_config(self, key: Optional[str] = None) -> Any:
        """读取实验配置。"""
        self._ensure_setup()
        return self.experiment.get_config(key)

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------
    def _ensure_setup(self) -> None:
        """确保实验已经完成 setup。"""
        if not self.is_setup:
            raise RuntimeError("请先完成实验初始化（setup）")

    def _setup_experiment(self, **kwargs: Any) -> None:
        """基类不实现具体 setup，需在子类中自定义。"""
        raise NotImplementedError("子类必须实现 _setup_experiment 方法")
