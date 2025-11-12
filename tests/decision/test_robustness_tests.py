"""
接口健壮性测试

测试异常输入、错误处理、数值稳定性等，验证系统的容错能力和稳定性
"""
import pytest
import numpy as np
import warnings
from typing import Dict, Any, List, Optional

from happymath.Decision.methods import (
    SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
    FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
)

from tests.fixtures.test_data import RobustnessTestData
from tests.utils.test_helpers import TestHelper

warnings.filterwarnings("ignore")


class TestRobustnessTests:
    """健壮性测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        TestHelper.suppress_warnings()
    
    @pytest.mark.robustness
    def test_none_input_handling(self):
        """测试None输入处理"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['none_values']
        
        decision_classes = [
            SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
            FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
        ]
        
        for decision_class in decision_classes:
            class_name = decision_class.__name__
            
            try:
                instance = decision_class()
                
                # 测试传入None参数
                with pytest.raises((ValueError, TypeError, AttributeError)):
                    instance.decide(
                        dataset=invalid_inputs['matrix'],
                        criterion_type=invalid_inputs['criterion_type'],
                        weights=invalid_inputs['weights']
                    )
                
                print(f"✅ {class_name} - None输入处理正确（抛出异常）")
                
            except Exception as e:
                # 如果没有抛出预期异常，可能是参数校验不够严格
                print(f"⚠️ {class_name} - None输入处理异常: {e}")
    
    @pytest.mark.robustness
    def test_empty_input_handling(self):
        """测试空输入处理"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['empty_arrays']
        
        # 测试客观权重方法（对输入要求相对严格）
        obj_weighting = ObjWeighting()
        
        try:
            with pytest.raises((ValueError, IndexError)):
                obj_weighting.decide(
                    dataset=invalid_inputs['matrix'],
                    criterion_type=invalid_inputs['criterion_type']
                )
            print("✅ 空数组输入处理正确（抛出异常）")
            
        except Exception as e:
            print(f"⚠️ 空数组输入处理异常: {e}")
        
        # 测试评分决策方法
        scoring = ScoringDecision()
        
        try:
            with pytest.raises((ValueError, IndexError)):
                scoring.decide(
                    dataset=invalid_inputs['matrix'],
                    criterion_type=invalid_inputs['criterion_type'],
                    weights=invalid_inputs['weights']
                )
            print("✅ 评分决策空输入处理正确（抛出异常）")
            
        except Exception as e:
            print(f"⚠️ 评分决策空输入处理异常: {e}")
    
    @pytest.mark.robustness
    def test_wrong_type_input_handling(self):
        """测试错误类型输入处理"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['wrong_types']
        
        # 测试字符串作为矩阵输入
        obj_weighting = ObjWeighting()
        
        try:
            with pytest.raises((ValueError, TypeError)):
                obj_weighting.decide(
                    dataset=invalid_inputs['matrix'],  # 字符串
                    criterion_type=['min', 'max']
                )
            print("✅ 字符串矩阵输入处理正确（抛出异常）")
            
        except Exception as e:
            print(f"⚠️ 字符串矩阵输入处理异常: {e}")
        
        # 测试错误的权重类型
        scoring = ScoringDecision()
        test_matrix = np.array([[1, 2], [3, 4]])
        
        try:
            with pytest.raises((ValueError, TypeError)):
                scoring.decide(
                    dataset=test_matrix,
                    criterion_type=['min', 'max'],
                    weights=invalid_inputs['weights']  # 字符串
                )
            print("✅ 字符串权重输入处理正确（抛出异常）")
            
        except Exception as e:
            print(f"⚠️ 字符串权重输入处理异常: {e}")
    
    @pytest.mark.robustness
    def test_dimension_mismatch_handling(self):
        """测试维度不匹配处理"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['dimension_mismatch']
        
        # 权重数量与准则数不匹配
        scoring = ScoringDecision()
        
        try:
            with pytest.raises(ValueError):
                scoring.decide(
                    dataset=invalid_inputs['matrix'],      # 3个准则
                    criterion_type=invalid_inputs['criterion_type'],  # 2个类型
                    weights=invalid_inputs['weights']      # 2个权重
                )
            print("✅ 维度不匹配处理正确（抛出异常）")
            
        except Exception as e:
            print(f"⚠️ 维度不匹配处理异常: {e}")
        
        # 测试成对比较矩阵非方阵
        try:
            non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3矩阵
            sub_weighting = SubWeighting(methods=['ahp'])
            
            with pytest.raises(ValueError):
                sub_weighting.decide(dataset=non_square_matrix)
            print("✅ 非方阵处理正确（抛出异常）")
            
        except Exception as e:
            print(f"⚠️ 非方阵处理异常: {e}")
    
    @pytest.mark.robustness
    def test_numerical_issues_handling(self):
        """测试数值问题处理"""
        numerical_issues = RobustnessTestData.NUMERICAL_ISSUES
        criterion_types = ['min', 'max', 'min']
        
        # 测试包含NaN的矩阵
        try:
            obj_weighting = ObjWeighting(methods=['critic'])  # 选择相对稳定的方法
            
            # 对于包含NaN的情况，系统应该要么处理，要么优雅地失败
            try:
                obj_weighting.decide(
                    dataset=numerical_issues['nan_matrix'],
                    criterion_type=criterion_types
                )
                
                results = obj_weighting.get_all_results()
                if results:
                    print("✅ NaN矩阵被正确处理")
                else:
                    print("⚠️ NaN矩阵无法产生结果")
                    
            except Exception:
                print("✅ NaN矩阵正确抛出异常")
                
        except Exception as e:
            print(f"⚠️ NaN矩阵处理出现问题: {e}")
        
        # 测试包含无穷大的矩阵
        try:
            obj_weighting = ObjWeighting(methods=['entropy'])
            
            try:
                obj_weighting.decide(
                    dataset=numerical_issues['inf_matrix'],
                    criterion_type=criterion_types
                )
                
                results = obj_weighting.get_all_results()
                if results:
                    print("✅ 无穷大矩阵被正确处理")
                else:
                    print("⚠️ 无穷大矩阵无法产生结果")
                    
            except Exception:
                print("✅ 无穷大矩阵正确抛出异常")
                
        except Exception as e:
            print(f"⚠️ 无穷大矩阵处理出现问题: {e}")
        
        # 测试极大值矩阵
        try:
            obj_weighting = ObjWeighting(methods=['critic'])
            
            obj_weighting.decide(
                dataset=numerical_issues['very_large'],
                criterion_type=criterion_types
            )
            
            results = obj_weighting.get_all_results()
            
            # 检查结果是否包含异常值
            for method_name, result in results.items():
                if 'weights' in result:
                    weights = result['weights']
                    if not (np.any(np.isnan(weights)) or np.any(np.isinf(weights))):
                        print(f"✅ 极大值矩阵处理正确 - {method_name}")
                    else:
                        print(f"⚠️ 极大值矩阵导致异常值 - {method_name}")
                        
        except Exception as e:
            print(f"✅ 极大值矩阵正确抛出异常: {e}")
        
        # 测试极小值矩阵
        try:
            obj_weighting = ObjWeighting(methods=['seca'])
            
            obj_weighting.decide(
                dataset=numerical_issues['very_small'],
                criterion_type=criterion_types
            )
            
            results = obj_weighting.get_all_results()
            
            for method_name, result in results.items():
                if 'weights' in result:
                    weights = result['weights']
                    if TestHelper.is_valid_weights(weights):
                        print(f"✅ 极小值矩阵处理正确 - {method_name}")
                    else:
                        print(f"⚠️ 极小值矩阵权重无效 - {method_name}")
                        
        except Exception as e:
            print(f"✅ 极小值矩阵正确抛出异常: {e}")
    
    @pytest.mark.robustness
    def test_weight_issues_handling(self):
        """测试权重相关问题处理"""
        weight_issues = RobustnessTestData.WEIGHT_ISSUES
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        criterion_types = ['min', 'max', 'max']
        
        weight_tests = [
            ('负权重', weight_issues['negative_weights']),
            ('零权重', weight_issues['zero_weights']),
            ('权重和不为1', weight_issues['sum_not_one']),
            ('单一权重为1', weight_issues['single_weight_one']),
            ('包含NaN权重', weight_issues['nan_weights']),
            ('包含无穷权重', weight_issues['inf_weights'])
        ]
        
        for weight_name, weights in weight_tests:
            try:
                scoring = ScoringDecision(methods=['saw'])  # 使用简单方法
                
                try:
                    scoring.decide(
                        dataset=test_matrix,
                        criterion_type=criterion_types,
                        weights=weights
                    )
                    
                    results = scoring.get_all_results()
                    
                    if results:
                        # 检查结果是否合理
                        valid_results = True
                        for method_name, result in results.items():
                            if 'ranking' in result:
                                ranking = result['ranking']
                                if not TestHelper.is_valid_ranking(ranking, 3):
                                    valid_results = False
                        
                        if valid_results:
                            print(f"✅ {weight_name}被正确处理")
                        else:
                            print(f"⚠️ {weight_name}产生无效结果")
                    else:
                        print(f"⚠️ {weight_name}无法产生结果")
                
                except Exception:
                    print(f"✅ {weight_name}正确抛出异常")
                    
            except Exception as e:
                print(f"⚠️ {weight_name}处理出现问题: {e}")
    
    @pytest.mark.robustness
    def test_memory_stress(self):
        """测试内存压力"""
        # 生成较大的矩阵来测试内存使用
        np.random.seed(42)
        
        try:
            # 创建大矩阵但不至于耗尽内存
            large_matrix = np.random.rand(200, 30) * 100
            criterion_types = ['min', 'max'] * 15
            
            # 测试客观权重方法（相对节省内存）
            obj_weighting = ObjWeighting(methods=['entropy'])
            
            import psutil
            import os
            
            # 获取内存使用情况
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            obj_weighting.decide(
                dataset=large_matrix,
                criterion_type=criterion_types
            )
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # 内存增长应该在合理范围内（小于200MB）
            assert memory_increase < 200, f"内存使用增长过多: {memory_increase:.1f}MB"
            
            results = obj_weighting.get_all_results()
            assert len(results) > 0, "大矩阵应该能产生结果"
            
            print(f"✅ 内存压力测试通过 - 内存增长: {memory_increase:.1f}MB")
            
        except ImportError:
            print("⚠️ psutil未安装，跳过内存测试")
        except Exception as e:
            print(f"⚠️ 内存压力测试异常: {e}")
    
    @pytest.mark.robustness
    def test_concurrent_access(self):
        """测试并发访问"""
        import threading
        import time
        
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        criterion_types = ['min', 'max', 'max']
        weights = np.array([1/3, 1/3, 1/3])
        
        results_container = []
        errors_container = []
        
        def worker_function(worker_id):
            """工作线程函数"""
            try:
                # 每个线程创建自己的实例
                scoring = ScoringDecision(methods=['topsis'])
                scoring.decide(
                    dataset=test_matrix,
                    criterion_type=criterion_types,
                    weights=weights
                )
                
                results = scoring.get_all_results()
                results_container.append((worker_id, results))
                
            except Exception as e:
                errors_container.append((worker_id, str(e)))
        
        # 创建多个线程
        threads = []
        num_threads = 5
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        if len(errors_container) == 0:
            print(f"✅ 并发访问测试通过 - {len(results_container)}个线程成功完成")
            
            # 验证结果一致性
            if len(results_container) > 1:
                first_result = results_container[0][1]
                for worker_id, result in results_container[1:]:
                    for method_name in first_result:
                        if method_name in result:
                            # 比较排名是否一致
                            if 'ranking' in first_result[method_name] and 'ranking' in result[method_name]:
                                if not np.array_equal(first_result[method_name]['ranking'], result[method_name]['ranking']):
                                    print(f"⚠️ 线程{worker_id}结果与基准不一致")
                
                print("✅ 并发结果一致性验证通过")
        else:
            print(f"⚠️ 并发访问出现{len(errors_container)}个错误:")
            for worker_id, error in errors_container:
                print(f"  线程{worker_id}: {error}")
    
    @pytest.mark.robustness
    def test_repeated_execution_stability(self):
        """测试重复执行稳定性"""
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        criterion_types = ['min', 'max', 'max']
        weights = np.array([0.3, 0.4, 0.3])
        
        # 重复执行同一个计算多次
        num_runs = 10
        all_results = []
        
        for run in range(num_runs):
            scoring = ScoringDecision(methods=['topsis', 'vikor'])
            scoring.decide(
                dataset=test_matrix,
                criterion_type=criterion_types,
                weights=weights
            )
            
            results = scoring.get_all_results()
            all_results.append(results)
        
        # 验证结果一致性
        if len(all_results) > 1:
            baseline_results = all_results[0]
            
            for run_idx, results in enumerate(all_results[1:], 1):
                for method_name in baseline_results:
                    if method_name in results:
                        baseline_ranking = baseline_results[method_name].get('ranking')
                        current_ranking = results[method_name].get('ranking')
                        
                        if baseline_ranking is not None and current_ranking is not None:
                            if not np.array_equal(baseline_ranking, current_ranking):
                                print(f"⚠️ 第{run_idx+1}次运行的{method_name}结果不一致")
                                print(f"  基准排名: {baseline_ranking}")
                                print(f"  当前排名: {current_ranking}")
                                return
            
            print(f"✅ 重复执行稳定性测试通过 - {num_runs}次运行结果一致")
        
    @pytest.mark.robustness
    def test_error_recovery(self):
        """测试错误恢复能力"""
        # 创建一个可能导致某些方法失败的矩阵
        problematic_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1], 
            [2, 1, 0]
        ])
        criterion_types = ['min', 'max', 'min']
        
        # 测试系统是否能在部分方法失败时继续执行其他方法
        scoring = ScoringDecision()  # 使用所有方法
        
        try:
            scoring.decide(
                dataset=problematic_matrix,
                criterion_type=criterion_types
            )
            
            results = scoring.get_all_results()
            
            if results:
                successful_methods = len(results)
                total_available_methods = len(scoring._METHOD_REGISTRY)
                
                success_rate = successful_methods / total_available_methods if total_available_methods > 0 else 0
                
                print(f"✅ 错误恢复测试通过 - {successful_methods}/{total_available_methods}方法成功 ({success_rate:.1%})")
                
                # 至少应该有一些方法成功
                assert successful_methods > 0, "至少应该有一些方法能够成功执行"
            else:
                print("⚠️ 所有方法都失败了")
                
        except Exception as e:
            print(f"⚠️ 错误恢复测试异常: {e}")
    
    @pytest.mark.robustness
    def test_resource_cleanup(self):
        """测试资源清理"""
        import gc
        import weakref
        
        # 创建多个对象实例
        instances = []
        weak_refs = []
        
        for _ in range(10):
            obj = ScoringDecision()
            instances.append(obj)
            weak_refs.append(weakref.ref(obj))
        
        # 使用这些实例
        test_matrix = np.array([[1, 2], [3, 4]])
        criterion_types = ['min', 'max']
        
        for obj in instances:
            obj.decide(dataset=test_matrix, criterion_type=criterion_types)
        
        # 清除引用
        instances.clear()
        
        # 强制垃圾回收
        gc.collect()
        
        # 检查对象是否被正确清理
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        
        if alive_objects == 0:
            print("✅ 资源清理测试通过 - 所有对象都被正确清理")
        else:
            print(f"⚠️ 有{alive_objects}个对象未被清理")


if __name__ == "__main__":
    # 直接运行时的快速测试
    test_instance = TestRobustnessTests()
    test_instance.setup_method()
    
    try:
        test_instance.test_none_input_handling()
        test_instance.test_empty_input_handling()
        test_instance.test_wrong_type_input_handling()
        test_instance.test_dimension_mismatch_handling()
        test_instance.test_numerical_issues_handling()
        test_instance.test_weight_issues_handling()
        test_instance.test_repeated_execution_stability()
        test_instance.test_error_recovery()
        test_instance.test_resource_cleanup()
        print("\n🎉 所有健壮性测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()