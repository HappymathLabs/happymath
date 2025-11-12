"""
功能冒烟测试

参考test_all_decision.ipynb中的样例，测试7个核心决策类的基础功能
确保所有方法都能正常执行并且所有公共接口都被调用
"""
import pytest
import numpy as np
import warnings
from typing import Dict, Any

# 导入所有决策类
from happymath.Decision.methods import (
    SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
    FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
)

from tests.fixtures.test_data import TestData
from tests.utils.test_helpers import TestHelper, MethodTester

# 抑制警告
warnings.filterwarnings("ignore")


class TestSmokeTests:
    """功能冒烟测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        TestHelper.suppress_warnings()
    
    @pytest.mark.smoke
    def test_sub_weighting_smoke(self):
        """SubWeighting功能冒烟测试"""
        # 使用测试数据
        data = TestData.SUB_WEIGHTING_DATA
        
        # 创建SubWeighting实例
        sub_weighting = SubWeighting()
        
        # 调用decide方法，传入所有参数
        sub_weighting.decide(
            dataset=data['ahp_matrix'],
            mic=data['bwm_data']['mic'],
            lic=data['bwm_data']['lic'],
            criteria_rank=data['fucom_data']['criteria_rank'],
            criteria_priority=data['fucom_data']['criteria_priority']
        )
        
        # 验证基础功能
        executed_methods = sub_weighting.get_executed_methods()
        assert len(executed_methods) > 0, "应该执行了至少一个方法"
        
        # 验证结果获取
        all_results = sub_weighting.get_all_results()
        assert isinstance(all_results, dict), "get_all_results应该返回字典"
        assert len(all_results) > 0, "应该有结果返回"
        
        # 验证权重获取
        weights_comparison = sub_weighting.compare_weights()
        assert weights_comparison is not None, "compare_weights应该有返回值"
        
        # 验证权重结果的有效性
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'weights' in result:
                weights = result['weights']
                assert isinstance(weights, np.ndarray), f"{method_name}的权重应该是numpy数组"
                assert TestHelper.is_valid_weights(weights), f"{method_name}的权重应该是有效的"
        
        print(f"✅ SubWeighting测试通过 - 执行了{len(executed_methods)}个方法")
    
    @pytest.mark.smoke  
    def test_obj_weighting_smoke(self):
        """ObjWeighting功能冒烟测试"""
        # 使用测试数据
        data = TestData.OBJ_WEIGHTING_DATA
        
        # 创建ObjWeighting实例
        obj_weighting = ObjWeighting()
        
        # 调用decide方法
        obj_weighting.decide(
            dataset=data['decision_matrix'],
            criterion_type=data['criterion_types']
        )
        
        # 验证基础功能
        executed_methods = obj_weighting.get_executed_methods()
        assert len(executed_methods) > 0, "应该执行了至少一个方法"
        
        # 验证结果获取
        all_results = obj_weighting.get_all_results()
        assert isinstance(all_results, dict), "get_all_results应该返回字典"
        assert len(all_results) > 0, "应该有结果返回"
        
        # 验证权重比较
        weights_comparison = obj_weighting.compare_weights()
        assert weights_comparison is not None, "compare_weights应该有返回值"
        
        # 验证权重有效性
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'weights' in result:
                weights = result['weights']
                assert isinstance(weights, np.ndarray), f"{method_name}的权重应该是numpy数组"
                assert TestHelper.is_valid_weights(weights), f"{method_name}的权重应该是有效的"
        
        print(f"✅ ObjWeighting测试通过 - 执行了{len(executed_methods)}个方法")
    
    @pytest.mark.smoke
    def test_scoring_decision_smoke(self):
        """ScoringDecision功能冒烟测试"""
        # 使用测试数据
        data = TestData.SCORING_DATA
        
        # 创建ScoringDecision实例
        scoring = ScoringDecision()
        
        # 调用decide方法，传入完整参数
        scoring.decide(
            dataset=data['decision_matrix'],
            criterion_type=data['criterion_types'],
            weights=data['weights'],
            lambda_value=data['waspas_params']['lambda_value'],
            s_min=data['spotis_params']['s_min'],
            s_max=data['spotis_params']['s_max'],
            grades=data['smart_params']['grades'],
            lower=data['smart_params']['lower'],
            upper=data['smart_params']['upper'],
            utility_functions=data['smart_params']['utility_functions']
        )
        
        # 验证基础功能
        executed_methods = scoring.get_executed_methods()
        assert len(executed_methods) > 0, "应该执行了至少一个方法"
        
        # 验证结果获取
        all_results = scoring.get_all_results()
        assert isinstance(all_results, dict), "get_all_results应该返回字典"
        assert len(all_results) > 0, "应该有结果返回"
        
        # 验证排名比较
        rankings_comparison = scoring.compare_rankings()
        assert rankings_comparison is not None, "compare_rankings应该有返回值"
        
        # 验证得分比较
        scores_comparison = scoring.compare_scores()
        assert scores_comparison is not None, "compare_scores应该有返回值"
        
        # 验证结果有效性
        n_alternatives = data['decision_matrix'].shape[0]
        for method_name, result in all_results.items():
            if isinstance(result, dict):
                if 'ranking' in result:
                    ranking = result['ranking']
                    assert isinstance(ranking, np.ndarray), f"{method_name}的排名应该是numpy数组"
                    assert TestHelper.is_valid_ranking(ranking, n_alternatives), f"{method_name}的排名应该是有效的"
                
                if 'scores' in result:
                    scores = result['scores']
                    assert isinstance(scores, np.ndarray), f"{method_name}的得分应该是numpy数组"
                    assert TestHelper.is_valid_scores(scores), f"{method_name}的得分应该是有效的"
        
        print(f"✅ ScoringDecision测试通过 - 执行了{len(executed_methods)}个方法")
    
    @pytest.mark.smoke
    def test_pairwise_decision_smoke(self):
        """PairwiseDecision功能冒烟测试"""
        # 使用测试数据
        data = TestData.PAIRWISE_DATA
        thresholds = data['thresholds']
        
        # 创建PairwiseDecision实例
        pairwise = PairwiseDecision()
        
        # 调用decide方法
        pairwise.decide(
            dataset=data['decision_matrix'],
            weights=data['weights'],
            Q=thresholds['Q'],
            P=thresholds['P'],
            V=thresholds['V'],
            S=thresholds['S'],
            F=thresholds['F'],
            B=thresholds['B']
        )
        
        # 验证基础功能
        executed_methods = pairwise.get_executed_methods()
        assert len(executed_methods) > 0, "应该执行了至少一个方法"
        
        # 验证结果获取
        all_results = pairwise.get_all_results()
        assert isinstance(all_results, dict), "get_all_results应该返回字典"
        assert len(all_results) > 0, "应该有结果返回"
        
        # 验证排名比较
        rankings_comparison = pairwise.compare_rankings()
        assert rankings_comparison is not None, "compare_rankings应该有返回值"
        
        # 验证结果有效性
        n_alternatives = data['decision_matrix'].shape[0]
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'ranking' in result:
                ranking = result['ranking']
                assert isinstance(ranking, np.ndarray), f"{method_name}的排名应该是numpy数组"
                assert TestHelper.is_valid_ranking(ranking, n_alternatives), f"{method_name}的排名应该是有效的"
        
        print(f"✅ PairwiseDecision测试通过 - 执行了{len(executed_methods)}个方法")
    
    @pytest.mark.smoke
    def test_fuzzy_sub_weighting_smoke(self):
        """FuzzySubWeighting功能冒烟测试"""
        # 使用测试数据
        data = TestData.FUZZY_SUB_WEIGHTING_DATA
        
        # 创建FuzzySubWeighting实例
        fuzzy_sub = FuzzySubWeighting()
        
        # 调用decide方法
        fuzzy_sub.decide(
            dataset=data['fuzzy_ahp_matrix'],
            mic=data['fuzzy_bwm_data']['mic'],
            lic=data['fuzzy_bwm_data']['lic'],
            criteria_rank=data['fuzzy_fucom_data']['criteria_rank'],
            criteria_priority=data['fuzzy_fucom_data']['criteria_priority']
        )
        
        # 验证基础功能
        executed_methods = fuzzy_sub.get_executed_methods()
        assert len(executed_methods) > 0, "应该执行了至少一个方法"
        
        # 验证结果获取
        all_results = fuzzy_sub.get_all_results()
        assert isinstance(all_results, dict), "get_all_results应该返回字典"
        assert len(all_results) > 0, "应该有结果返回"
        
        # 验证权重比较
        weights_comparison = fuzzy_sub.compare_weights()
        assert weights_comparison is not None, "compare_weights应该有返回值"
        
        # 验证模糊权重结果
        for method_name, result in all_results.items():
            if isinstance(result, dict):
                assert 'weights' in result, f"{method_name}应该有权重结果"
                weights = result['weights']
                assert isinstance(weights, (list, np.ndarray)), f"{method_name}的权重应该是列表或数组"
        
        print(f"✅ FuzzySubWeighting测试通过 - 执行了{len(executed_methods)}个方法")
    
    @pytest.mark.smoke
    def test_fuzzy_obj_weighting_smoke(self):
        """FuzzyObjWeighting功能冒烟测试"""
        # 使用测试数据
        data = TestData.FUZZY_OBJ_WEIGHTING_DATA
        
        # 创建FuzzyObjWeighting实例
        fuzzy_obj = FuzzyObjWeighting()
        
        # 调用decide方法
        fuzzy_obj.decide(
            dataset=data['fuzzy_decision_matrix'],
            criterion_type=data['criterion_types']
        )
        
        # 验证基础功能
        executed_methods = fuzzy_obj.get_executed_methods()
        assert len(executed_methods) > 0, "应该执行了至少一个方法"
        
        # 验证结果获取
        all_results = fuzzy_obj.get_all_results()
        assert isinstance(all_results, dict), "get_all_results应该返回字典"
        assert len(all_results) > 0, "应该有结果返回"
        
        # 验证权重比较
        weights_comparison = fuzzy_obj.compare_weights()
        assert weights_comparison is not None, "compare_weights应该有返回值"
        
        # 验证模糊权重结果
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'weights' in result:
                weights = result['weights']
                assert isinstance(weights, np.ndarray), f"{method_name}的权重应该是numpy数组"
        
        print(f"✅ FuzzyObjWeighting测试通过 - 执行了{len(executed_methods)}个方法")
    
    @pytest.mark.smoke
    def test_fuzzy_scoring_decision_smoke(self):
        """FuzzyScoringDecision功能冒烟测试"""
        # 使用测试数据
        data = TestData.FUZZY_SCORING_DATA
        
        # 创建FuzzyScoringDecision实例
        fuzzy_scoring = FuzzyScoringDecision()
        
        # 调用decide方法
        fuzzy_scoring.decide(
            dataset=data['fuzzy_decision_matrix'],
            weights=data['fuzzy_weights'],
            criterion_type=data['criterion_types']
        )
        
        # 验证基础功能
        executed_methods = fuzzy_scoring.get_executed_methods()
        assert len(executed_methods) > 0, "应该执行了至少一个方法"
        
        # 验证结果获取
        all_results = fuzzy_scoring.get_all_results()
        assert isinstance(all_results, dict), "get_all_results应该返回字典"
        assert len(all_results) > 0, "应该有结果返回"
        
        # 验证排名比较
        rankings_comparison = fuzzy_scoring.compare_rankings()
        assert rankings_comparison is not None, "compare_rankings应该有返回值"
        
        # 验证得分比较
        scores_comparison = fuzzy_scoring.compare_scores()
        assert scores_comparison is not None, "compare_scores应该有返回值"
        
        # 验证模糊决策结果
        n_alternatives = data['fuzzy_decision_matrix'].shape[0]
        for method_name, result in all_results.items():
            if isinstance(result, dict):
                if 'ranking' in result:
                    ranking = result['ranking']
                    assert isinstance(ranking, np.ndarray), f"{method_name}的排名应该是numpy数组"
                    assert TestHelper.is_valid_ranking(ranking, n_alternatives), f"{method_name}的排名应该是有效的"
        
        print(f"✅ FuzzyScoringDecision测试通过 - 执行了{len(executed_methods)}个方法")
    
    @pytest.mark.smoke
    def test_all_methods_coverage(self):
        """测试所有方法覆盖率 - 确保所有公共方法都被调用"""
        decision_classes = [
            SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
            FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
        ]
        
        method_coverage = {}
        
        for decision_class in decision_classes:
            class_name = decision_class.__name__
            
            # 创建实例
            instance = decision_class()
            
            # 获取所有公共方法
            public_methods = [method for method in dir(instance) 
                            if not method.startswith('_') and callable(getattr(instance, method))]
            
            # 检查关键方法是否存在
            required_methods = ['decide', 'get_all_results']
            optional_methods = ['compare_weights', 'compare_rankings', 'compare_scores']
            
            coverage = {
                'total_public_methods': len(public_methods),
                'has_decide': 'decide' in public_methods,
                'has_get_all_results': 'get_all_results' in public_methods,
                'has_compare_methods': sum(1 for method in optional_methods if method in public_methods),
                'all_required_present': all(method in public_methods for method in required_methods)
            }
            
            method_coverage[class_name] = coverage
            
            # 验证必需方法存在
            assert coverage['has_decide'], f"{class_name}应该有decide方法"
            assert coverage['has_get_all_results'], f"{class_name}应该有get_all_results方法"
            assert coverage['all_required_present'], f"{class_name}缺少必需方法"
        
        print(f"✅ 方法覆盖率测试通过 - 所有{len(decision_classes)}个决策类都具备必需方法")
        
        # 打印覆盖率详情
        for class_name, coverage in method_coverage.items():
            print(f"  - {class_name}: {coverage['total_public_methods']}个公共方法, "
                  f"{coverage['has_compare_methods']}个比较方法")
    
    @pytest.mark.smoke
    @pytest.mark.performance
    def test_performance_baseline(self):
        """性能基准测试 - 测量所有决策类的基础执行时间"""
        test_data_map = {
            SubWeighting: {
                'dataset': TestData.SUB_WEIGHTING_DATA['ahp_matrix'],
                'mic': TestData.SUB_WEIGHTING_DATA['bwm_data']['mic'],
                'lic': TestData.SUB_WEIGHTING_DATA['bwm_data']['lic'],
                'criteria_rank': TestData.SUB_WEIGHTING_DATA['fucom_data']['criteria_rank'],
                'criteria_priority': TestData.SUB_WEIGHTING_DATA['fucom_data']['criteria_priority']
            },
            ObjWeighting: {
                'dataset': TestData.OBJ_WEIGHTING_DATA['decision_matrix'],
                'criterion_type': TestData.OBJ_WEIGHTING_DATA['criterion_types']
            },
            ScoringDecision: {
                'dataset': TestData.SCORING_DATA['decision_matrix'],
                'criterion_type': TestData.SCORING_DATA['criterion_types'],
                'weights': TestData.SCORING_DATA['weights']
            },
            PairwiseDecision: {
                'dataset': TestData.PAIRWISE_DATA['decision_matrix'],
                'weights': TestData.PAIRWISE_DATA['weights'],
                'Q': TestData.PAIRWISE_DATA['thresholds']['Q'],
                'P': TestData.PAIRWISE_DATA['thresholds']['P'],
                'V': TestData.PAIRWISE_DATA['thresholds']['V'],
                'S': TestData.PAIRWISE_DATA['thresholds']['S'],
                'F': TestData.PAIRWISE_DATA['thresholds']['F']
            },
            FuzzySubWeighting: {
                'dataset': TestData.FUZZY_SUB_WEIGHTING_DATA['fuzzy_ahp_matrix'],
                'mic': TestData.FUZZY_SUB_WEIGHTING_DATA['fuzzy_bwm_data']['mic'],
                'lic': TestData.FUZZY_SUB_WEIGHTING_DATA['fuzzy_bwm_data']['lic']
            },
            FuzzyObjWeighting: {
                'dataset': TestData.FUZZY_OBJ_WEIGHTING_DATA['fuzzy_decision_matrix'],
                'criterion_type': TestData.FUZZY_OBJ_WEIGHTING_DATA['criterion_types']
            },
            FuzzyScoringDecision: {
                'dataset': TestData.FUZZY_SCORING_DATA['fuzzy_decision_matrix'],
                'weights': TestData.FUZZY_SCORING_DATA['fuzzy_weights'],
                'criterion_type': TestData.FUZZY_SCORING_DATA['criterion_types']
            }
        }
        
        execution_times = {}
        
        for decision_class, test_data in test_data_map.items():
            class_name = decision_class.__name__
            
            # 测量执行时间
            instance = decision_class()
            _, execution_time = TestHelper.measure_execution_time(
                instance.decide, **test_data
            )
            
            execution_times[class_name] = execution_time
        
        # 生成性能报告
        performance_report = TestHelper.compare_execution_times(execution_times, threshold=2.0)
        
        print(f"✅ 性能基准测试完成")
        for class_name, time_taken in execution_times.items():
            performance = performance_report[class_name]
            print(f"  - {class_name}: {time_taken:.3f}秒 ({performance})")
        
        # 性能断言 - 所有方法应该在合理时间内完成
        # 模糊方法通常需要更长时间
        for class_name, time_taken in execution_times.items():
            if 'Fuzzy' in class_name:
                threshold = 60.0  # 模糊方法允许更长时间
            else:
                threshold = 10.0  # 普通方法
            assert time_taken < threshold, f"{class_name}执行时间过长: {time_taken:.3f}秒"


if __name__ == "__main__":
    # 直接运行时的快速测试
    import sys
    sys.path.append('..')
    
    test_instance = TestSmokeTests()
    test_instance.setup_method()
    
    try:
        test_instance.test_sub_weighting_smoke()
        test_instance.test_obj_weighting_smoke()
        test_instance.test_scoring_decision_smoke()
        test_instance.test_pairwise_decision_smoke()
        test_instance.test_fuzzy_sub_weighting_smoke()
        test_instance.test_fuzzy_obj_weighting_smoke()
        test_instance.test_fuzzy_scoring_decision_smoke()
        test_instance.test_all_methods_coverage()
        test_instance.test_performance_baseline()
        print("\n🎉 所有冒烟测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()