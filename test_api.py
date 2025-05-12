import requests
import json

# API基础URL
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查端点"""
    response = requests.get(f"{BASE_URL}/health")
    print("健康检查结果:", response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "健康"

def test_status():
    """测试状态端点"""
    response = requests.get(f"{BASE_URL}/status")
    print("状态检查结果:", json.dumps(response.json(), indent=2, ensure_ascii=False))
    assert response.status_code == 200
    assert "status" in response.json()
    assert "models_loaded" in response.json()

def test_models():
    """测试模型信息端点"""
    response = requests.get(f"{BASE_URL}/models")
    print("模型信息:", json.dumps(response.json(), indent=2, ensure_ascii=False))
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) >= 2

def test_examples():
    """测试示例端点"""
    response = requests.get(f"{BASE_URL}/examples")
    print("API示例:", json.dumps(response.json(), indent=2, ensure_ascii=False))
    assert response.status_code == 200
    assert "endpoints" in response.json()

def test_predict_single():
    """测试单个分子预测端点"""
    payload = {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "model_type": "12min"
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("单分子预测结果:", json.dumps(response.json(), indent=2, ensure_ascii=False))
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "predicted_rt_min" in response.json()
    assert "predicted_rt_sec" in response.json()

def test_predict_batch():
    """测试批量分子预测端点"""
    payload = {
        "molecules": [
            {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
                "model_type": "12min"
            },
            {
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
                "model_type": "6min"
            }
        ]
    }
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print("批量预测结果:", json.dumps(response.json(), indent=2, ensure_ascii=False))
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert all("predicted_rt_min" in result for result in response.json())

def test_reload_models():
    """测试模型重载端点"""
    response = requests.post(f"{BASE_URL}/reload")
    print("模型重载结果:", response.json())
    assert response.status_code == 200
    assert "status" in response.json()

def run_all_tests():
    """运行所有测试"""
    print("开始API测试...\n")
    
    # 基本端点测试
    test_health()
    print("=" * 50)
    
    test_status()
    print("=" * 50)
    
    test_models()
    print("=" * 50)
    
    test_examples()
    print("=" * 50)
    
    # 预测功能测试
    test_predict_single()
    print("=" * 50)
    
    test_predict_batch()
    print("=" * 50)
    
    # 管理功能测试
    test_reload_models()
    print("=" * 50)
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    run_all_tests()