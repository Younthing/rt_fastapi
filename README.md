# 分子保留时间预测系统 (RT-FastAPI)

RT-FastAPI是一个基于AttentiveFP模型的分子保留时间预测系统，提供快速准确的液相色谱保留时间预测功能。系统通过FastAPI实现，支持单分子和批量预测，可用于化合物鉴定和实验优化。

## 系统功能

- 提供6分钟和12分钟C18反相色谱保留时间预测
- 支持单个分子和批量分子的保留时间预测
- REST API接口，易于集成到其他系统
- Docker支持，便于部署和扩展
- 模型热加载，无需重启服务即可更新模型

## 依赖要求

- docker

## 安装指南

### 使用Docker（推荐）

1. 构建Docker镜像
   ```bash
   docker build -t rt-fastapi -f docker/dockerfile .
   ```

2. 运行Docker容器
   ```bash
   docker run -p 8000:8000 rt-fastapi
   ```

## 使用方法


docker 启动后可通过 http://localhost:8000/docs 访问交互式API文档。

### API测试

可以使用提供的测试脚本验证API功能：

```bash
python test_api.py
```

## API文档

### 1. 健康检查

- **端点**: `/health`
- **方法**: GET
- **描述**: 检查服务是否正常运行
- **示例响应**:
  ```json
  {"status": "健康"}
  ```

### 2. 服务状态

- **端点**: `/status`
- **方法**: GET
- **描述**: 获取服务运行状态，包括运行时间、模型加载状态等
- **示例响应**:
  ```json
  {
    "status": "运行中",
    "uptime": "0:10:30",
    "uptime_seconds": 630.5,
    "models_loaded": true,
    "models_info": {
      "model_6min": true,
      "model_12min": true
    }
  }
  ```

### 3. 单分子预测

- **端点**: `/predict`
- **方法**: POST
- **描述**: 预测单个分子的保留时间
- **请求体**:
  ```json
  {
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "model_type": "12min"
  }
  ```
- **示例响应**:
  ```json
  {
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "predicted_rt_min": 5.82,
    "predicted_rt_sec": 349.2,
    "success": true
  }
  ```

### 4. 批量预测

- **端点**: `/predict/batch`
- **方法**: POST
- **描述**: 批量预测多个分子的保留时间
- **请求体**:
  ```json
  {
    "molecules": [
      {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "model_type": "12min"
      },
      {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "model_type": "6min"
      }
    ]
  }
  ```

### 5. 模型信息

- **端点**: `/models`
- **方法**: GET
- **描述**: 获取可用模型的详细信息

### 6. API示例

- **端点**: `/examples`
- **方法**: GET
- **描述**: 获取API使用示例

### 7. 重新加载模型

- **端点**: `/reload`
- **方法**: POST
- **描述**: 重新加载预测模型，无需重启服务

## 示例调用

```bash
# 单分子预测
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "model_type": "12min"}'

# 批量预测
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"molecules": [{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "model_type": "12min"}, {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "model_type": "6min"}]}'
```

## 文件结构

```
rt-fastapi/
├── api.py              # FastAPI应用主文件
├── predict_rt.py       # 预测核心功能
├── requirements.txt    # 依赖项列表
├── test_api.py         # API测试脚本
├── docker/             # Docker相关文件
│   └── dockerfile      # Docker构建文件
├── model/              # 预训练模型目录
│   ├── AttentiveModel_RP6min/
│   └── AttentiveModel_RP12min/
├── data/               # 数据目录
└── results/            # 结果输出目录
```

## 注意事项

1. 首次启动服务时，模型加载可能需要一些时间
2. 预测结果仅供参考，实际应用中请结合实验验证
3. 无效的SMILES结构会返回错误信息
4. 支持gpu，需配置gpu依赖
