from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import time
import datetime
from predict_rt import (
    load_model,
    setup_featurizer,
    featurize_molecule,
    predict_molecule_rt,
    Args_RT6min,
    Args_RT12min
)

app = FastAPI(
    title="分子保留时间预测API",
    description="基于AttentiveFP模型的分子保留时间预测服务",
    version="1.0.0"
)

# 全局变量存储模型和特征化器
model_6min = None
model_12min = None
loader = None
start_time = time.time()
models_loaded = False

class MoleculeInput(BaseModel):
    smiles: str = Field(..., description="分子的SMILES字符串")
    model_type: str = Field("12min", description="模型类型：'6min' 或 '12min'")

class MoleculeBatchInput(BaseModel):
    molecules: List[MoleculeInput] = Field(..., description="分子列表")
    
class PredictionResponse(BaseModel):
    smiles: str
    predicted_rt_min: float
    predicted_rt_sec: float
    success: bool
    error_message: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    uptime: str
    uptime_seconds: float
    models_loaded: bool
    models_info: Dict[str, bool]

class HealthResponse(BaseModel):
    status: str

class ModelInfo(BaseModel):
    name: str
    description: str
    max_rt: float
    n_tasks: int
    num_layers: int
    graph_feat_size: int
    model_dir: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

class ExampleResponse(BaseModel):
    endpoints: Dict[str, Any]

async def load_models():
    """加载所有模型"""
    global model_6min, model_12min, loader, models_loaded
    
    try:
        # 加载6分钟模型
        args_6min = Args_RT6min(input_file="dummy.csv")
        model_6min = load_model(
            args_6min.model_dir,
            n_tasks=args_6min.n_tasks,
            num_layers=args_6min.num_layers,
            graph_feat_size=args_6min.graph_feat_size
        )
        
        # 加载12分钟模型
        args_12min = Args_RT12min(input_file="dummy.csv")
        model_12min = load_model(
            args_12min.model_dir,
            n_tasks=args_12min.n_tasks,
            num_layers=args_12min.num_layers,
            graph_feat_size=args_12min.graph_feat_size
        )
        
        # 设置特征化器
        loader = setup_featurizer("smiles")
        models_loaded = True
        
    except Exception as e:
        models_loaded = False
        raise Exception(f"模型加载失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    await load_models()

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_molecule(molecule: MoleculeInput):
    """预测单个分子的保留时间"""
    try:
        # 选择模型
        model = model_6min if molecule.model_type == "6min" else model_12min
        max_rt = 6.0 if molecule.model_type == "6min" else 12.0
        
        # 特征化分子
        X, success, error_msg = featurize_molecule(molecule.smiles, loader)
        
        if not success:
            return PredictionResponse(
                smiles=molecule.smiles,
                predicted_rt_min=0.0,
                predicted_rt_sec=0.0,
                success=False,
                error_message=error_msg
            )
        
        # 预测保留时间
        rt_min = predict_molecule_rt(model, X, molecule.smiles, min_rt=0.0, max_rt=max_rt)
        rt_sec = rt_min * 60
        
        return PredictionResponse(
            smiles=molecule.smiles,
            predicted_rt_min=rt_min,
            predicted_rt_sec=rt_sec,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch_molecules(batch: MoleculeBatchInput):
    """批量预测分子的保留时间"""
    results = []
    for molecule in batch.molecules:
        try:
            result = await predict_single_molecule(molecule)
            results.append(result)
        except Exception as e:
            results.append(PredictionResponse(
                smiles=molecule.smiles,
                predicted_rt_min=0.0,
                predicted_rt_sec=0.0,
                success=False,
                error_message=str(e)
            ))
    return results

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取API状态，包括运行时间、模型加载状态等"""
    uptime_seconds = time.time() - start_time
    uptime_str = str(datetime.timedelta(seconds=int(uptime_seconds)))
    
    return StatusResponse(
        status="运行中",
        uptime=uptime_str,
        uptime_seconds=uptime_seconds,
        models_loaded=models_loaded,
        models_info={
            "model_6min": model_6min is not None,
            "model_12min": model_12min is not None
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """提供简单的健康检查，用于监控系统"""
    return HealthResponse(status="健康")

@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """获取可用模型的详细信息"""
    args_6min = Args_RT6min(input_file="dummy.csv")
    args_12min = Args_RT12min(input_file="dummy.csv")
    
    models = [
        ModelInfo(
            name="6min",
            description="6分钟C18反相色谱保留时间预测模型",
            max_rt=args_6min.max_rt,
            n_tasks=args_6min.n_tasks,
            num_layers=args_6min.num_layers,
            graph_feat_size=args_6min.graph_feat_size,
            model_dir=args_6min.model_dir
        ),
        ModelInfo(
            name="12min",
            description="12分钟C18反相色谱保留时间预测模型",
            max_rt=args_12min.max_rt,
            n_tasks=args_12min.n_tasks,
            num_layers=args_12min.num_layers,
            graph_feat_size=args_12min.graph_feat_size,
            model_dir=args_12min.model_dir
        )
    ]
    
    return ModelsResponse(models=models)

@app.get("/examples")
async def get_examples():
    """提供API使用示例"""
    return ExampleResponse(
        endpoints={
            "predict": {
                "description": "预测单个分子的保留时间",
                "method": "POST",
                "url": "/predict",
                "example_payload": {
                    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
                    "model_type": "12min"
                }
            },
            "predict_batch": {
                "description": "批量预测分子的保留时间",
                "method": "POST",
                "url": "/predict/batch",
                "example_payload": {
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
            },
            "status": {
                "description": "获取API状态",
                "method": "GET",
                "url": "/status"
            },
            "health": {
                "description": "健康检查",
                "method": "GET",
                "url": "/health"
            },
            "models": {
                "description": "获取可用模型信息",
                "method": "GET",
                "url": "/models"
            },
            "reload": {
                "description": "重新加载模型",
                "method": "POST",
                "url": "/reload"
            }
        }
    )

@app.post("/reload")
async def reload_models():
    """通过API重新加载模型"""
    try:
        await load_models()
        return {"status": "成功", "message": "模型已重新加载"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新加载模型失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 