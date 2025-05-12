## 构建

docker build -t deepchem-api:cpu -f docker/dockerfile .

## 运行（交互式）

# 可以增加-p参数映射网络端口，-v 参数映射本地文件
docker run --rm -it deepchem-api:cpu
docker run -p 8000:8000 deepchem-api:cpu
## 保存



## 发送 

## 加载 略