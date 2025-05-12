## 构建

docker build -t deepchem-api:cpu -f docker/dockerfile .

## 运行

docker run  --rm -p 8000:8000 deepchem-api:cpu
## 保存

docker save -o deepchem-api.tar deepchem-api:cpu
## 发送 

scp deepchem-api.tar root@<your-server-ip>:/path/to/destination

## 加载 
docker load -i deepchem-api.tar