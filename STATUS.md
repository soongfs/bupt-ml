# BUPT ML 课程大作业 — ModelNet40 点云分类 项目状态

## 作业目标

- 赛道一：ModelNet40 点云分类
- 评分标准：Instance Accuracy ≥ 92% → +5 分，Class Accuracy ≥ 90% → +5 分，满分 50 分
- 验收时老师下发未知测试集，现场推理输出 CSV

## 项目概况

- 路径：`/Users/soongfs/bupt-ml`（本机 Mac mini M4 写代码），`/home/songqijian/bupt-ml`（服务器训练）
- 服务器：cistn01，8×A100 40GB，SSH `songqijian@10.160.4.87`
- 环境：uv 管理 Python，src-layout，click CLI，git 版本控制
- CLI 入口：`uv run pointcls <command>`
  - `train-all`：下载（如缺失）+ 训练 DGCNN + 训练 PointMLP + 对比
  - `train --config configs/dgcnn.yaml`：单模型训练
  - `test --checkpoint ... --test-dir ... --output ...`：Voting 推理 + 输出 CSV

## 数据

- 来源：百度网盘 `modelnet40_train_data.zip`，解压到 `data/modelnet40/`
- 格式：`.txt` 文件，每行 6 列（x, y, z, nx, ny, nz），已预采样 10000 点/文件，坐标已归一化到 [-1,1]
- 布局：unsplit — 40 个类目录（如 `airplane/`），每个目录内含该类所有 .txt 文件，无预设 train/test 划分
- 划分方式：`ModelNet40Dataset` 按文件名字母序 80/20 自动划分（train 7863，test 1980，总计 9843）
- 残留文件：解压后多出一个 `ModelNet40/` 大写目录（非类目录，已被 dataset 过滤）

## 模型

### DGCNN

- 配置：`configs/dgcnn.yaml`
- 结构：4 层 EdgeConv（k=20）+ 多尺度拼接 + max/avg 双池化 + 3 层 MLP 分类器
- 参数：~1.9M
- 当前配置：use_normals: true, input_dim=6, optimizer=Adam, lr=0.001, scheduler=CosineAnnealingLR (T_max=250), dropout=0.5, label_smoothing=0.2, batch_size=128, epochs=250

### PointMLP

- 配置：`configs/pointmlp.yaml`
- 结构：4 stage + GeometricAffine + ResidualMLP (elite)
- 当前配置：use_normals: false, optimizer=AdamW, batch_size=128, epochs=300

## 训练结果（最新）

| 模型 | Inst Acc | Class Acc | 备注 |
|------|----------|-----------|------|
| DGCNN | 0.8889 | 0.8472 | Voting 10，use_normals=True |
| PointMLP | 0.8753 | 0.8295 | 无 Voting，use_normals=False |

- 旧 DGCNN（use_normals=False，Voting 10）：Inst 0.8869 / Class 0.8435
- 加法向量后提升：Inst +0.0020, Class +0.0037（几乎无变化）

## 实验历史

1. 初始 DGCNN（use_normals=False，epoch 175/250，batch_size=128）：Inst 0.8869（Voting 10）
2. 启用法向量重训（epoch 250/250）：Inst 0.8889（Voting 10）—— 法向量几乎无提升
3. PointMLP 首训（use_normals=False，epoch 300/300）：Inst 0.8753（无 Voting）

## 已修复的 Bug（本次 session）

1. **augment.py**：旧增强代码只旋转 xyz 不旋转法向量。点云旋转后法向量指向错误方向。Codex 修复为同一旋转矩阵同时作用于 xyz 和 normal。
2. **download.py verify_modelnet40**：只认 split 布局（类目录下须有 train/test/），unsplit 布局通不过，每次触发下载。改为同时接受两种布局。
3. **dataset.py 类目录扫描**：未过滤残留目录 `ModelNet40/`（大写），导致 41 个类、标签越界。Codex 添加过滤。
4. **test.py 准确率输出**：推理完不打印准确率。已改为有标签时自动计算并打印 Inst Acc / Class Acc。
5. **test.py unsplit 布局**：原来遍历全量 9843 文件而非只取 20% 测试集。已改为复用 ModelNet40Dataset(split="test")。

## 当前 checkpoint 位置

- 服务器：`runs/dgcnn/best.pth`（最新，use_normals=True 训练）
- 旧 checkpoint 已被删除以兼容新 input_dim=6 架构

## 关键行为特性

- 训练 resume：检测 `runs/{model}/checkpoint.pth` 存在则自动续训
- GPU 预留：preload 前占 5GB 显存（gpu_reserve_mb=5000），防止共享机器被抢卡
- 数据 preload：全量读入内存 + GPU FPS + normalize，num_workers=0
- Voting 推理：test 命令默认 10 票随机旋转，取平均 logits 后 argmax
- 验收输出：CSV 格式 `id,predicted_class`
