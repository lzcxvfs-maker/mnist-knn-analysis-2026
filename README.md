# MNIST Handwritten Digits Classification via k-Nearest Neighbors (KNN)

> 计算机视觉与模式识别课程技术报告配套开源工程  
> 包含 100 组超参数验证网格搜索、SVD PCA / HOG 特征探索、分批向量化 GPU/CPU 距离检索及 10,000 张测试集完整评测。

## 核心指标一览

- **测试集规模**：完整 10,000 张独立测试集（使用全量 60,000 张训练集检索）
- **最终锁定配置**：HOG 特征（324 维）+ $L_2$ 欧氏距离 + $k=5$ + 均匀投票（uniform）
- **最终分类表现**：
  - **Top-1 准确率**：**96.83%** (9683 / 10000)
  - **Top-3 覆盖率**：**99.01%**
  - **Top-5 覆盖率**：**99.26%**
  - **Macro-F1**：**0.9682**
- **验证集搜索**：100 组参数配置，最佳验证准确率 **96.00%**
- **QA 检验**：11 项算法单元测试全部通过；10 组与 scikit-learn 独立对照预测一致率 100%

## 在线浏览成果

- 网页版报告与交互式代码浏览页：[https://lzcxvfs-maker.github.io/mnist-knn-analysis-2026/](https://lzcxvfs-maker.github.io/mnist-knn-analysis-2026/)

## 项目目录结构

- `core.py`: SVD PCA、分批向量化距离计算、平票仲裁、多数表决加权
- `qa.py`: 11 项算法确定性检验与 sklearn 独立对照
- `run.py`: 100 组网格搜索与最终测试评估
- `prepare.py`: 原始 IDX 数据集解压与预处理
- `hog_cache.py`: 局部梯度直方图 (HOG) 特征提取
- `handoff.py`: 交付物生成与检验报告
- `index.html`: 现代学术报告静态网页
