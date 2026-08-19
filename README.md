# 可配置联邦学习实验框架

基于 PyTorch/Torchvision 的联邦学习与恶意客户端实验框架。

## 数据集与模型

| 数据集参数 | 模型 | 类别数 | 输入 |
| --- | --- | --- | --- |
| `mnist` | CNN2 | 10 | 1×28×28 |
| `fashionmnist` | CNN2 | 10 | 1×28×28 |
| `cifar10` | ResNet18 | 10 | 3×32×32 |
| `cifar100` | ResNet18 | 100 | 3×32×32 |

CNN2 包含两个带 BatchNorm 的 `3×3` 卷积层、两次最大池化和 Dropout，
池化后直接展平 `64×7×7` 特征。CIFAR 的 ResNet18 使用适合 32×32 图像的
`3×3, stride=1` 输入卷积，并移除了原始 ImageNet MaxPool。

## 功能

- 聚合：FedAvg、Krum、Trimmed Mean
- 数据划分：IID 或 Dirichlet Non-IID
- 可配置客户端总数与每轮参与比例
- 可配置恶意客户端比例
- 攻击：Sign Flip、高斯噪声、Model Replacement
- 自动使用 CUDA、Apple MPS 或 CPU
- 保存逐轮指标、完整配置和 PyTorch 模型检查点

## 安装

要求 Python 3.9 或更高版本。

```bash
python3 -m pip install -r requirements.txt
```

## 运行

首次运行默认自动下载数据集：

```bash
python3 -m fl_sim --config config/default.json
```

选择 FashionMNIST：

```bash
python3 -m fl_sim \
  --config config/default.json \
  --dataset fashionmnist \
  --aggregation trimmed_mean
```

选择 CIFAR-10 和 ResNet18：

```bash
python3 -m fl_sim \
  --config config/default.json \
  --dataset cifar10 \
  --aggregation krum \
  --num-clients 20 \
  --malicious-fraction 0.2
```

选择 CIFAR-100：

```bash
python3 -m fl_sim \
  --config config/default.json \
  --dataset cifar100 \
  --aggregation fedavg \
  --rounds 100 \
  --local-epochs 2
```

命令行参数会覆盖 JSON 配置。完整参数：

```bash
python3 -m fl_sim --help
```

## 关键配置

| 参数 | 说明 |
| --- | --- |
| `dataset` | `mnist`、`fashionmnist`、`cifar10`、`cifar100` |
| `data_dir` | 数据集保存目录 |
| `download` | 是否自动下载数据 |
| `iid` | `true` 为 IID，`false` 为 Dirichlet Non-IID |
| `partition` | `auto`、`iid`、`dirichlet` 或每客户端单标签的 `label_per_client` |
| `non_iid_alpha` | Dirichlet α；越小客户端数据差异越大 |
| `num_clients` | 客户端总数 |
| `participation_fraction` | 每轮客户端参与比例 |
| `malicious_fraction` | 恶意客户端比例 |
| `aggregation` | `fedavg`、`krum`、`trimmed_mean` |
| `attack` | `none`、`sign_flip`、`gaussian`、`model_replacement` |
| `device` | `auto`、`cpu`、`cuda`、`mps` |

Krum 要求当轮客户端数 `n >= 2f + 3`。ResNet18 更新较大，客户端数量较多时
Trimmed Mean 会消耗较多内存；框架使用 `aggregation_chunk_size` 分块聚合来控制峰值。

## 输出

实验结果保存在：

```text
runs/时间-数据集-聚合方式-seed/
  config.json
  metrics.csv
  model.pt
```

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## MNIST/QMNIST POOD 最小构造

使用训练完成的 MNIST 全局模型，从 MNIST 测试集选择一个被正确分类的目标
样本，并从 QMNIST `test50k` 随机抽取候选池。候选池会先删除与目标相同标签
的样本，再用 CNN2 的 128 维倒数第二层特征执行余弦 kNN，并从近邻中选择
数量最多的同标签组作为 `p` 个 POOD 种子样本。

筛选完成后，程序会进一步优化一个由全部 `p` 个样本共享的通用扰动。优化目标
是最小化扰动样本特征与目标样本特征之间的平均 L2 距离。扰动在原始 `[0, 1]`
像素空间中施加，并使用可配置的 L-infinity 预算约束。

默认自动使用 `runs/` 下最新的 `model.pt`：

```bash
FUA/bin/python -m fl_sim.pood \
  --device cpu \
  --target-label 3 \
  --candidate-count 1000 \
  --k 20 \
  --p 5 \
  --perturb-steps 200 \
  --perturb-lr 0.01 \
  --perturb-epsilon 0.3
```

也可以明确指定模型：

```bash
FUA/bin/python -m fl_sim.pood \
  --checkpoint runs/实验目录/model.pt \
  --device cpu
```

结果保存在 `pood_runs/`，包括 `summary.json`、`pood_knn.pt`、目标样本、完整
kNN 邻居图、同标签 POOD 种子图和添加通用扰动后的 `perturbed_pood.png`。
`summary.json` 会记录优化前后的平均特征距离、平均目标特征相似度以及扰动范数。

## CIFAR-10/CIFAR-100 PUA 集成测试

`fl_sim.CIFAR_PUA` 使用 CIFAR-10 作为联邦训练与目标数据集，使用 CIFAR-100
测试集作为 POOD 候选池。由于两个数据集的标签空间不同，程序以 CIFAR-100
真实类别保证候选的语义一致性，并使用 CIFAR-10 代理模型的非目标预测作为
注入训练标签。检索在 ResNet18 的 512 维倒数第二层特征空间中执行精确余弦
Top-K。

先训练一个 CIFAR-10 代理模型：

```bash
FUA-clean/bin/python -m fl_sim \
  --dataset cifar10 \
  --data-dir data \
  --download \
  --partition iid \
  --aggregation fedavg \
  --num-clients 10 \
  --malicious-fraction 0 \
  --participation-fraction 1 \
  --attack none \
  --rounds 50 \
  --local-epochs 1 \
  --learning-rate 0.01 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --device cpu \
  --output-dir runs/cifar10-proxy
```

再把生成的 `model.pt` 路径传给端到端 PUA：

```bash
FUA-clean/bin/python -m fl_sim.CIFAR_PUA \
  --proxy-checkpoint "runs/cifar10-proxy/实验目录/model.pt" \
  --data-dir data \
  --download \
  --device cpu \
  --non-iid-alpha 0.1 \
  --num-clients 10 \
  --malicious-client-id 6 \
  --target-label 3 \
  --candidate-count 5000 \
  --k 500 \
  --p 5 \
  --perturb-steps 200 \
  --perturb-lr 0.005 \
  --perturb-epsilon 0.031372549 \
  --poison-repeats 100 \
  --rounds 50 \
  --local-epochs 1 \
  --learning-rate 0.01 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --unlearning-delta-t 5 \
  --calibration-local-epochs 1
```

该入口会自动完成 POOD 检索与扰动、Dirichlet Non-IID 联邦训练、注入记录的
部分数据 FedEraser，以及全局、遗忘集、目标样本和各 CIFAR-10 类别的前后评估。
结果默认写入 `cifar_pua_runs/`。

## CIFAR-10/STL-10 PUA 集成测试

`fl_sim.STL_PUA` 保留 CIFAR-10 作为联邦训练集和目标数据集，使用 STL-10
有标签测试集作为 POOD 候选池。STL-10 图片会从 `96×96` 缩放到 `32×32`，
并使用 CIFAR-10 的归一化参数。标签使用固定映射：airplane、bird、car、cat、
deer、dog、horse、ship、truck 分别映射到对应 CIFAR-10 类别，其中 car 映射为
automobile；无法映射的 monkey 被排除。映射后与目标相同的类别也会被排除，
因此注入标签来自 STL-10 真实标签映射，而不是代理模型伪标签。

Windows PowerShell 完整实验指令：

```powershell
python -m fl_sim.STL_PUA `
  --proxy-checkpoint "runs/cifar10-proxy/20260812-190736-908765-cifar10-fedavg-seed42/model.pt" `
  --data-dir data `
  --download `
  --device cuda `
  --non-iid-alpha 0.1 `
  --num-clients 10 `
  --malicious-client-id 6 `
  --target-label 2 `
  --candidate-count 5000 `
  --k 200 `
  --p 5 `
  --perturb-steps 200 `
  --perturb-lr 0.005 `
  --perturb-epsilon 0.031372549 `
  --poison-repeats 100 `
  --rounds 50 `
  --local-epochs 1 `
  --learning-rate 0.01 `
  --batch-size 128 `
  --eval-batch-size 256 `
  --unlearning-delta-t 5 `
  --calibration-local-epochs 1
```

首次运行使用 `--download` 下载 STL-10，以后可改为 `--no-download`。结果默认
写入 `stl_pua_runs/`，其中 `summary.json` 会同时保存 STL-10 原始类别、映射后
CIFAR-10 注入类别、缩放方式、选中源索引以及 Unlearning 前后指标。

### POOD 检索消融与必要对照

`fl_sim.STL_PUA` 默认使用余弦 Top-K 和优化后的 POOD。以下参数可在保持训练、
目标和 FedEraser 设置不变时执行消融：

```text
--retrieval-metric cosine|l2
--experiment-mode optimized|no_pood|unperturbed|random
```

- `no_pood`：不注入也不删除数据，只执行相同的 FedEraser 校准回放；结果中的
  POOD 指标标记为 probe-only。
- `unperturbed`：注入 Top-K 检索得到的原始 POOD，不执行扰动优化。
- `random`：从 Top-K 选定的同一 STL-10 语义/CIFAR-10 标签组内随机取 `p` 张
  原始图片，控制类别和注入标签不变。
- `l2`：用未经归一化的512维特征欧氏距离执行精确 Top-K；默认 `cosine` 行为
  与旧版本一致。

端到端 `fl_sim.PUA`、`fl_sim.CIFAR_PUA` 和 `fl_sim.STL_PUA` 会直接使用内存中的
FedEraser 历史完成本次 Unlearning，默认不再写入体积很大的
`federaser_history.pt`。只有需要稍后使用 `fl_sim.unlearning` 独立重放同一次训练
时，才在集成命令中加入：

```text
--keep-unlearning-history
```

## FedEraser部分数据遗忘

FedEraser需要在原始联邦训练期间保存历史客户端更新。`unlearning_delta_t`
表示每隔多少轮保存一次更新；如果最后一轮不是该间隔的整数倍，程序也会保存
最后一轮。当前基线要求使用FedAvg：

```bash
FUA/bin/python -m fl_sim \
  --config config/default.json \
  --device cpu \
  --save-unlearning-history \
  --unlearning-delta-t 5
```

这里是独立训练入口，只有显式指定 `--save-unlearning-history` 才会在训练目录中
新增 `federaser_history.pt`。部分遗忘时，索引指请求客户端本地
数据集中的位置，而不是MNIST原始数据集的全局索引。例如遗忘客户端0的本地
位置10、11和12：

```bash
FUA/bin/python -m fl_sim.unlearning \
  --run-dir runs/训练目录 \
  --forget-client-id 0 \
  --forget-local-indices 10,11,12 \
  --calibration-local-epochs 1 \
  --device cpu \
  --no-download
```

索引较多时可使用`--forget-indices-file path/to/indices.json`，文件内容是JSON
整数列表。输出包括`summary.json`和`unlearned_model.pt`。如需客户端级遗忘，
将索引参数替换为`--forget-all-client-data`。

遗忘期间终端会逐个显示snapshot、客户端校准进度以及耗时。遗忘完成后会分别
使用原始模型和遗忘模型评估被遗忘集合上的准确率。结果保存在`summary.json`
的`forgotten_set_metrics`字段中。

## 单标签客户端 Non-IID 遗忘测试

`label_per_client` 会要求客户端数量等于数据集类别数，并固定令客户端ID与标签
一致。对于MNIST的10个客户端，客户端0只持有数字0，客户端1只持有数字1，
依此类推。训练并保存FedEraser历史：

```bash
FUA-clean/bin/python -m fl_sim \
  --config config/mnist_label_per_client.json
```

例如遗忘只持有数字3的客户端3：

```bash
FUA-clean/bin/python -m fl_sim.unlearning \
  --run-dir runs/训练目录 \
  --forget-client-id 3 \
  --forget-all-client-data \
  --calibration-local-epochs 1 \
  --device cpu \
  --no-download
```

遗忘结果除了整体测试集和遗忘训练集指标，还会在`per_label_test_metrics`中记录
每个标签的测试ACC，并在`forgotten_label_metrics`中单独记录数字3遗忘前后的
ACC与下降幅度。这样可以区分目标标签遗忘效果和模型整体性能退化。
