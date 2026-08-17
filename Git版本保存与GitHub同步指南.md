# Git 版本保存与 GitHub 同步指南

本文档适用于以下项目：

```text
/Users/pennywise/Desktop/Federated Unlearning POOD
```

当前 GitHub 私有仓库：

```text
https://github.com/pennywisehah/Federated-Unlearning-POOD
```

当前第一版基线标签：

```text
v0.1-pood-federaser-baseline
```

## 1. 最重要的概念

修改本地文件不会自动修改 GitHub 仓库。

完整流程如下：

```text
编辑文件
  ↓
git add：选择准备保存的修改
  ↓
git commit：在本地创建版本记录
  ↓
git push：主动上传到 GitHub
```

只有执行 `git push` 后，GitHub 上的对应分支才会更新。

`v0.1-pood-federaser-baseline` 标签固定指向第一版基线。后续创建新提交不会改变这个标签，也不会覆盖第一版。

## 2. 每次开始工作前进入项目目录

打开终端后先执行：

```bash
cd "/Users/pennywise/Desktop/Federated Unlearning POOD"
```

确认当前位置：

```bash
pwd
```

预期输出：

```text
/Users/pennywise/Desktop/Federated Unlearning POOD
```

确认当前目录是 Git 仓库：

```bash
git status
```

如果出现以下错误，说明终端不在项目目录：

```text
fatal: not a git repository (or any of the parent directories): .git
```

重新执行前面的 `cd` 命令即可。

## 3. 使用开发分支保护 main 基线

建议后续修改都在 `develop` 分支进行，让 `main` 保持为第一版基线。

第一次创建开发分支：

```bash
git switch -c develop
```

如果 `develop` 已经存在，切换到它：

```bash
git switch develop
```

查看当前分支：

```bash
git branch
```

预期输出类似：

```text
* develop
  main
```

星号表示当前所在分支。

分支关系可以理解为：

```text
main ── v0.1 第一版基线
          │
          └── develop ── 后续开发提交
```

## 4. 修改代码后的标准保存流程

### 4.1 查看修改状态

```bash
git status
```

### 4.2 查看具体修改内容

```bash
git diff
```

### 4.3 激活虚拟环境并运行测试

```bash
source FUA-clean/bin/activate
python -m unittest discover -s tests -v
```

测试完成后，如需退出虚拟环境：

```bash
deactivate
```

### 4.4 选择需要保存的文件

添加主要项目目录和文件：

```bash
git add fl_sim tests config README.md requirements.txt
```

如果本次修改还包括其他文档，应单独添加，例如：

```bash
git add "Git版本保存与GitHub同步指南.md"
```

如果只修改了一个文件，可以只添加该文件：

```bash
git add fl_sim/pood.py
```

添加后再次检查：

```bash
git status
```

`Changes to be committed` 下的文件是即将进入新版本的文件。

### 4.5 创建本地提交

```bash
git commit -m "说明这次修改做了什么"
```

示例：

```bash
git commit -m "feat: integrate POOD samples into malicious client training"
```

常见提交说明前缀：

| 前缀 | 用途 | 示例 |
| --- | --- | --- |
| `feat` | 新功能 | `feat: add POOD poisoning pipeline` |
| `fix` | 修复问题 | `fix: correct forgotten sample indices` |
| `test` | 增加或修改测试 | `test: cover POOD client injection` |
| `docs` | 文档修改 | `docs: add Git workflow guide` |
| `refactor` | 重构但不改变功能 | `refactor: simplify data partition code` |

完成 `git commit` 后，版本只保存在本地，还没有上传 GitHub。

## 5. 决定是否上传 GitHub

### 5.1 暂时不想上传

不要执行 `git push` 即可。本地代码和提交不会自动传到 GitHub。

### 5.2 上传 develop 分支作为远程备份

第一次上传 `develop`：

```bash
git push -u origin develop
```

以后继续上传同一个分支，只需要：

```bash
git push
```

这只会更新 GitHub 的 `develop` 分支，不会改变 `main` 分支，也不会改变第一版标签。

上传前可确认当前分支：

```bash
git branch --show-current
```

只有输出 `develop` 时，才执行开发版本的推送。

## 6. 保存第二个正式版本

当第二版功能完成且测试通过后，先确认所有修改已提交：

```bash
git status
```

理想状态应显示：

```text
nothing to commit, working tree clean
```

然后创建第二版标签：

```bash
git tag -a v0.2-pood-attack-pipeline \
  -m "POOD poisoning and federated unlearning attack pipeline"
```

先上传开发分支：

```bash
git push origin develop
```

再上传第二版标签：

```bash
git push origin v0.2-pood-attack-pipeline
```

此时 GitHub 中会同时保留：

```text
v0.1-pood-federaser-baseline   第一版基线
v0.2-pood-attack-pipeline      第二版
```

两个标签指向不同提交，第二版不会覆盖第一版。

## 7. 查看版本历史

查看简洁历史和标签：

```bash
git log --oneline --decorate --graph --all
```

查看第一版基线的信息：

```bash
git show v0.1-pood-federaser-baseline
```

查看第二版的信息：

```bash
git show v0.2-pood-attack-pipeline
```

查看本地标签：

```bash
git tag --list
```

## 8. 查看或恢复第一版

### 8.1 临时查看第一版

先确保当前修改已经提交，否则不要切换版本。

```bash
git switch --detach v0.1-pood-federaser-baseline
```

查看结束后返回开发分支：

```bash
git switch develop
```

在 detached HEAD 状态下不要直接开始长期开发；查看旧版本后应切回 `develop`。

### 8.2 比较当前代码和第一版

比较全部改动：

```bash
git diff v0.1-pood-federaser-baseline
```

只比较某个文件：

```bash
git diff v0.1-pood-federaser-baseline -- fl_sim/pood.py
```

### 8.3 从第一版恢复单个文件

执行前先确认当前修改是否需要提交。

```bash
git restore --source v0.1-pood-federaser-baseline -- fl_sim/pood.py
```

该命令只恢复指定文件，不会自动创建提交。恢复后仍需检查、测试、`git add` 和 `git commit`。

## 9. 检查 GitHub 连接与同步状态

查看远程仓库：

```bash
git remote -v
```

预期地址：

```text
https://github.com/pennywisehah/Federated-Unlearning-POOD.git
```

查看当前分支与远程关系：

```bash
git status -sb
```

如果 `develop` 已上传，输出可能类似：

```text
## develop...origin/develop
```

如果显示 `ahead 1`，表示本地比 GitHub 多一个尚未上传的提交：

```text
## develop...origin/develop [ahead 1]
```

此时是否执行 `git push` 由自己决定。

## 10. 哪些内容不会进入 Git

当前 `.gitignore` 已排除：

```text
FUA/
FUA-clean/
data/
runs/
pood_runs/
__pycache__/
*.py[cod]
.DS_Store
```

这些文件仍保存在电脑上，但不会通过普通 `git add` 进入版本历史，也不会上传到 GitHub。

其中包括：

- Python 虚拟环境；
- MNIST 和 QMNIST 数据集；
- 训练模型和 FedEraser 历史；
- POOD 图片与实验结果。

重要实验结果应另外复制到移动硬盘、云盘或其他备份位置。

## 11. 推荐的日常操作清单

每次开始工作：

```bash
cd "/Users/pennywise/Desktop/Federated Unlearning POOD"
git switch develop
git status
source FUA-clean/bin/activate
```

修改完成后：

```bash
python -m unittest discover -s tests -v
git diff
git status
git add fl_sim tests config README.md requirements.txt
git status
git commit -m "说明本次修改"
```

需要上传备份时：

```bash
git push
```

暂时不希望 GitHub 发生变化时，不执行 `git push`。

## 12. 遇到问题时应停止的情况

出现以下情况时，不要盲目继续执行命令：

- `fatal: not a git repository`；
- `merge conflict`；
- `detached HEAD`，但自己并非有意查看旧版本；
- Git 提示将覆盖本地修改；
- 不确定当前位于 `main` 还是 `develop`；
- 不确定即将提交哪些文件；
- 推送被拒绝或要求强制推送。

先执行以下三个只读命令并保存输出：

```bash
git status
git branch
git log --oneline --decorate -5
```

不要使用以下可能破坏未保存修改的命令，除非已经确认其影响：

```text
git reset --hard
git clean -fd
git push --force
```

