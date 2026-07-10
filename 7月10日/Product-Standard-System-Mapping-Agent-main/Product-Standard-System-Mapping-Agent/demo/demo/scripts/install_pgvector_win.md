# Windows 安装 pgvector（PostgreSQL 18 / Conda）

当前环境的 PostgreSQL 来自 Conda（`E:\miniconda3\Library`）。Conda 上的 `pgvector=0.8.3` 包目前依赖 **PostgreSQL 16**，与 PG18 数据目录不兼容，不能直接 `conda install pgvector`。

## 方式一：源码编译（推荐）

1. 安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选 **C++ 桌面开发**
2. 打开 **x64 Native Tools Command Prompt for VS**
3. 执行：

```cmd
set "PGROOT=E:\miniconda3\Library"
cd %TEMP%
git clone --branch v0.8.3 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

4. 重启 PostgreSQL，再运行：

```bash
python -m src.entry.setup_pgvector
```

## 方式二：项目已内置的内存矩阵加速

若暂未安装 pgvector，服务启动时会自动：

- 将 21090 条向量加载为 NumPy 矩阵
- 用矩阵乘法做 Top-K 检索（约 10~50ms，替代原先 4~12s 的逐条循环）

安装 pgvector 后会自动切换为 HNSW 索引检索。
