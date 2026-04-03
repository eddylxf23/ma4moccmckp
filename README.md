# MA4MOCCMCKP: Multi-Agent Framework for Chance-Constrained Multi-Objective Optimization

基于 Multi-Agent LLM 的机会约束多目标优化框架。Agent-based implementation of OPERA-MC & NHILS algorithms from [arxiv 2026 paper: Multi-Objective Evolutionary Optimization of Chance-Constrained Multiple-Choice Knapsack Problems with Implicit Probability Distributions](https://doi.org/10.48550/arXiv.2603.08209).

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/eddylxf23/ma4moccmckp.git
cd ma4moccmckp

# 2. 创建环境
conda env create -f environment.yml
conda activate ma4moccmckp

# 3. 运行示例
python demo/app.py
```

## 引用
```bash
如果你使用了本项目的代码，请引用：
@misc{li2026moccmckp,
      title={Multi-Objective Evolutionary Optimization of Chance-Constrained Multiple-Choice Knapsack Problems with Implicit Probability Distributions}, 
      author={Xuanfeng Li and Shengcai Liu and Wenjie Chen and Yew-Soon Ong and Ke Tang},
      year={2026},
      eprint={2603.08209},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2603.08209}, 
}
```

## 项目结构
```bash
agents/: Multi-Agent 系统实现（Analyst, Sampler, Solver, Arbiter）
core/: MO-CCMCKP 问题定义与 OPERA-MC 算法
tools/: Agent 可调用的外部工具（Gurobi, Local Search）
evaluation/: 与 NSGA-II/MOEA/D 的对比实验
demo/: Streamlit 可视化界面
```
