# 灰度图像逆问题交替求解器（NumPy + Pillow）

在 Linux CPU 上运行的小尺度灰度图像逆问题求解器。每轮外层迭代交替执行：

1. **像素空间残差梯度步**：`x <- x - α Aᵀ(Ax - y)`；
2. **生成范围内的近似投影**：优化两层小型 tanh 生成映射的隐变量，
   投影目标为观测空间距离 `½‖A G(z,s) − y‖²`（不是像素距离），
   并对稀疏偏移 `s` 支持 L1 惩罚（`lam=0` 时完全关闭）。

观测算子为「零填充边界小核模糊 + 随机缺失像素掩码」，前向与伴随
算子严格互为伴随（有测试断言 `<Ax,y> = <x,Aᵀy>`）。

## 依赖

仅需 NumPy 与 Pillow：

```bash
pip install -r requirements.txt
```

## 运行演示

```bash
python3 demo.py
```

固定种子生成合成图、掩码、模糊核和生成映射权重（不加载任何预训练权重），
结果写入 `outputs/`：

- `observation.png`：带缺失（黑像素）的观测图；
- `reconstruction.png`：重建图；
- `difference.png`：与真值的绝对差值图（标题标注共享放大倍数）；
- `report.json`：可解析的数值报告。

所有 PNG 标题均明确标注 `FULL` 或 `CROP r0:r1 c0:c1` 并用黑框圈出
展示区域；裁剪仅用于展示，报告中的 RMSE 等指标一律基于**未裁剪**数组。

## 作为库使用

```python
from grayrecon.solver import solve, solve_inverse, Status
from grayrecon.reporting import save_outputs

# 高层接口：固定种子的完整合成问题
result = solve(shape=(12, 12), missing_fraction=0.3,
               pixel_step=1.9, max_iters=100, tol=1e-6,
               lam=0.0, inner_max_iters=100, inner_tol=1e-11)
print(result.status, result.final_residual_norm)
save_outputs(result, "outputs", crop_size=20)
```

也可用 `solve_inverse(operator, observation, generator, ...)` 提供自定义
矩阵/偏置（`W1,b1,W2,b2`，见 `grayrecon.generator.Generator`）与
观测算子（`grayrecon.operators.MaskedBlur`）。

生成映射定义为：

```
x = tanh(tanh(z W1 + b1) W2 + b2) + s     # s 为可选稀疏偏移
```

投影内部对 `(z, h, s)` 逐层优化：`z` 块恢复第一层 tanh 耦合，
`h` 块驱动观测残差，`s` 块为 ISTA 近端正交软阈值；各块使用局部
Lipschitz 估计并带目标值回退，保证内层目标下降，内层迭代次数
严格受 `inner_max_iters` 限制。

## 停止状态与错误处理

- `converged`：残差达到容差或外层相对改进连续低于 `stall_tol`；
- `max_iters_reached`：达到外层迭代预算；
- `inner_not_converged`：外层停滞但内层投影未达容差，保留有限结果；
- `unobservable`：观测全部缺失，问题不可约束（不做无意义的求解）。

非有限输入（`NaN/Inf`）、尺寸冲突、非法步长（非正或不满足
`0 < α < 2/‖A‖²`）均抛出 `SolveError`。

`report.json` 逐轮记录：观测残差、梯度步残差、投影目标（观测空间）、
含惩罚的分裂目标、偏移 L1 范数以及实际内层迭代数与是否收敛。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：算子伴随关系、三处内层梯度与像素梯度的有限差分、
固定可解样例的残差改善与容差/预算状态、L1 惩罚对偏移的收缩作用、
全缺失不可约束状态、内层未收敛保留有限结果、非法输入拒绝，
以及导出 PNG 的**真实像素内容**（含裁剪标注与黑框），而非仅检查文件存在。
