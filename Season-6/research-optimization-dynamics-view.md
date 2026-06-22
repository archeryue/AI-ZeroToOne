# 从 ODE 到 AdamW:用动力学与数值分析视角串通优化算法

> 缘起:读苏剑林《[从动力学角度看优化算法(一):从 SGD 到动量加速](https://kexue.fm/archives/5655)》之后的延伸讨论笔记。这篇文章把 SGD/Momentum/Nesterov 解释成"同一个 ODE 的不同离散化",非常优雅。本文把这个视角再向两端延伸:
>
> - **向上**(更基础):把 ODE 动力系统、欧拉法这些概念讲透,看清 GD 与数值积分的对应关系
> - **向下**(更现代):把同一根线拉到 Newton、L-BFGS、Adam、AdamW,看清整个优化器家族的设计哲学

整个故事可以浓缩成一句话:

> **所有现代优化器都是在回答同一个问题:"想蹭 Newton 法的好处,但买不起 Newton 法的账单,怎么办?"**

---

## Part 1:优化即动力系统(连续时间视角)

### 1.1 ODE 与动力系统的定义

**ODE(Ordinary Differential Equation,常微分方程)** 是只含一个独立变量(通常是时间 $t$)的微分方程:
$$\dot\theta = f(\theta, t)$$

三个要件:
- **状态(state)** $\theta$:你关心的量
- **时间(time)** $t$:唯一的独立变量
- **演化规则** $f$:告诉你"当前状态下该往哪个方向走、走多快"

若 $f$ 不显含 $t$,称为 **autonomous(自治)** ODE,形式为 $\dot\theta = f(\theta)$。

**动力系统(dynamical system)** 是更一般的概念,只要给定:
1. 状态空间 $\mathcal{M}$
2. 时间集合 $T$(连续或离散)
3. 演化规则

按时间分两类:

| 类型 | 时间 | 形式 | 例子 |
|---|---|---|---|
| **流(flow)** | 连续 $t \in \mathbb{R}$ | $\dot\theta = f(\theta)$ —— 就是 ODE | 行星轨道、单摆 |
| **映射(map)** | 离散 $n \in \mathbb{Z}$ | $\theta_{n+1} = g(\theta_n)$ | GD 迭代本身、Logistic map |

"**ODE 动力系统**" = "用 ODE 描述演化的连续时间动力系统"。

### 1.2 向量场图像

$f(\theta)$ 可以看作 **状态空间上的一个向量场** —— 每一点贴一根箭头,告诉你下一刻往哪走。沿箭头方向"流"出来的轨迹就是 ODE 的解。

### 1.3 GD 作为梯度流 ODE 的欧拉离散化

梯度下降的迭代:
$$\theta_{n+1} = \theta_n - \gamma \nabla L(\theta_n)$$

移项,假装 $\theta_n$ 是连续函数 $\theta(t)$ 在 $t=n\gamma$ 处的采样:
$$\frac{\theta_{n+1} - \theta_n}{\gamma} = -\nabla L(\theta_n) \xrightarrow{\gamma\to 0} \dot\theta = -\nabla L(\theta)$$

这就是 **梯度流(gradient flow)**:
$$\boxed{\dot\theta = -\nabla L(\theta)}$$

**GD 在做的事 = 这个 ODE 的欧拉法离散化。** 学习率 $\gamma$ 在数值积分语言里就是 **时间步长**。

#### 为什么值得做这个翻译?

一旦进入 ODE 框架,经典动力系统理论全部可用:

- **不动点**:$\dot\theta = 0$ ⇔ $\nabla L(\theta) = 0$,即 $L$ 的驻点
- **稳定性**:Jacobian $-H$ 的特征值全负 ⇔ $H$ 全正 ⇔ 极小值点 ⇔ 渐近稳定
- **Lyapunov 函数**:$L$ 自己就是 Lyapunov 函数,$\frac{d}{dt}L = -\|\nabla L\|^2 \le 0$,直接证明收敛性
- **加噪声变 SDE**:平稳分布、Fokker-Planck 等概率工具可用

**最关键的好处**:ODE 描述的是"算法的渐近本质",与具体离散化无关。这就是为什么 Momentum 和 Nesterov 可以解释成 **同一个二阶 ODE 的不同离散化**。

### 1.4 SGD 作为 Langevin SDE

mini-batch 引入的梯度噪声近似为高斯白噪声:
$$\dot\theta = -\nabla L(\theta) + \sigma\xi$$

这是 **Langevin 方程**(一个 SDE)。它的 **平稳分布** 是 Gibbs 测度:
$$P(\theta) \propto \exp\left(-\frac{L(\theta)}{\sigma^2}\right)$$

这个公式信息量极大:

- $L(\theta)$ 的极小值点 ⇔ $P(\theta)$ 的极大值点
- $\sigma^2$ 越小,$P$ 越集中在极小值附近;$\sigma^2$ 越大,$P$ 越接近均匀分布
- $\sigma^2 \sim 1/B$($B$ = batch size),所以 batch size 影响"温度"

**实践推论**:小 batch + 大 LR(高温)→ 广泛探索;大 batch + 小 LR(低温)→ 精细收敛。这解释了 Google 的"Don't Decay LR, Increase Batch Size"。

#### 重要的 caveat

苏神的推导有几个被悄悄当真的简化:

1. **真实噪声非各向同性常数高斯**:$\text{Cov}(g(\theta)) = \Sigma(\theta)$ 是依赖 $\theta$ 的,且实证上 $\Sigma(\theta) \approx H(\theta)$(Sagun et al 2017, Zhu et al 2019)
2. **有效温度同时受 LR 和 batch 影响**:$T \approx \eta\sigma^2/2 \approx \eta C(\theta)/B$,完整版见 Smith & Le 2018
3. **SDE 是 SGD 的弱近似,有 $O(\eta)$ 偏差**:严格分析见 Li, Tai, E 2017

### 1.5 Momentum 作为带摩擦的牛顿系统

把一阶 ODE 升级为带摩擦的二阶 ODE:
$$\ddot\theta + \lambda\dot\theta = -\nabla L(\theta)$$

物理解释:
- $\ddot\theta$:加速度
- $-\nabla L$:作用力(沿势能下降方向)
- $\lambda\dot\theta$:**摩擦阻力**(消耗能量,保证收敛)

不动点($\dot\theta = \ddot\theta = 0$)仍然是 $L$ 的极小点,但 **路径不同**:从山顶滚下的小球可以借助动能冲过浅"坑",越过局部极小,达到更深的谷。

#### Heavy Ball vs Nesterov

容易混淆的一点:

| 算法 | 摩擦系数 | 连续极限 |
|---|---|---|
| **Heavy Ball**(Polyak 1964 / 经典 Momentum) | 常数 $\lambda$ | $\ddot\theta + \lambda\dot\theta = -\nabla L$ |
| **Nesterov 1983** | 时变 $\lambda \sim 3/t$ | $\ddot\theta + \frac{3}{t}\dot\theta = -\nabla L$ |

后者的 vanishing friction 与 Nesterov 在凸函数上的 $O(1/k^2)$ 收敛率直接相关,出处是 **Su-Boyd-Candès 2014:A Differential Equation for Modeling Nesterov's Accelerated Gradient**,是这套连续视角的奠基性论文。

---

## Part 2:离散化(数值分析视角)

### 2.1 欧拉法

给一个 ODE $\dot\theta = f(\theta)$,绝大多数情况没有解析解,只能 **数值近似**:求一系列离散时刻的近似值。

**欧拉法的核心想法**:"切线代替曲线"。在很小的步长 $\gamma$ 内,$f$ 几乎不变,所以用区间起点的值近似整段:
$$\theta_{n+1} = \theta_n + \gamma f(\theta_n)$$

**几何上**:站在当前点,沿脚下箭头走 $\gamma$ 长度,落到新点,重复。光滑曲线变成折线。

### 2.2 欧拉法 = 一阶 Taylor 截断

把 $\theta(t+\gamma)$ 在 $t$ 处 Taylor 展开:
$$\theta(t+\gamma) = \theta(t) + \gamma\dot\theta + \frac{\gamma^2}{2}\ddot\theta + O(\gamma^3)$$

代入 $\dot\theta = f$:
$$\theta(t+\gamma) = \theta(t) + \gamma f(\theta) + \underbrace{\frac{\gamma^2}{2}\ddot\theta + \cdots}_{\text{丢弃}}$$

**欧拉法 = 丢弃二阶及以上的所有项**。局部截断误差 $O(\gamma^2)$,全局误差 $O(\gamma)$ —— 因此叫"一阶方法"。

### 2.3 所有数值 ODE 方法的统一图景

**所有方法都是"对 Taylor 展开做手脚"**:

| 方法 | 保留到 | 计算 $f', f'', \ldots$ 的策略 |
|---|---|---|
| **欧拉(显式)** | $O(\gamma)$ | 直接丢弃高阶项 |
| **Taylor 法** | $O(\gamma^k)$ | 显式求 $f$ 的各阶导数(几乎不用) |
| **Heun / 改进欧拉** | $O(\gamma^2)$ | 多算一次 $f$,用两点平均隐式逼出 $f'$ |
| **RK4** | $O(\gamma^4)$ | 算 4 个不同位置的 $f$,精心组合系数 |
| **隐式欧拉** | $O(\gamma)$ | 在 $\theta_{n+1}$ 处 Taylor 展开 |
| **梯形法** | $O(\gamma^2)$ | 显式 + 隐式各一半 |
| **Leapfrog(蛙跳)** | $O(\gamma^2)$ | 错开半步采样,自然消掉奇数阶误差 |

**关键设计哲学**:除了"Taylor 法"直接求 $f$ 的高阶导数,**其他方法都是通过多评估几次 $f$ 本身,间接逼出高阶 Taylor 系数**。因为 $f$ 复杂时,$f', f''$ 写出来更头疼。

### 2.4 GD 的稳定性问题

欧拉法在 $\dot\theta = -\lambda\theta$ 上的稳定条件:
$$|1 - \gamma\lambda| < 1 \quad \Leftrightarrow \quad \gamma < \frac{2}{\lambda}$$

**翻译到优化**:学习率不能超过 $2/L_{\max}$($L_{\max}$ = 最大 Hessian 特征值),**否则损失爆炸**。这就是"训练发散"的数学根源,也是 Edge of Stability 现象的起点。

### 2.5 Leapfrog → Momentum

Momentum 对应的二阶 ODE:
$$\ddot\theta + \lambda\dot\theta = -\nabla L$$

写成一阶系统:
$$\dot\theta = \eta, \quad \dot\eta = -\lambda\eta - \nabla L(\theta)$$

**Leapfrog 离散化**:$\theta$ 在整数时刻 $n$ 采样,$\eta$ 在半整数时刻 $n+\frac{1}{2}$ 采样:
$$\frac{\theta_{n+1}-\theta_n}{\gamma} = \eta_{n+1/2}$$
$$\frac{\eta_{n+1/2}-\eta_{n-1/2}}{\gamma} = -\lambda\frac{\eta_{n+1/2}+\eta_{n-1/2}}{2} - \nabla L(\theta_n)$$

设 $v_{n+1} = \gamma\eta_{n+1/2}$,$\beta = \frac{1-\lambda\gamma/2}{1+\lambda\gamma/2}$,$\alpha = \frac{\gamma^2}{1+\lambda\gamma/2}$,得到:
$$\boxed{v_{n+1} = \beta v_n - \alpha\nabla L(\theta_n), \quad \theta_{n+1} = \theta_n + v_{n+1}}$$

**这就是带 Momentum 的 GD。**

#### 为什么用 leapfrog 而不是普通二阶欧拉?

Leapfrog 是 **symplectic integrator(辛积分器)** —— 能在每一步上近似保持某种能量,长时间积分不漂移。对二阶力学系统(质量-阻尼-力)特别合适,分子动力学和天体轨道用的就是这一类。

#### 加速原理:$\sqrt\alpha$ vs $\alpha$

- GD:学习率 $\gamma$,步长 $\gamma$,精度 $O(\gamma)$
- Momentum:学习率 $\alpha \approx \gamma^2$,**步长 $\gamma = \sqrt\alpha$**,精度 $O(\alpha)$

固定学习率 $\alpha$ 时,**Momentum 实际以 $\sqrt\alpha$ 前进**($\sqrt\alpha > \alpha$ 当 $\alpha < 1$)。

严格优化理论里,这对应 Nesterov 在条件数 $\kappa$ 的强凸问题上的 $O(\sqrt\kappa\log(1/\epsilon))$ 收敛率 vs GD 的 $O(\kappa\log(1/\epsilon))$ —— **$\sqrt\kappa$ 这个 1/2 次方是同一件事的两种表达**。

### 2.6 Nesterov 作为隐式积分

同一个二阶 ODE,**隐式离散化** 给出:
$$\theta_{n+1} = \theta_n + v_{n+1}, \quad v_{n+1} = \beta v_n - \alpha\nabla L\left(\theta_n + \frac{\beta}{2}v_n\right)$$

(把 $\beta/2$ 改成 $\beta$ 就是标准 Nesterov。)

**隐式方法稳定域更广**,所以通常比显式 Momentum 效果更好。

#### 工程实现的换元技巧

数值自动微分框架(PyTorch / TF)**不会替换求导变量**。所以求 $\nabla L(\theta_n + \beta v_n)$ 不容易直接实现。引入 $\Theta_n = \theta_n + \beta v_n$ 换元,等价改写为:
$$\Theta_{n+1} = \Theta_n + \beta v_{n+1} - \alpha\nabla L(\Theta_n)$$
$$v_{n+1} = \beta v_n - \alpha\nabla L(\Theta_n)$$

**这就是 PyTorch `torch.optim.SGD(momentum=..., nesterov=True)` 实际跑的形式。** 看源码会看到 `d_p = d_p.add(buf, alpha=momentum)` 这一行。

---

## Part 3:现代优化器(成本-精度 Pareto frontier)

### 3.1 起点:二阶 Taylor 与 Newton 法

GD 用一阶 Taylor:$L(\theta+\Delta\theta) \approx L(\theta) + \nabla L\cdot\Delta\theta$ —— **只知道方向,不知道坡陡**。

保留二阶 Taylor:
$$L(\theta+\Delta\theta) \approx L(\theta) + \nabla L\cdot\Delta\theta + \frac{1}{2}\Delta\theta^\top H\Delta\theta$$

对 $\Delta\theta$ 求导=0:$\Delta\theta = -H^{-1}\nabla L$,即:
$$\boxed{\theta_{n+1} = \theta_n - H(\theta_n)^{-1}\nabla L(\theta_n)} \quad \text{(Newton 法)}$$

Newton 一步跳到当前二次近似的"谷底",**严格凸问题上二次收敛**:误差每步平方衰减。

#### 但 DL 里用不动 Newton

| 问题 | 后果 |
|---|---|
| $H \in \mathbb{R}^{d\times d}$,$d \sim 10^{9\sim 12}$ | **存不下**(1B 参数 → Hessian 4 EB) |
| 算 $H$ 需要二阶自动微分 | **比反向传播贵 5~10×** |
| 求 $H^{-1}$ 是 $O(d^3)$ | **算不完** |
| 非凸场景 $H$ 不一定正定 | **可能往坡上爬** |

**整部"DL 优化器进化史"就是一个问题:怎么蹭 Newton 的好处,而不付 Newton 的账单?**

### 3.2 路线 A:Quasi-Newton(BFGS 家族)

**核心想法**:别真算 $H$,**用前后两步梯度差反推**。

一阶 Taylor 推出 **secant 方程**:
$$H\underbrace{(\theta_k - \theta_{k-1})}_{s_k} \approx \underbrace{\nabla L(\theta_k) - \nabla L(\theta_{k-1})}_{y_k}$$

#### BFGS

每步对 $H_k^{-1}$ 做 **秩-2 修正**,保证:
- 满足新的 secant 方程
- 保持正定
- 离上一个 $H_{k-1}^{-1}$"最近"(某种 Frobenius 范数下)

中等规模光滑确定性优化的事实标准。**但要存 $O(d^2)$ 矩阵,DL 用不起。**

#### L-BFGS

**不存矩阵,只存最近 $m$ 步**($m\sim 10\text{-}20$)的 $(s_k, y_k)$ 对,用 two-loop recursion 现场算 $H^{-1}\nabla L$。

代价降到 $O(md)$ 内存 + $O(md)$ 计算。**SciPy `minimize` 默认算法,科学计算工程界的主力。**

**为什么 DL 不用?** secant 方程在 mini-batch 噪声下完全失效:$y_k$ 同时受到"参数变化"和"batch 切换"两个因素污染,$H$ 估计被炸烂。**只在 full-batch 或近 full-batch 场景能用**(典型应用:风格迁移、NeRF、INR 这种小数据全批次场景)。

### 3.3 路线 B:对角近似(Adam 家族)

DL 的标准路线。**核心简化:假设 $H$ 是对角的**。

为什么这个粗暴假设没爆炸?因为 DL 网络的 Hessian **scale 差异主要体现在对角上** —— 不同层、不同神经元的梯度大小差几个数量级,抓住对角就抓住了 90% 的好处。

#### Adagrad(Duchi et al, 2011)

$$G_t = \sum_{k=1}^t g_k^2 \quad \text{(逐元素)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}}\cdot g_t$$

**问题**:$G_t$ 单调累积 → 学习率单调归零 → 训练卡死。

#### RMSProp(Hinton 2012)

把累积换成 **EMA**:
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\cdot g_t$$

$v_t$ 永远不会爆炸到无穷,也不会归零。一个 trick 解决了 Adagrad 的痼疾。

#### Adam(Kingma & Ba, 2014)

加一阶矩(动量):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \eta\cdot\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

**Adam = "一阶矩用动量做平滑 + 二阶矩用对角 Hessian 做缩放"**。默认 $\beta_1=0.9, \beta_2=0.999$。

#### AdamW(Loshchilov & Hutter, 2017)

原版 Adam 把 weight decay 加进梯度,会被 $1/\sqrt{v_t}$ 缩放,**大梯度参数被正则压得轻,小梯度被压得狠**,与初衷相反。

AdamW 解耦:
$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\theta_t\right)$$

**这一个修正让 Transformer 训练稳定性大幅提升,LLM 训练几乎全部用 AdamW。**

### 3.4 深层洞察:为什么 $g^2$ 能近似 $\text{diag}(H)$?

这是对角路线最神秘也最优雅的一步。

#### Fisher 信息矩阵就是个隐藏的 H

对平方损失或负对数似然,**接近最优点时** Hessian 近似等于 **Fisher 信息矩阵**:
$$H \approx F = \mathbb{E}[\nabla L \cdot \nabla L^\top] = \mathbb{E}[g g^\top]$$

(Gauss-Newton 近似 / empirical Fisher)

直觉:$F$ 特征值大表示"梯度方向变化剧烈",而梯度方向变化剧烈正是"损失曲率大"的另一种表达。

#### 取对角线就是 $\mathbb{E}[g^2]$

$$\text{diag}(H) \approx \text{diag}(F) = \mathbb{E}[g^2]$$

**这就是 Adam 的 $v_t$ 在估计的东西**:EMA of $g_t^2$ 是 $\mathbb{E}[g^2]$ 的在线估计。

#### 为什么是 $\sqrt{}$,不是直接除以 $v_t$?

Newton 步 $\Delta\theta_i = -g_i / H_{ii}$。但 $H_{ii} \approx \mathbb{E}[g_i^2]$ 本身已经是 $g_i$ 平方的量级,直接除会得到 $g_i/g_i^2 = 1/g_i$ —— **梯度越大步长越小,直觉相反**。

实际想要的"自适应"是 **步长对梯度尺度不变**:
$$\Delta\theta_i = -\frac{g_i}{\sqrt{\mathbb{E}[g_i^2]}}$$

这个量是无量纲的,**只保留方向 + 符号,大小由全局学习率 $\eta$ 决定**。

这也解释了 **SignSGD / Lion / Sophia** 为什么干脆只取梯度符号 —— 把 $\sqrt{v_t}$ 归一化推到极限就是用 $|g|$ 归一化,再推就是用 $\text{sign}(g)$。

### 3.5 全家福:Pareto frontier

| 算法 | Hessian 近似 | 内存 | 单步计算 | 适用 |
|---|---|---|---|---|
| **GD** | 完全无视(用 $I$) | $O(d)$ | $O(d)$ | 弱基线 |
| **Momentum / Nesterov** | 时间方向的隐式 H 信息 | $O(d)$ | $O(d)$ | 凸或近凸 |
| **Newton** | 真 $H$ | $O(d^2)$ | $O(d^3)$ | 中小规模 |
| **Gauss-Newton / NGD** | $J^\top J$ / Fisher | $O(d^2)$ | $O(d^2)$ | 物理仿真、小网络 |
| **BFGS** | 秩-2 累积 | $O(d^2)$ | $O(d^2)$ | 中规模光滑确定性 |
| **L-BFGS** | 限定 $m$ 步秩-2 | $O(md)$ | $O(md)$ | 大规模确定性 (SciPy) |
| **Adagrad** | $\text{diag}$ 累积 $g^2$ | $O(d)$ | $O(d)$ | 稀疏特征(NLP) |
| **RMSProp** | $\text{diag}$ EMA of $g^2$ | $O(d)$ | $O(d)$ | RNN |
| **Adam** | RMSProp + 一阶动量 | $O(d)$ | $O(d)$ | DL 默认 |
| **AdamW** | Adam + 解耦 weight decay | $O(d)$ | $O(d)$ | **LLM 训练标配** |
| **Lion / SignSGD** | $\text{diag}$ 的极限版 + 动量 | $O(d)$ | $O(d)$ | LLM 新热点(更省内存) |
| **Sophia** | 显式 diag Hessian 估计 + clip | $O(d)$ | $\sim 2\times$ Adam | 实验中,声称比 AdamW 快 2× |
| **Shampoo / SOAP** | 块对角(per-layer Kronecker) | $O(d) + $ 矩阵开销 | 显著 > Adam | 大型训练新兴方案 |

**读这张表的关键**:横轴"Hessian 近似精度",纵轴"计算/内存成本"。每个流行优化器都站在 Pareto frontier 的一个点上,**没有谁绝对优于谁,只有"在你的预算和场景下,哪一格最划算"**。

---

## Part 4:三个视角的统一图景

这套故事可以从三个层次看,**每一层都是同一件事的不同剖面**:

### 视角 1:连续动力学

| 算法 | 对应 ODE | 物理类比 |
|---|---|---|
| GD | $\dot\theta = -\nabla L$ | 黏滞流体里的小球 |
| SGD | $\dot\theta = -\nabla L + \sigma\xi$ | 有热噪声的 Langevin 动力学 |
| Momentum / Nesterov | $\ddot\theta + \lambda\dot\theta = -\nabla L$ | 带摩擦的牛顿力学 |
| SGD + Momentum | $\ddot\theta + \lambda\dot\theta = -\nabla L + \sigma\xi$ | Kramers 方程 |

### 视角 2:数值积分

| 算法 | 数值方法 | 精度 |
|---|---|---|
| GD | 显式欧拉 | $O(\gamma)$ |
| Momentum | Leapfrog(辛积分) | $O(\gamma^2)$ |
| Nesterov | 隐式 / 梯形法 | $O(\gamma^2)$ |
| Newton | "Taylor 法"二阶截断 | $O(\gamma^2)$ 但要求 $f'$ |

### 视角 3:Hessian 近似

| 算法 | $H$ 的估计 |
|---|---|
| GD | $I$(不估计) |
| Momentum | 时间方向隐式 H 信息 |
| BFGS | 全矩阵秩-2 拼凑 |
| L-BFGS | 限定窗口的秩-2 拼凑 |
| Adam | 对角(EMA of $g^2$) |
| Newton | 真 $H$ |

**三个视角是同一个数学母题的不同分类法,各自抓住了优化器的不同本质特征**:
- 连续动力学告诉你 **"它在长时间下渐近行为如何"**
- 数值积分告诉你 **"它的单步逼近误差和稳定性"**
- Hessian 近似告诉你 **"它在每个点上的局部曲率信息有多准"**

---

## 看新优化器时该问的三个问题

一旦你把这三个视角内化,**看任何新优化器都会下意识问**:

1. **它隐式对应哪个连续动力学?** (一阶 ODE? 二阶?有噪声项?)
2. **它用了哪几个评估点 + 怎么组合系数?** (这就是它的 Butcher tableau)
3. **它在 Hessian 近似的 Pareto frontier 上占哪个点?** (对角?块对角?低秩?)

这三个问题问完,这个优化器就 **没什么神秘的了** —— 它不再是"某某大佬拍脑袋发明的",而是 **"Taylor 二阶 + 计算预算 + 噪声鲁棒"** 这个三元约束下的一个**可预测的解**。

---

## 一句话总结

> **优化算法的演化史 = 数值积分方法的演化史 + 对 Newton 法近似策略的演化史**,而 ODE 动力学视角把这两条线缝在了一起。AdamW 之所以是 LLM 训练标配,**不是因为经验上好用,而是因为在 $d=10^{11}$、batch 噪声大、单步预算紧 这三个约束下,它是 Pareto frontier 上唯一现实可行的近 Newton 法**。

---

## 参考文献

### 苏剑林系列(中文,极推荐)
- [从动力学角度看优化算法(一):从 SGD 到动量加速](https://kexue.fm/archives/5655)
- 后续(二)(三)(四)(五)对应 Adam / 学习率调度 / Nesterov 精细分析

### 奠基性论文
- **Su, Boyd, Candès 2014**: *A Differential Equation for Modeling Nesterov's Accelerated Gradient Method*. 推导 Nesterov 的 $\ddot\theta + (3/t)\dot\theta = -\nabla L$ 连续极限
- **Polyak 1964**: *Some methods of speeding up the convergence of iteration methods*. Heavy Ball 起源
- **Nesterov 1983**: *A method for solving the convex programming problem with convergence rate $O(1/k^2)$*. Nesterov 加速起源

### SDE 视角与 SGD 分析
- **Mandt, Hoffman, Blei 2017**: *Stochastic Gradient Descent as Approximate Bayesian Inference*
- **Smith, Le 2018**: *A Bayesian Perspective on Generalization and Stochastic Gradient Descent*
- **Smith et al 2018**: *Don't Decay the Learning Rate, Increase the Batch Size*(Google,直接受 Langevin 视角启发)
- **Li, Tai, E 2017**: *Stochastic Modified Equations and Adaptive Stochastic Gradient Algorithms*(SDE 近似的严格分析)
- **Chaudhari, Soatto 2018**: *Stochastic Gradient Descent Performs Variational Inference, Converges to Limit Cycles for Deep Networks*

### Hessian 几何与现代分析
- **Sagun et al 2017**: *Empirical Analysis of the Hessian of Over-Parametrized Neural Networks*
- **Cohen et al 2021**: *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*(打破上面所有连续时间分析的成立前提)
- **Keskar et al 2017**: *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*

### 优化器演化
- **Duchi, Hazan, Singer 2011**: Adagrad
- **Kingma, Ba 2014**: Adam
- **Loshchilov, Hutter 2017**: *Decoupled Weight Decay Regularization*(AdamW)
- **Chen et al 2023**: *Symbolic Discovery of Optimization Algorithms*(Lion)
- **Liu et al 2023**: *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*
- **Shi et al 2023**: *A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer*

### 数值分析教科书
- **Hairer, Nørsett, Wanner**: *Solving Ordinary Differential Equations I/II*(数值 ODE 的圣经)
- **Nocedal, Wright**: *Numerical Optimization*(BFGS / L-BFGS 的标准引用)
