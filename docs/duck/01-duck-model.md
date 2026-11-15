# 鸭子模型设计规范

本文定义鸭形双足机器人的 MuJoCo 模型几何、运动学拓扑与执行器契约。模型以 `src/reinforcelearning/assets/duck.xml` 为唯一物理资产，启动时由 `DuckModelContract` 严格校验。

## 1. 建模原则

1. **几何极简**：使用基本几何体（box、capsule、sphere），视觉与碰撞几何体分离。
2. **行走最小自由度**：每条腿保留 4-DoF，无髋 yaw、脖子与翅膀。
3. **位置驱动契约**：所有主动关节由 position 执行器驱动，通过实体命名绑定接口。

## 2. 坐标系与基本规格

- **世界坐标系**：`+x` 向前、`+y` 向左、`+z` 向上；初始朝向 `+x`；重力 $g = 9.81\text{ m/s}^2$ 向下。
- **基座拓扑**：躯干根节点使用 6-DoF 无驱动 `freejoint`。
- **全局参考尺度**：总质量约 $1.51\text{ kg}$；默认站立时 root 高度 $z \approx 0.233\text{ m}$，脚底最低点 $z \approx 0.001\text{ m}$ 贴地。

| 刚体部件 | 几何类型 | 名义尺寸 | 质量 | 接触配置与拓扑说明 |
| :--- | :---: | :--- | :--- | :--- |
| **躯干 (torso)** | box | $0.26 \times 0.18 \times 0.14\text{ m}$ | $0.90\text{ kg}$ | 主刚体，挂载 freejoint |
| **头部 (head)** | sphere | 半径 $0.07\text{ m}$ | $0.15\text{ kg}$ | 固定于躯干前上方，兼作触地倾覆检测体 |
| **大腿 (thigh)** | capsule | 间距 $0.08\text{ m}$，半径 $0.030\text{ m}$ | $0.10\text{ kg/侧}$ | 位于 hip body 内，端帽扩展视觉范围 |
| **小腿 (shin)** | capsule | 间距 $0.10\text{ m}$，半径 $0.025\text{ m}$ | $0.08\text{ kg/侧}$ | 端帽终点上抬 $0.025\text{ m}$（避免默认站姿穿地） |
| **脚掌 (foot)** | box | $0.12 \times 0.06 \times 0.02\text{ m}$ | $0.05\text{ kg/侧}$ | 平底支撑面，最低点对齐 $z \approx 0$ |

## 3. 运动学结构

```text
world
└── torso (freejoint)
    ├── head_geom (固定于躯干的几何体)
    ├── left_hip (body: 含 hip_roll/pitch 关节与大腿 capsule)
    │   └── left_shin (body: 含 knee_pitch 关节与小腿 capsule)
    │       └── left_foot (body: 含 ankle_pitch 关节与脚掌 box)
    └── right_hip (body: 对称右腿分支)
        └── right_shin (body)
            └── right_foot (body)
```

## 4. 关节与执行器契约

系统包含 8 个主动单自由度转动关节（`hinge`），由 `position` 执行器驱动。左右腿参数严格对称。

### 4.1 动作映射模型
网络输出归一化动作 $a \in [-1, 1]^8$，映射为位置执行器目标角 $q^* \in \mathbb{R}^8$，并在写入 `data.ctrl` 前完成限位保护：

$$q_i^* = \text{clip}\left(q_{\text{default}, i} + s_i a_i,\quad q_{\min, i} + \delta_{\text{margin}},\quad q_{\max, i} - \delta_{\text{margin}}\right)$$

其中安全边界 $\delta_{\text{margin}} = 0.02\text{ rad}$。

### 4.2 单腿拓扑参数（左/右腿同构）

| 关节名后缀 | 旋转轴 | 限位 $[q_{\min}, q_{\max}]$ (rad) | 默认站姿 $q_{\text{default}}$ | 缩放 $s$ | 名义目标区间 $[q_{\text{def}} \pm s]$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `_hip_roll` | $(1, 0, 0)$ | $[-0.35, +0.35]$ | $0.00$ | $0.20$ | $[-0.20, +0.20]$ |
| `_hip_pitch` | $(0, 1, 0)$ | $[-1.30, +0.80]$ | $-0.30$ | $0.35$ | $[-0.65, +0.05]$ |
| `_knee_pitch` | $(0, 1, 0)$ | $[0.00, +2.20]$ | $+0.60$ | $0.50$ | $[+0.10, +1.10]$ |
| `_ankle_pitch` | $(0, 1, 0)$ | $[-0.90, +0.60]$ | $-0.30$ | $0.25$ | $[-0.55, -0.05]$ |

*全局顺序契约：`left_` 依次 4 关节，后跟 `right_` 依次 4 关节（见 `DuckEnvConfig.JOINT_ORDER`）。*

## 5. 接触与动力学约定

- **接触掩码**：鸭子刚体统一设置 `contype="2" conaffinity="1"`，地面设置 `contype="1" conaffinity="1"`；保证鸭-地接触生效，鸭体自碰撞关闭。
- **关键实体命名**：`floor`、`torso_geom`、`head_geom`、`left_foot_geom`、`right_foot_geom`，禁止在代码中使用魔法数字下标。
- **地面摩擦**：切向摩擦系数配置为 $1.0$。

## 6. MJCF 关键实现细节

1. **类继承绑定**：主动关节、执行器及鸭体 geom 显式声明 `class="duck"`；禁止在 `torso` body 级继承，防止 `freejoint` 误继承阻尼与电枢惯量。
2. **执行器配置**：必须为位置执行器（fixed gain + affine bias），配置 `forcerange="-20 20"`；增益按负载调校（`kp≈20~60`，`kv≈0.5~2.0`）。
3. **范围继承机制**：MuJoCo 4.0.1 的 `inheritrange` 采用实数比例（`1` 表示 100% 关节限位），确保加载后 `ctrlrange` 完全覆盖动作名义范围，避免 MuJoCo 二次截断导致动作失真。
4. **姿态基准**：XML 中 `qpos0`（全 0 直腿）仅用于编译检查；物理站姿基准以 `reset()` 写入的 $q_{\text{default}}$ 为准。

## 7. 验收标准

模型加载、仿真稳定性与契约校验由 `tests/test_duck_model.py` 覆盖；历史实现记录见 [已归档的实施计划](archive/03-implementation-plan.md)。
