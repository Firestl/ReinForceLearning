# 鸭子模型设计

本文定义第一版鸭形双足机器人的 MuJoCo 模型。目标是先获得一个**可控、稳定、可训练**的简化模型，而不是一开始就追求外观和生物拟真。

## 1. 建模原则

1. 先使用基本几何体（box、capsule、sphere），视觉网格后续替换。
2. 视觉几何与碰撞几何分离：外观可以精细，碰撞体必须简单。
3. 每条腿只保留行走所需的最小自由度。
4. 所有关节由位置执行器驱动，便于后续策略输出平滑的目标角度。
5. 通过 `site`、`geom` 名称和传感器名称固定接口，避免环境代码依赖数组下标猜测。

## 2. 坐标系和单位

- 世界坐标系：`+x` 向前，`+y` 向左，`+z` 向上。
- 重力：`0 0 -9.81 m/s²`。
- 角度：弧度；长度：米；质量：千克；时间：秒。
- 初始朝向：鸭子面向 `+x`。
- 根节点使用 `freejoint`，浮动基座具有 6 个无驱动自由度。

## 3. 初始几何与质量参数

以下为第一版占位参数，实际实现后需根据模型总高度和接触测试调整。

| 部件 | 几何体 | 初始尺寸 | 质量 | 备注 |
| --- | --- | --- | --- | --- |
| 躯干 | box | 0.26 × 0.18 × 0.14 m | 0.90 kg | 主刚体，替代鸭身 |
| 头部 | sphere | 半径 0.07 m | 0.15 kg | 可与躯干固定连接，增加视觉辨识度 |
| 大腿 | capsule | 长度 0.08 m，半径 0.030 m | 0.10 kg/侧 | 从髋关节到膝关节 |
| 小腿 | capsule | 长度 0.10 m，半径 0.025 m | 0.08 kg/侧 | 从膝关节到踝关节 |
| 脚掌 | box | 0.12 × 0.06 × 0.02 m | 0.05 kg/侧 | 提供接触面积；碰撞体不能过尖 |

目标总质量约 1.5 kg，站立高度约 0.30~0.40 m。尺寸不是硬性指标，但应保持左右对称，且默认姿态下脚掌贴合地面。

## 4. 运动学结构

```text
world
└── torso (freejoint)
    ├── head (固定连接，可带视觉几何)
    ├── left_hip_body
    │   └── left_thigh
    │       └── left_shin
    │           └── left_foot
    └── right_hip_body
        └── right_thigh
            └── right_shin
                └── right_foot
```

为简化第一版，不单独建模髋部 yaw、脚趾、脖子和翅膀。翅膀可以作为固定视觉几何，不参与控制。

## 5. 关节定义

每条腿 4 个主动关节，共 **8 个动作维度**。关节名、轴和限位将成为环境代码的接口契约。

| 关节名 | 轴（子坐标系） | 初始限位 (rad) | 作用 |
| --- | --- | --- | --- |
| `left_hip_roll` | `1 0 0` | `[-0.35, 0.35]` | 腿左右外摆，帮助平衡 |
| `left_hip_pitch` | `0 1 0` | `[-1.30, 0.80]` | 大腿前后摆动 |
| `left_knee_pitch` | `0 1 0` | `[0.00, 2.20]` | 膝盖弯曲 |
| `left_ankle_pitch` | `0 1 0` | `[-0.90, 0.60]` | 脚掌俯仰，保持贴地 |
| `right_hip_roll` | `1 0 0` | `[-0.35, 0.35]` | 右腿外摆 |
| `right_hip_pitch` | `0 1 0` | `[-1.30, 0.80]` | 右大腿前后摆动 |
| `right_knee_pitch` | `0 1 0` | `[0.00, 2.20]` | 右膝弯曲 |
| `right_ankle_pitch` | `0 1 0` | `[-0.90, 0.60]` | 右脚俯仰 |

### 默认站立姿态

第一版建议给出略微屈膝的默认姿态，避免完全伸直的奇异位形：

```text
hip_roll   =  0.00
hip_pitch  = -0.30
knee_pitch =  0.60
ankle_pitch= -0.30
```

该姿态应满足脚掌大致水平、身体重心位于两脚之间。实际符号和姿态需要通过 MuJoCo 可视化确认。

## 6. 执行器

为每个主动关节添加一个 `position` 执行器：

- `ctrlrange` 与关节限位一致或略小。
- `kp` 初始范围约 20~60，`kv` 初始范围约 0.5~2.0；具体值按关节负载调。
- 使用 `forcerange` 模拟电机力矩上限，防止策略获得不现实的控制能力。
- 环境输出归一化动作 `a ∈ [-1, 1]^8`，再映射为目标关节角：

```text
q_target[i] = q_default[i] + action_scale[i] * a[i]
q_target[i] = clip(q_target[i], q_min[i] + margin, q_max[i] - margin)
data.ctrl[:] = q_target
```

`action_scale` 的初始建议：

```text
[0.20, 0.35, 0.50, 0.25, 0.20, 0.35, 0.50, 0.25]  # rad
```

## 7. 碰撞体和接触设置

- 地面：`plane` 或大尺寸静态 box，位于 `z = 0`。
- 大腿、小腿的碰撞体使用简化 capsule。
- 脚掌使用 box，并确保默认姿态下最低点位于 `z = 0` 附近。
- 初始版本关闭鸭子自碰撞，避免关节附近几何体重叠导致数值爆炸。
- 必须检测：脚掌与地面接触、躯干/头部与地面碰撞。
- 摩擦系数初始取 `1.0` 左右；不能太小，否则策略会持续滑行。

推荐给关键几何体命名，例如：

```text
floor
torso_geom
left_foot_geom
right_foot_geom
```

这样环境代码可以通过名称或 ID 读取接触信息，而不是硬编码数组下标。

## 8. MJCF 结构草案

以下是结构示意，不是最终可运行文件；实现时需要补齐所有 `pos`、`quat`、`inertial` 和几何参数。

```xml
<mujoco model="duck">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <default>
    <default class="duck">
      <joint armature="0.005" damping="0.05" frictionloss="0.01"/>
      <geom friction="1.0 0.02 0.001" condim="3"/>
      <position kp="40" kv="1.0" inheritrange="true"/>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>

    <body name="torso" pos="0 0 0.32">
      <freejoint name="root"/>
      <geom name="torso_geom" class="duck" type="box" size="0.13 0.09 0.07" mass="0.9"/>

      <!-- head / wings / beak：先作为躯干上的固定视觉几何 -->

      <body name="left_hip" pos="0 0.10 -0.05">
        <joint name="left_hip_roll" axis="1 0 0" range="-0.35 0.35"/>
        <joint name="left_hip_pitch" axis="0 1 0" range="-1.30 0.80"/>
        <geom class="duck" type="capsule" fromto="0 0 0 0 0 -0.08" size="0.03" mass="0.10"/>
        <body name="left_shin" pos="0 0 -0.08">
          <joint name="left_knee_pitch" axis="0 1 0" range="0 2.20"/>
          <geom class="duck" type="capsule" fromto="0 0 0 0 0 -0.10" size="0.025" mass="0.08"/>
          <body name="left_foot" pos="0 0 -0.10">
            <joint name="left_ankle_pitch" axis="0 1 0" range="-0.90 0.60"/>
            <geom name="left_foot_geom" class="duck" type="box" size="0.06 0.03 0.01" mass="0.05"/>
          </body>
        </body>
      </body>

      <!-- right leg：与 left 对称，y 坐标取负 -->
    </body>
  </worldbody>

  <actuator>
    <position name="left_hip_roll_act" joint="left_hip_roll" kp="40" kv="1.0"/>
    <!-- 其余 7 个关节同理 -->
  </actuator>
</mujoco>
```

注意：

- 上述 XML 中 `pos` 和层级只用于表达结构，必须根据最终尺寸调整。
- 如果 `inheritrange="true"` 的位置执行器属性不适用当前 MuJoCo 版本，改用显式 `ctrlrange`。
- 质量应由几何体密度或显式 `mass` 决定；不要让多个几何体重复设置总质量导致数值错误。
- 模型加载后先检查警告，再进入环境实现。

## 9. 第一版模型验收标准

- [ ] MuJoCo 能成功加载 `duck.xml`，没有 XML 解析错误。
- [ ] 在仿真中运行 1~2 秒，所有位置和速度保持有限值。
- [ ] 默认姿态下脚掌接地，躯干不会穿透地面。
- [ ] 8 个位置执行器都能被正确控制。
- [ ] 姿态角、关节角、关节速度、足底接触可通过代码稳定读取。
- [ ] 通过 MuJoCo viewer 或离屏渲染能清楚看到模型和运动方向。
- [ ] 模型左右对称，关节名称和索引顺序固定。

## 10. 待定问题

- 脚掌是否需要单独的脚趾关节，以改善跑步时的蹬地效果？
- 头部和翅膀使用独立刚体还是固定视觉几何？
- 第一版使用基本几何体，还是直接引入外部网格？
- 质量分布和腿长是否需要按“鸭子比例”进一步调整？
