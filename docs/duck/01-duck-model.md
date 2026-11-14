# 鸭子模型设计

本文定义第一版鸭形双足机器人的 MuJoCo 模型。目标是先获得一个可控、稳定、可训练的简化模型，而不是一开始就追求外观和生物拟真。

## 1. 建模原则

1. 使用基本几何体（box、capsule、sphere），视觉网格后续替换。
2. 视觉几何与碰撞几何分离，碰撞体尽量简单；第一版头部可复用同一个球体。
3. 每条腿只保留行走所需的最小自由度；第一版暂不建模髋 yaw、脖子和翅膀。
4. 所有关节由位置执行器驱动；用 joint、geom、actuator 名称固定接口，后续需要 IMU/足底传感器时再补 `site`/sensor 名称。

## 2. 坐标系和单位

- 世界系：`+x` 向前、`+y` 向左、`+z` 向上；鸭子初始面向 `+x`。
- 单位：米、弧度、秒、千克；重力 `0 0 -9.81 m/s²`。
- 根节点使用 `freejoint`，浮动基座有 6 个无驱动自由度。

## 3. 初始几何与质量参数

以下为第一版占位参数，实现后需按总高度和接触测试调整。目标总质量约 1.5 kg；默认姿态下 root（躯干中心）高度约 0.233 m、躯干顶部约 0.30 m、头顶约 0.37 m，脚掌贴合地面；保持左右对称。

| 部件 | 几何体 | 初始尺寸 | 质量 | 备注 |
| --- | --- | --- | --- | --- |
| 躯干 | box | 0.26 × 0.18 × 0.14 m | 0.90 kg | 主刚体，替代鸭身 |
| 头部 | sphere | 半径 0.07 m | 0.15 kg | 固定在躯干前上方，兼作简化碰撞体 |
| 大腿 | capsule | 髋-膝间距 0.08 m，半径 0.030 m | 0.10 kg/侧 | 髋到膝；capsule 端帽使视觉总长更长 |
| 小腿 | capsule | 膝-踝间距 0.10 m，半径 0.025 m | 0.08 kg/侧 | 膝到踝；capsule 端帽使视觉总长更长 |
| 脚掌 | box | 0.12 × 0.06 × 0.02 m | 0.05 kg/侧 | 提供接触面积；碰撞体不能过尖 |

尺寸不是硬性指标，但应保持左右对称，且默认姿态下脚掌贴合地面。表中大腿/小腿长度指关节间距，XML `fromto` 与之对应（小腿碰撞胶囊除外，见 §7 防穿地截断）；capsule 端帽会让视觉和碰撞范围大于该值，运动学连杆长度仍以关节间距为准。

## 4. 运动学结构

```text
world
└── torso (freejoint)
    ├── head_geom (躯干上的固定几何)
    ├── left_hip (body；含髋关节和 thigh geom)
    │   └── left_shin (body)
    │       └── left_foot (body)
    └── right_hip (body；含髋关节和 thigh geom)
        └── right_shin (body)
            └── right_foot (body)
```

XML 中大腿用 `left_hip`/`right_hip` body 内的 capsule geom 表示，不单独建 `thigh` body；节点名以 XML 为准。

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

按当前腿长（0.08 + 0.10 m）和上述屈膝角度，髋到踝的垂直距离约 0.172 m；root 取约 0.233 m 时脚底最低点接近 `z = 0`。改变腿长、髋位置或默认姿态后需重新核算。

## 6. 执行器

为每个主动关节添加一个 `position` 执行器：

- `ctrlrange` 与关节限位一致或略小；`kp` 约 20~60，`kv` 约 0.5~2.0，按负载调整。
- 用 `forcerange` 模拟电机力矩上限，避免策略获得不现实的力矩。
- 环境输出归一化动作 `a ∈ [-1, 1]^8`，映射为目标角并裁剪：

```text
q_target[i] = q_default[i] + action_scale[i] * a[i]
q_target[i] = clip(q_target[i], q_min[i] + margin, q_max[i] - margin)
data.ctrl[:] = q_target
```

动作顺序必须与关节顺序、XML 中 actuator 的定义顺序一致。初始映射：

| # | 关节 | `q_default` (rad) | `action_scale` (rad) | 名义目标范围 (rad) |
| --- | --- | --- | --- | --- |
| 0 | `left_hip_roll` | 0.00 | 0.20 | `[-0.20, 0.20]` |
| 1 | `left_hip_pitch` | -0.30 | 0.35 | `[-0.65, 0.05]` |
| 2 | `left_knee_pitch` | 0.60 | 0.50 | `[0.10, 1.10]` |
| 3 | `left_ankle_pitch` | -0.30 | 0.25 | `[-0.55, -0.05]` |
| 4 | `right_hip_roll` | 0.00 | 0.20 | `[-0.20, 0.20]` |
| 5 | `right_hip_pitch` | -0.30 | 0.35 | `[-0.65, 0.05]` |
| 6 | `right_knee_pitch` | 0.60 | 0.50 | `[0.10, 1.10]` |
| 7 | `right_ankle_pitch` | -0.30 | 0.25 | `[-0.55, -0.05]` |

名义目标范围是 `q_default ± action_scale`；实际还要按关节限位和 margin 裁剪。`data.ctrl` 按 XML 中 actuator 的定义顺序写入，因此 actuator 顺序必须与上表一致。

## 7. 碰撞体和接触设置

- 地面：`z = 0` 的 `plane` 或大尺寸静态 box。
- 大腿使用 capsule 碰撞体；小腿 capsule 的 `fromto` 终点在踝关节上方 `0.025 m` 处（`-0.075` 而非 `-0.10`），否则半径 `0.025 m` 的端帽在默认姿态下会穿地约 `14 mm`；运动学连杆长度仍为 `0.10 m`。脚掌使用 box，默认姿态下最低点实测 `z = +0.001 m`。
- 第一版用位掩码关闭鸭子自碰撞：鸭子 geom 统一 `contype="2" conaffinity="1"`，地面 `contype="1" conaffinity="1"`；这样鸭子-地面接触保留，鸭子-鸭子不接触。
- 后续需要腿-腿或腿-躯干碰撞时，恢复相关 geom 的 `contype`/`conaffinity`，再用 `<contact><exclude body1="..." body2="..."/></contact>` 排除父子/相邻 pair，而不是直接打开全部自碰撞。
- 必须检测足底接触以及躯干/头部与地面碰撞。
- 摩擦系数初始约 `1.0`，过小会导致持续滑行。
- 关键几何体命名为 `floor`、`torso_geom`、`head_geom`、`left_foot_geom`、`right_foot_geom`，避免硬编码下标。

## 8. MJCF 结构草案

以下是结构示意，不是最终可运行文件；实现时需要补齐所有 `pos`、`quat`、`inertial` 和几何参数。

```xml
<mujoco model="duck">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <default>
    <default class="duck">
      <joint armature="0.005" damping="0.05" frictionloss="0.01"/>
      <geom friction="1.0 0.02 0.001" condim="3" contype="2" conaffinity="1"/>
      <position kp="40" kv="1.0" inheritrange="1"/>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" contype="1" conaffinity="1" friction="1.0 0.02 0.001"/>

    <body name="torso" pos="0 0 0.233">
      <freejoint name="root"/>
      <geom name="torso_geom" class="duck" type="box" size="0.13 0.09 0.07" mass="0.9"/>

      <!-- 头部：固定在躯干前上方；pos 按外观调整 -->
      <geom name="head_geom" class="duck" type="sphere" size="0.07" mass="0.15" pos="0.13 0 0.07"/>
      <!-- wings / beak：后续作为躯干上的固定视觉几何 -->

      <body name="left_hip" pos="0 0.10 -0.05">
        <joint name="left_hip_roll" class="duck" axis="1 0 0" range="-0.35 0.35"/>
        <joint name="left_hip_pitch" class="duck" axis="0 1 0" range="-1.30 0.80"/>
        <geom class="duck" type="capsule" fromto="0 0 0 0 0 -0.08" size="0.03" mass="0.10"/>
        <body name="left_shin" pos="0 0 -0.08">
          <joint name="left_knee_pitch" class="duck" axis="0 1 0" range="0 2.20"/>
          <geom class="duck" type="capsule" fromto="0 0 0 0 0 -0.075" size="0.025" mass="0.08"/>
          <!-- 小腿碰撞胶囊在踝上方 0.025 m 截断，端帽不穿地；运动学长度仍为 0.10 m -->
          <body name="left_foot" pos="0 0 -0.10">
            <joint name="left_ankle_pitch" class="duck" axis="0 1 0" range="-0.90 0.60"/>
            <geom name="left_foot_geom" class="duck" type="box" size="0.06 0.03 0.01" mass="0.05"/>
          </body>
        </body>
      </body>

      <!-- right leg：与 left 对称，y 坐标取负；关节、geom 和执行器都要加 class="duck" -->
    </body>
  </worldbody>

  <actuator>
    <position name="left_hip_roll_act" joint="left_hip_roll" class="duck"/>
    <!-- 其余 7 个执行器同样加 class="duck" -->
  </actuator>
</mujoco>
```

注意：

- `pos` 和层级仅表达结构，需按最终尺寸调整；root 的 `0.233 m` 是按当前腿长和默认姿态核算的接地高度，改尺寸后需同步。
- MuJoCo 4.0.1 中 `inheritrange` 是正实数比例（`0` 表示禁用，`1` 表示使用完整关节范围，也允许大于 1），不是布尔值。已验证 `inheritrange="1"` 加载后 `ctrlrange` 等于关节限位且 `ctrllimited` 自动启用。
- 质量由密度或显式 `mass` 决定，避免重复计入；草案只展开左腿，补齐右腿后总质量约 `1.51 kg`。
- 头部的 `pos="0.13 0 0.07"` 是占位值；按当前尺寸头顶约 `0.37 m`，调整头部位姿后需同步总高和重心检查。
- `class="duck"` 必须写在要应用该默认类的元素上；草案中 8 个主动关节、8 个 `position` 执行器和所有鸭子 geom 都要写，不能只写 `<default class="duck">`。
- 电机力矩上限已实现为 `forcerange="-20 20"`；`autolimits="true"` 下会自动启用 `forcelimited`，已验证生效。
- actuator 的定义顺序必须与关节/动作顺序一致；草案只展示一个 actuator，补齐其余 7 个时不要改变顺序。
- 没有把 `class="duck"` 加在 `torso` body 上，是为了避免 `freejoint` 也继承 `armature`、`damping` 和 `frictionloss`；若改用 body 级继承，需要给 `root` 显式清除这些属性。
- 模型的 `qpos0`（关节全 0）是直腿姿态，脚底最低点约 `-0.007 m`（轻微穿地），仅供加载/编译检查；运行时站姿契约以 `DuckEnvConfig.default_joint_pos` 与 `DuckEnv.reset()` 为准，不要用 `qpos0` 验证脚底接地。

## 9. 验收

模型加载、仿真稳定性、默认姿态和命名接口的验收项见[已归档的实施计划](archive/03-implementation-plan.md)阶段 B。
