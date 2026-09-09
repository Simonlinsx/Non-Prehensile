# FR3 + 原装闭合夹爪：仿真对齐记录

最终硬件按用户确认使用 FR3 + 原装闭合夹爪。此前 Panda 结果仅作诊断；正式随机50场已于2026-09-09 07:59:42 UTC启动，门槛为至少21场成功（严格>40%）、全场零C1违规。下文保留对齐过程，当前执行以末尾正式批量状态和SIMULATION_ACCEPTANCE_GOAL.md为准。

## 模型与原版 OSC

官方来源：https://github.com/frankarobotics/franka_description ，固定提交 `7aeeddc449edf8d62b594f9e36a81da53e7796f9`，选择 `fr3` 和 `franka_hand`。文件及许可在 `data/robot_models/fr3_stock_20260909/vendor/franka_description`，展开参数在 `upstream_provenance.json`。

`scripts/prepare_fr3_robot_models.py` 从同一份官方 URDF 生成 Isaac 与 Drake 模型。保留官方几何、质量、COM、惯量和限位；移除没有质量、几何的固定坐标帧时，将其变换合并到子关节。内部 `panda_*` 名称仅用于兼容既有消息、传感器与控制代码，物理参数来自 FR3。

Isaac 保留可动手指并命令闭合，Drake 将手指固定在零开度。共同任务点为手掌局部 Z +0.1034 m。Drake 任务坐标相对手掌旋转 Rz(pi/4) Rx(pi)，使原版 OSC 的任务姿态目标与闭合手掌一致。

附加转子惯量由官方扩展字段 `gear_ratio² * motor_inertia` 得到。Drake 通过 transmission 读取，Isaac 显式配置 armature。保留每关节 0.003 Nm·s/rad 黏性阻尼；Drake 不支持这里的 URDF 库仑摩擦，名义对齐基线在 Isaac 也显式关闭此项。官方参数是名义模型，尚未进行真机辨识。

原版 OSC 的优化控制与增益未改动。补丁 0024 仅增加可选完整机械臂模型入口，0026 仅增加仿真专用 OSC 状态通道入口。启动器通过 `PUSH_ANYTHING_ROBOT_MODEL_MANIFEST` 校验 URDF 与网格哈希，向仿真和原版控制器传递配对模型。

## 实际导入：刚体参数通过；摩擦写回问题已定位

证据：`outputs/contact_planner_m3/workspace_height_20260908/fr3_native_osc_hold2/model_alignment_audit.json`，原始结果在同目录 `result.json`，保留本次执行源码、运行参数与二进制。

| 核验项 | 最大绝对差 |
|---|---:|
| 刚体质量 | 9.35e-8 kg |
| 刚体 COM | 3.43e-9 m |
| 刚体惯量 | 1.63e-8 kg·m² |
| 重力补偿 | 1.06e-5 Nm |
| 含反射惯量的关节质量矩阵 | 7.25e-7 kg·m² |
| 末端位置 FK | 1.08e-7 m |
| 末端姿态 FK | 2.70e-7 rad |

2 秒静止保持：2000/2000 条新鲜扭矩指令，0 超时，最大/最终 TCP 位移 0.2267 mm。**随后推动测试发现引擎实际摩擦仍为 0.2，所以该保持结果不能证明完整动态跟踪已对齐。** 原先 Panda 与原装夹爪/推杆混合模型相同长度测试约 9.006 mm；这只是历史诊断对比，不能将改善全部归因于质量模型，也不能证明推动任务成功。

PhysX `get_inertias()` 已返回 COM 处、刚体轴系表达的惯量；不能再乘 `get_coms()` 的主惯量轴旋转。参见 [NVIDIA Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_inertias)。PhysX 广义质量矩阵需加实际 armature 对角阵后与 Drake 比较。

## 接触几何和控制时序：继续验证

官方 FR3 每根手指有四个盒体。`data/contact_models/fr3_stock_hammer_physx_20260909_v2.json` 从真实导入的 USD/PhysX 导出八个指面盒体和目标的 16 个烹制凸包。补丁 0025 保留两个手指刚体，每指注册四个独立形状，并让接触配对、采样间隙计算遍历全部形状。没有用整体凸包填满形状间的空隙。

C1 的 FR3 点云从实际模型生成，并随手掌、两手指各自的实测刚体姿态变化。旧 Panda 点云和球形推杆距离模型不能用于 FR3 最终验收。

新增可选 `PUSH_ANYTHING_PLANNER_PERIOD_MS=50`：20 Hz C3+ 状态通道与 1 kHz OSC 状态通道分离。除启动首帧外，仿真等待当前规划完成后才执行该时刻的 OSC 扭矩；之后的 49 个物理步由原版 OSC 连续跟踪。分频保持测试 `fr3_native_osc_hold2_clock20_v2` 已通过：2 秒恰好 40 次规划、2000 次新鲜扭矩，末端漂移仍为 0.2267 mm。证据见同目录 `clock_audit.json` 和 `model_alignment_audit.json`。

早期 `fr3_native_osc_push10_scene007` 为 10 秒原版 C3+/OSC 闭环推动调试，现已结束；保持诊断关闭，不计入正式 50 场。下一步审计接触与固定目标，再扩展 FR3 调试验证并冻结正式验收版本。扭矩执行器仍标记 `acceptance_eligible=false`，固定目标、独立姿态保持与接触审计尚未全部接齐。不连接真机。

## 推动测试暴露的摩擦写回缺失

`fr3_native_osc_push10_scene007` 完成 10 秒，未建立合法接触，XY 误差约59.93mm。只读 OSC 调试显示已有约 2.5m/s² 期望/求解加速度，但实测末端速度接近零。实际关节摩擦读回0.2。

本地 `/data1/linsixu/IsaacLab-2.2.0/source/isaaclab/isaaclab/assets/articulation/articulation.py` 的 Sim5 分支，三个关节摩擦 setter 只修改 `get_dof_friction_properties()` 的缓存，缺少对应 `set_dof_friction_properties()` 调用。配置值不能代替引擎读回。

仓库内新增 `fr3_model_runtime.synchronize_arm_friction`，在重置之后通过 PhysX 后端提交旧/新接口的零摩擦合同，并强制读回核验，保留手指自身属性。未修改外部 IsaacLab 安装。`fr3_native_osc_push10_friction_commit` 已完成：后端旧/新摩擦读回均为零，2.592秒首次合法接触，最终XY误差48.5736mm、SO(3)误差0.398616rad，全程零C1违规，尚未成功。目前运行相同配置的 `fr3_native_osc_scene007_180`。模型审计已追加实际摩擦、零隐藏刚度和匹配阻尼检查。

另一个已验证的改进是 TCP 快速确认：`fr3_native_osc_hold2_quickack/transport_comparison.json` 中，2000步的全部记录扭矩、关节状态和物体轨迹与计时基线逐项相同，墙钟约153秒降为40秒。墙钟比值受系统负载影响；通信部分从约97秒降到11秒。

读回补充：旧式负载相关摩擦系数在修复前是0.2，新式静/动摩擦力和黏性字段修复前已为零。因此关键修复还包括显式清除旧式摩擦参数，不能把0.2误称为新式静摩擦力。早期已通过的动力学审计不覆盖全部夹爪限位；随后补齐如下核验。

最新目标：冻结50场，严格>40%（至少21/50）、全场零C1违规；验收完成后整理仓库并提交推送。正式50场尚未启动，仓库未提交/推送。


## 手指限位后端核验与修复

导入后的指关节2硬限位是[-0.008,0.048]m，早期速度上限还残留浮点最大值。FR3 手部执行器现在显式配置官方速度0.2m/s、力100N；重置后通过 Articulation writer 将两指硬限位恢复为[0,0.04]m，同步 IsaacLab 缓存和 PhysX，并读回三类限位。保持导入的手指耦合和闭合驱动，不修改外部安装。

`fr3_native_osc_hold2_finger_limits/model_alignment_audit.json` 已通过包含上述限位的扩展审计。2秒保持共2000条新鲜扭矩、无超时、无C1违规，TCP最大位移0.01688mm、最终位移0.01258mm。该结果使用实际零臂摩擦；它验证静态保持和名义参数，不能证明推动接触动力学已匹配。相关后端/时序/协议测试18项通过。

`fr3_native_osc_scene007_180` 继续保留旧手指限位版本的长时诊断，不作为最终模型验收。约25秒只读OSC消息显示升高重定位时实际速度接近期望0.18m/s，物体却在首次接触后停滞；规划器反复切换推动与重新定位，尚未证明具体原因。

新启动 `fr3_native_osc_push15_finger_limits` 使用官方限位版本，并在 effort 模式启用原有只读物体预测、优化器和完整末端轨迹审计，用于对比预测效果与实测效果。控制消息仍由原版OSC提供。正式50场依然为0场，不能宣称仿真已全部通过。


## 动态闭合仍需核验：导入的软联动

`fr3_native_osc_push15_finger_limits` 已完成15秒：15000条新鲜扭矩、零C1违规，最终XY32.214mm、高度0.987mm、SO(3)0.085335rad。独立重算与报告一致；三项未同时满足并保持0.5秒，仍失败。初步75ms物体效果审计保存为`object_effect_audit.json`，跨越50ms重规划，不能单独归因于接触模型。

逐步开度发现指2曾达到9.5296mm，指1最大0.7186mm，不能用静态模型审计通过替代动态闭合核验。峰值发生在2.401秒，尚未接触目标。只读USD检查`imported_finger_joint_audit.json`发现导入器在原本q2=q1关系上加入naturalFrequency=25、dampingRatio=0.005的软联动。

根据[NVIDIA PhysX5.6.1联动文档](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)，零频率/阻尼表示硬联动。候选修复在实际场景生成后、物理初始化前将这两项置零，保留gearing=-1、offset=0和原参考关节，不改共享缓存USD；`fr3_native_osc_push3_rigid_mimic`正在进行3秒对照，尚未证实修复效果。已有两个180秒版本都保留软联动，因此仅作诊断，不能计入正式验收。


硬联动3秒对照现已完成：`fr3_native_osc_push3_rigid_mimic/finger_coupling_audit.json` 记录每物理步开度，最大指1/指2开度0.4214/0.4199mm，最大不同步0.2140mm；3000条新鲜扭矩、合法接触、零C1违规。静态模型/实际摩擦/驱动/限位扩展审计也通过。硬联动是保留URDF名义同步关系，尚未辨识真机联动柔度。

已保存两个旧软联动180秒版本的中止说明和现有日志，向各自已核验的Isaac子进程发送SIGINT，核验旧执行器/控制子进程退出；不将中止试验伪装为完整失败或成功场景。新运行`fr3_native_osc_scene007_180_rigid_mimic`（GPU0、端口10570/10571），继续检查长时实际闭合与共同位姿收敛。正式随机50场仍未启动。


## 首个严格FR3调试成功与正式批量准备

`fr3_native_osc_scene007_180_rigid_mimic` 在14.182秒停止并成功。最终XY18.7537mm、高度0.9878mm、完整SO(3)0.084820rad；独立`strict_pose_dwell_audit.json`确认最后连续500个1ms物理步都满足阈值。真实合法手部接触，14182条新鲜扭矩，无过期、看门狗或C1违规。静态模型/摩擦/驱动/限位审计再次通过。`terminal_summary.json`保留哈希和范围；这是调试scene007，不能计入正式50场。

新增`result.finger_state.jsonl`逐物理步记录实际指位置和指位置目标；`audit_fr3_finger_closure.py`检查完整、有序、精确时戳、零闭合目标，并在指定容差下重算开度和不同步。名义闭合检查暂采用每指/不同步1mm，正式冻结时需记录这一容差。3秒核验全3000步通过，且新增记录前后的控制指令、关节和物体状态逐项相同。`fr3_native_osc_scene007_180_closure_audit`正在重复成功场景以取得完整推动期间的密集闭合证据，不用于选择最佳正式重跑。

`prepare_fr3_osc_scenes.py`已准备`fr3_osc_acceptance50_prepared`的50个固定随机场景并逐一核对初始姿态、Isaac目标和原版目标的坐标映射。只允许q_init_objects、固定目标位姿、预先指定的sampling_seed变化。`fr3_runtime_inputs.py`遍历参数引用和URDF/SDF网格，每场枚举60项输入；去除上述场景变量后，50场控制参数和接触几何签名一致。

正式50场还未启动。仍需冻结测试过的执行源码、两个原版二进制、机器人/物体输入和这些运行时依赖，并完成原版OSC批量执行与审计入口。旧正式审计只支持task执行器，不能简单将原版OSC结果改名来通过。验收目标维持至少21/50、全50场零C1；达标后整理并推送仓库。


完整密集闭合复测现已结束并通过：仍14.182秒严格成功，全部14182个物理步的开度/指令记录完整，零非零闭合目标、零1mm闭合容差违规，最大开度0.4214mm、最大不同步0.2140mm。1899条完整轨迹记录与前次成功逐项相同；额外审计未改变本次控制/物理轨迹。

已冻结`fr3_osc_acceptance50_prepared/controller_freeze.json`，SHA256 `70b5474bc3dfd373eb83dc9a8fe034935c03bb65e1e5e4b6100bd3a47a94b34d`：196个执行源文件、原版规划器与OSC二进制、54个外部模型/资产/原生库输入（另存内容副本）、50场各自60项运行时依赖；场景引用文件已实体化到各自目录。`freeze_integrity_audit.json`核验原文件与归档哈希一致。保留实际执行的冻结脚本和Python包列表。原生runfiles与Isaac环境仍依赖本机，不能宣称这是可搬移的独立容器镜像。

本任务本轮启动的诊断仿真均已结束；下一项工作是批量worker和独立批量审计接入该冻结合同，然后执行正式50场。冻结后不能按各场结果调参或选最佳重跑；调试成功不计入分子。


## 正式50场已启动

2026-09-09 07:59:42 UTC，以固定index%4分组在GPU2/3/5/6启动正式场景；tmux会话`c1_fr3_formal50_0`至`_3`，首批为scene000–003，后续各worker按原清单顺序执行。启动前50场输入核验通过，并已核实四个worker和四个Isaac进程实际存活。冻结控制版本SHA不变。

新增`run_fr3_osc_batch_worker.py`每场核验冻结来源、清除继承的实验控制开关、只执行一次并保留启动/结束及返回码。`audit_fr3_osc_batch.py`直接核验原版扭矩schema，检查同一FR3模型、固定目标及实际场景坐标、扭矩新鲜度、完整逐步闭合、真实合法接触、独立500步位姿保持和C1结果；没有通过重命名task结果来验收。六项新增/相关测试通过；已知真实成功结果通过执行合同检查。

批量worker与审计工具单独存档在`verification_source`，哈希清单为`verification_source_manifest.json`。每次验收报告只声明`simulation_acceptance_pass`；仓库整理/提交/推送仍是达成完整goal的后续必要工作。启动时0场完成、0场正式成功，不能将调试成功计入。
