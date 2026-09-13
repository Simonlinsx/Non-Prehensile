# 相对布局与目标角度留出测试

冻结控制器与权重后生成 20 个新任务，hard=0 与 soft=0.5 m 共 40 次物理运行。所有尝试保留，无结果驱动调参或替换。

**这是已知锤子及 4 类已知杂物组合内的新布局／目标泛化，不是新物体类别泛化，也不是随机任意桌面任务的成功率。**

各障碍物独立平移 ±3 cm、绕世界 Z 轴旋转 ±25°；目标绕世界 Z 轴相对初始角度为 −30°、−15°、20°、35°、50°，移动距离 5–8 cm，并改变移动方向。原控制频率、物理参数、预算和严格成功条件保持不变。严格标准还沿用历史的至少一次 >0.5 N 安全部位机器人—目标接触证据；这和控制器的 0.02 N 接触门槛不同。

几何准入仅要求名义起点和目标对禁碰物体保留 12 mm、杂物间保留 1 mm；允许物体与目标的可接受接触。不使用控制器、候选代价或仿真成功结果筛选。20 场从 50 次几何抽样中取得，所有拒绝记录在 geometric_admission.json。几何准入不保证机械臂可执行。

| 布局 | hard 严格成功 | soft 严格成功 |
|---|---:|---:|
| 侧方布娃娃 | 2/5 | 1/5 |
| 前方布娃娃 | 1/5 | 1/5 |
| 可选接触小布娃娃 | 1/5 | 1/5 |
| 碗／海绵 | 1/5 | 1/5 |
| 合计 | 5/20 | 4/20 |

| 目标相对角度 | hard | soft |
|---|---:|---:|
| -30° | 1/4 | 1/4 |
| -15° | 3/4 | 2/4 |
| 20° | 0/4 | 0/4 |
| 35° | 0/4 | 0/4 |
| 50° | 1/4 | 1/4 |

角度分组同时包含不同布局、目标位置与移动方向，不能把差异单独归因于角度。

只有 soft 成功：[]；只有 hard 成功：['001']。

C1 违规场数 hard/soft：0/0；禁碰接触违规场数：0/0。

## 补充：到位与接触证据分开

保持位姿持续时间、安全和夹爪条件，但采用已有的 >0.02 N 安全接触证据，hard/soft 分别为 5/20、5/20。其中仅因历史 0.5 N 接触证据门槛不满足而失败：0 / 1 场。这是诊断性补充，不替换冻结的严格成功率。

## 失败逐场保留

| 组 | 场景 | 目标角度 | 最终 XY mm | 最终 SO(3) rad | 到位但仅缺0.5N证据 | 停止原因 |
|---|---|---:|---:|---:|---|---|
| hard | 000 | -30 | 6.84 | 0.5012 | False | simulation_budget |
| soft | 000 | -30 | 6.84 | 0.5012 | False | simulation_budget |
| soft | 001 | -15 | 19.88 | 0.0793 | True | planner_finished |
| hard | 002 | 20 | 25.98 | 0.2523 | False | planner_finished |
| soft | 002 | 20 | 25.98 | 0.2523 | False | planner_finished |
| soft | 003 | 35 | 42.41 | 0.5671 | False | planner_finished |
| hard | 003 | 35 | 42.41 | 0.5671 | False | planner_finished |
| soft | 005 | -30 | 22.61 | 0.0474 | False | simulation_budget |
| hard | 005 | -30 | 22.07 | 0.0104 | False | simulation_budget |
| soft | 007 | 20 | 44.87 | 0.4021 | False | simulation_budget |
| hard | 007 | 20 | 44.87 | 0.4021 | False | simulation_budget |
| hard | 008 | 35 | 20.74 | 0.2996 | False | simulation_budget |
| soft | 008 | 35 | 20.74 | 0.2996 | False | simulation_budget |
| soft | 009 | 50 | 69.98 | 0.8729 | False | planner_finished |
| hard | 009 | 50 | 69.98 | 0.8729 | False | planner_finished |
| hard | 010 | -30 | 6.15 | 0.3257 | False | simulation_budget |
| soft | 010 | -30 | 6.15 | 0.3257 | False | simulation_budget |
| hard | 012 | 20 | 12.37 | 0.3327 | False | planner_finished |
| soft | 012 | 20 | 12.37 | 0.3327 | False | planner_finished |
| soft | 013 | 35 | 79.86 | 0.6010 | False | planner_finished |
| hard | 013 | 35 | 79.86 | 0.6010 | False | planner_finished |
| hard | 014 | 50 | 32.66 | 0.7741 | False | planner_finished |
| soft | 014 | 50 | 32.66 | 0.7741 | False | planner_finished |
| hard | 016 | -15 | 20.78 | 0.3014 | False | simulation_budget |
| soft | 016 | -15 | 20.78 | 0.3014 | False | simulation_budget |
| soft | 017 | 20 | 54.98 | 0.3494 | False | planner_finished |
| hard | 017 | 20 | 54.98 | 0.3494 | False | planner_finished |
| hard | 018 | 35 | 21.65 | 0.6811 | False | simulation_budget |
| soft | 018 | 35 | 21.65 | 0.6811 | False | simulation_budget |
| soft | 019 | 50 | 49.95 | 0.8729 | False | planner_finished |
| hard | 019 | 50 | 49.95 | 0.8729 | False | planner_finished |

## 接触与耗时

接触下降也可能由失败、早停或推动不足导致，不能将全体接触减少直接解释为安全收益。应同时查看同一对的成功、最终误差与执行时长。原违规阈值 20 mN 不变；接触时长统计阈值为 2 mN，累计强度是力模长积分。

| 指标 | soft 降低 | 相同 | 升高 |
|---|---:|---:|---:|
| acceptable_target_force_peak_n | 0 | 19 | 1 |
| acceptable_contact_gt2mN_s | 0 | 19 | 1 |
| acceptable_force_norm_integral_ns | 0 | 19 | 1 |

| 录像 | 组 | 场数 | 完整进程中位数 s | 仿真中位数 s |
|---|---|---:|---:|---:|
| False | hard | 16 | 118.23 | 28.81 |
| True | hard | 4 | 390.39 | 60.00 |
| False | soft | 16 | 114.69 | 28.20 |
| True | soft | 4 | 385.59 | 60.00 |

同一对在同一 GPU 上依次执行，按场景编号交替先跑哪一组；8 GPU 并发，墙钟受共享 CPU／渲染负载影响。录像与不录像单列。

## 视频

预先指定每个布局第一场录像，没有按结果挑选；视频本身的成功与失败以本报告为准。

- 000：[hard](hard/000/native_result_video/native_result-step-0.mp4) · [soft](soft/000/native_result_video/native_result-step-0.mp4)
- 005：[hard](hard/005/native_result_video/native_result-step-0.mp4) · [soft](soft/005/native_result_video/native_result-step-0.mp4)
- 010：[hard](hard/010/native_result_video/native_result-step-0.mp4) · [soft](soft/010/native_result_video/native_result-step-0.mp4)
- 015：[hard](hard/015/native_result_video/native_result-step-0.mp4) · [soft](soft/015/native_result_video/native_result-step-0.mp4)

语义仍由助手审阅与缓存；几何为仿真真值；刚体代理不验证液体、柔性或爆破风险。逐场 CSV、完整审计、几何抽样与协议均在本目录。
