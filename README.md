# Flexible_Fishtail_Sim

Simulation of flexible fishtail and its deformation control.

## 1 Environment

MATLAB R2022b

## 2 File description

Agent_DDPG.mat                     ---    Trained DDPG agent.

Flexible_Fishtail_CTLSys.slx       ---    Simulation of deformation control framework for flexible fishtail with observer.

Flexible_Fishtail_CTLSys_RL.slx    ---    Simulation for deep reinforcement learning training, with the observer removed.

RL_CTL_LW_DDPG.m                   ---    A training program for deep reinforcement learning.

## 3 Operating instruction

### 3.1 Training
Open RL_CTL_LW_DDPG.m with MATLAB and click Run.

### 3.2 Simulation analysis
Open Flexible_Fishtail_CTLSys.slx in MATLAB Simulink, double-click in MATLAB working directory to load Agent_DDPG.mat or input code load(' agent_DDpg.mat '), then click Run in simulink.

## 4 Contact information

Author:  Junwen Gu (顾俊文)

Email:   gujunwen2022@ia.ac.cn


If you find this simulation useful in your research, please cite:

J. Gu, J. Wang, Z. Liu, M. Tan, J. Yu and Z. Wu, "Deformation Control and Thrust Analysis of a Flexible Fishtail With Muscle-Like Actuation," in IEEE Transactions on Robotics, vol. 41, pp. 159-179, 2025.

```
@article{gu_deformation_2025,
  author = {Gu, Junwen and Wang, Jian and Liu, Zhijie and Tan, Min and Yu, Junzhi and Wu, Zhengxing},
  journal = {IEEE Transactions on Robotics},
  title={Deformation Control and Thrust Analysis of a Flexible Fishtail With Muscle-Like Actuation},
  volume = {41},
  pages={159-179},
  year = {2025},
  doi = {10.1109/TRO.2024.3502203}
}
```
