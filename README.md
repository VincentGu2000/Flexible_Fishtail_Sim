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

