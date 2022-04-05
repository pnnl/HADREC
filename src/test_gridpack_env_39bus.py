'''
This python file shows the basic usage of the gridpack environment to run
a dynamic simulation with/without load shedding actions


Authors: Renke Huang, Qiuhua Huang
Contact: qiuhua.huang@pnnl.gov


'''

import sys, os, time, parser, math
import numpy as np
import gym, ray
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
#import logz, optimizers, utils
from policy_LSTM import *
import gridpack
from GridPackPowerDynSimEnvDef_v4 import GridPackPowerDynSimEnv
from statistics import *

# the FAULT_CASES list defines a list of "fault tuples" for the dynamic simulation
# each "fault tuples" has the following four element (powerflowcaseidx, faultbusidx, faultstarttime, faultduration)
# power flow cases idx: the index of the power flow raw files defined in the simulation input xml file (see below "simu_input_file")
# fault bus idx: the index of the fault buses defined in the RL configuration file (see below "rl_config_file")
# fault start time, when the fault starts in the dynamic simualtion, e.g. 1.0 means fault starts at 1.0 seconds of the dynamic simu.
# fault duration time, the fault las time, e.g. 0.1 means the fault will last for 0.1 seconds and be cleared

FAULTBUS_CANDIDATES = [0,1,2,3,4,5,6,7,8]
FAULT_START_TIME = 1.0
FTD_CANDIDATES = [0.1] #[0.00, 0.05, 0.1]
FAULT_CASES = [(0, FAULTBUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) for i in range(len(FAULTBUS_CANDIDATES)) for j in range(len(FTD_CANDIDATES))]
print(FAULT_CASES)

folder_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
print('-----------------root path of the rlgc:', folder_dir)

'''
To run the dynamic simulation of load shedding with gridpack env, you need two input files:
dynamic simulation input file: this file has the basic inputs such as which power flow raw file and 
dynamic parameters dyr file to use for the dynamic simulation, as well as the observations definition, 
which tells the gridpack env that what specific observations the env needs to output, currently the env supports
the following observation types:
bus voltage mag and angle, generator speed and rotor angle, dynamic load percentage.
Please take a look at this xml file to better understand it.

RL configuration file: this file defines some important specifications for the grid control, 
such as the list of fault bus, list of load shedding bus, load shedding ranges for each bus, 
weights of the reward function, etc
Please take a look at this xml file to better understand it.
'''

simu_input_file = folder_dir + '/testData/IEEE39/input_39bus_step005_training_v33_newacloadperc43_multipf.xml'
rl_config_file = folder_dir + '/testData/IEEE39/json/IEEE39_RL_loadShedding_3motor_5ft_gp_lstm.json'

print ('!!!!!!!!!-----------------start the env')

'''
# first create a gridpack env, note the argument "force_symmetric_continuous_action" is set to be true,
# which means the defined env will only accept the load actions from the range "-1.0 to 1.0", and automatically
# convert the load action from the range [-1.0 to 1.0] to the load shedding range defined in the RL configuration file
'''
env = GridPackPowerDynSimEnv(simu_input_file, rl_config_file, force_symmetric_continuous_action=True)
#grab a fault tuple from FAULT_CASES
faulttuple = FAULT_CASES[0]

# the validate env function, will reinitialize the env's dynamic simulation by taking a specific fault tuple
obs = env.validate(case_Idx = faulttuple[0], fault_bus_idx = faulttuple[1],
                       fault_start_time=faulttuple[2], fault_duration_time=faulttuple[3])
# just check the shape of the observations
print(obs.shape)

# get the dimension of the observations and action buses
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.shape[0]

'''
# define the action for each action bus, note that 1.0 means no any load shedding,
# while -1.0 means shed 20% (-0.2) of the load, based on the RL configuration file
# and force_symmetric_continuous_action=True
'''
action_lst = [1.0 for i in range(ac_dim)]

# here we run one episode without any load shedding actions
obs_noact = list()
actions = list()
episode_rew = 0.0
rollout_length = 150
for cnt in range(rollout_length):
    obs_noact.append(obs)
    actions.append(action_lst)

    # simulate the env by taking the actions from action_lst to the next step
    # and get the observations and reward for this step
    obs, rew, done, _ = env.step(action_lst)

    episode_rew += rew

    # check whether this episode is done
    if done:
        break

print("------------------- Total steps: %d, Episode total reward without any load shedding actions: "%(cnt), episode_rew)

#---------------here we run another episode with the same fault, while load shedding actions are manually set at several time steps
ob = env.validate(case_Idx = 0, fault_bus_idx = faulttuple[1],
                       fault_start_time=faulttuple[2], fault_duration_time=faulttuple[3])
actions = []
obs = []
total_reward = 0.

fauttimestep = round(faulttuple[2]/env.env_time_step)
cnt = 0
for j in range(rollout_length):

    action_lst = [1.0 for i in range(ac_dim)]

    if 2 + fauttimestep <= cnt <= 6 + fauttimestep:
        action_lst = [1, -1, 1]  # note that -1 means shed 20% load at the correspondig bus, 1 means no load shedding

    if 7 + fauttimestep <= cnt <= 9 + fauttimestep:
        action_lst = [-1, 1, -1]

    if cnt == 12 + fauttimestep:
        action_lst = [1, 1, -1]

    obs.append(ob)
    # simulate the env by taking the actions from action_org to the next step
    # and get the observations and reward for this step
    ob, reward, done, _ = env.step(action_lst)

    actions.append(action_lst)

    total_reward += reward
    cnt += 1
    if done:
        break

print("------------------- Total steps: %d, Episode total reward with manually provided load shedding actions: "%(cnt), total_reward)

#-----------------plot the observations------------------------
#-------------first define the voltage recovery envelope, assuming the time step is 0.1 sec

volt_lim = []
for i in range(0, 400):
    volt_lim.append(0.94)
    
for i in range(50,55):
    volt_lim[i]=0.0
    
for i in range(55,72):
    volt_lim[i]=0.7    
    
for i in range(72,80):
    volt_lim[i]=0.8

for i in range(80,131):
    volt_lim[i]=0.9
    
plotsimutime = []

for i in range(0, 400):
    plotsimutime.append(i*0.02)
    
simutime = []
for i in range(0, 80):
    simutime.append(i*0.1)

obs_noact_arr = np.array(obs_noact)
obs_arr = np.array(obs)

volt_ob_noact = obs_noact_arr[:,:(ob_dim-ac_dim)]
load_ob_noact = obs_noact_arr[:,-ac_dim:]

volt_ob = obs_arr[:,:(ob_dim-ac_dim)]
load_ob = obs_arr[:,-ac_dim:]

# plot the bus voltage observations
print ('volt_ob_noact.shape: ', volt_ob_noact.shape)
nstep = volt_ob_noact.shape[0]
plt.plot(simutime[0:nstep], volt_ob_noact[0:nstep,:])
plt.plot(plotsimutime[:nstep*5], volt_lim[:nstep*5], 'k--')
plt.title('voltages without any load shedding actions')
plt.xlabel('time sec')
plt.ylabel('voltage (p.u.)')
plt.show()

# plot the bus voltage observations
nstep = volt_ob.shape[0]
plt.plot(simutime[0:nstep], volt_ob[0:nstep,:])
plt.plot(plotsimutime[:nstep*5], volt_lim[:nstep*5], 'k--')
plt.title('voltages with mannully set load shedding actions')
plt.xlabel('time sec')
plt.ylabel('voltage (p.u.)')
plt.show()

# plot the remaining load observations for the AI load shedding case
plt.plot(simutime[0:nstep], load_ob[0:nstep,:])
plt.title('remaining load with mannully set load shedding actions')
plt.xlabel('time sec')
plt.ylabel('remaining load (percentage)')
plt.show()

#----------remember to de-allocate the env
env.close_env()

print ('!!!!!!!!!-----------------finished gridpack env testing')