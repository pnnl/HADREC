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
from GridPackPowerDynSimEnvDef_v10 import GridPackPowerDynSimEnv
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
#simu_input_file = folder_dir + '/testData/tamu2000/input_tamu2000_with230and500asob_noloadob_2acloads_v33.xml'
#rl_config_file = folder_dir + '/testData/tamu2000/json/RLGC_RL_tamu2000_loadShedding_500kV_faultbus.json'

simu_input_file1 = folder_dir + '/testData/IEEE300/input_ieee300_genrou_exdc1_tgov1_zone1ac.xml'

rl_config_file = folder_dir + '/testData/IEEE300/json/IEEE300_RL_loadShedding_zone1_continuous_LSTM_multipfcases_3ftdur_training_gp.json'

print ('!!!!!!!!!-----------------start the env')

'''
# first create a gridpack env, note the argument "force_symmetric_continuous_action" is set to be true,
# which means the defined env will only accept the load actions from the range "-1.0 to 1.0", and automatically
# convert the load action from the range [-1.0 to 1.0] to the load shedding range defined in the RL configuration file
'''
# define two different env with differen simu_input_files, one with ac motor only in zone 1, the other with ac motor at all three zones
env1 = GridPackPowerDynSimEnv(simu_input_file1, rl_config_file, force_symmetric_continuous_action=True)

#grab a fault tuple from FAULT_CASES
#faulttuple = FAULT_CASES[0]


# define function to run one episode testing with the trained LSTM
def one_episode_ai(env, fault_tuple, rollout_length=90):
    print("Fault tuple is ", fault_tuple)

    total_reward = 0.
    steps = 0
    ars_time = 0.0

    ob = env.validate(case_Idx = fault_tuple[0], fault_bus_idx = fault_tuple[1], \
                   fault_start_time=fault_tuple[2], fault_duration_time=fault_tuple[3])
    act_lst = []
    ob_lst = []
    
    policy.reset()
    
    for j in range(rollout_length):
        t1 = time.time()
        action_org = policy.act((ob - ob_mean) / (ob_std + 1e-8))
        t2 = time.time()
        ars_time += (t2-t1)
        
        ob_lst.append(ob)

        ob, reward, done, _ = env.step(action_org)
        act_lst.append(action_org)
        
        steps += 1
        total_reward += reward
        if done:
            break
    
    print ('-----one episode testing finished without AI-provided actions, total steps: %d total reward: '%(steps), total_reward)
    
    return {'ob_lst': np.array(ob_lst) , 'act_lst': np.array(act_lst), 'total_reward': total_reward}
	
# define function to run one episode without any load actions
def one_episode_noact(env, fault_tuple, rollout_length=90):
    print("Fault tuple is ", fault_tuple)

    total_reward = 0.
    steps = 0
    ars_time = 0.0

    ob = env.validate(case_Idx = fault_tuple[0], fault_bus_idx = fault_tuple[1], \
                   fault_start_time=fault_tuple[2], fault_duration_time=fault_tuple[3])
    act_lst = []
    ob_lst = []

    ac_dim = env.action_space.shape[0]
    action_lst = [1.0 for i in range(ac_dim)]
    for j in range(rollout_length):
        
        ob_lst.append(ob)

        ob, reward, done, _ = env.step(action_lst)
        act_lst.append(action_lst)
        
        steps += 1
        total_reward += reward
        if done:
            break
    
    print ('-----one episode testing finished without any load shedding, total steps: %d total reward: '%(steps), total_reward)
    
    return {'ob_lst': np.array(ob_lst) , 'act_lst': np.array(act_lst), 'total_reward': total_reward}

policyfile = "rk_outputs_training/ars_300_bussys_1_pf_8_faultbus_1_dur_lstm_gridpack_v1/nonlinear_policy_plus215.npz"

lstm_policy_org = np.load(policyfile, allow_pickle = True)
lstm_policy = lstm_policy_org['arr_0']

w_M = lstm_policy[0] # weights as a numpy array
ob_mean = lstm_policy[1]  # observation mean

ob_std = lstm_policy[2]  # observation standard deviation

print ('finished loading npz file')

# get the dimension of the observations and action buses
ob_dim = env1.observation_space.shape[0]
ac_dim = env1.action_space.shape[0]

print ('ob_dim: ', ob_dim, 'ac_dim: ', ac_dim)

policy_params={'type':'LSTM',
               'ob_filter':"MeanStdFilter",
               'ob_dim':ob_dim,
               'ac_dim':ac_dim, 
               'policy_network_size':[32, 32],
               'weights':w_M}

policy = LSTMPolicy(policy_params)
policy.update_weights(w_M)

print ('finished loading the weights to the policy network')


# here we run one episode without for the case only zone 1 has ac motors with AI provided actions

fault_tuple_test = (0, 0, 1.0, 0.08)
results1 = one_episode_ai(env=env1, fault_tuple=fault_tuple_test)

print("------------------- Episode total reward with AI provided actions: ", results1['total_reward'])

#---------------here we run another episode without any actions
fault_tuple_test = (0, 0, 1.0, 0.08)
results2 = one_episode_noact(env=env1, fault_tuple=fault_tuple_test)

print("------------------- Episode total reward without any action: ", results2['total_reward'])

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


'''
Note, here the observations at each time step have two parts: (1) bus voltages and (2) remaining load percentage for 
the buses where load shedding actions could be taken. The dimension for the bus voltage related observations could be 
obtained by env.nobbus, while the dimension for the remaining load related observations could be 
obtained by env.nobload
The total dimension of the observations of one timestep is equal to env.observation_space.shape[0] = env.nobbus + env.nobload
'''

# plot the bus voltage observations
plt.rcParams.update({'font.size': 17})

plt.subplot(121)
nstep = results1['ob_lst'].shape[0] # get how many steps this simulation is run to
plt.plot(simutime[0:nstep], results1['ob_lst'][0:nstep, :(ob_dim-ac_dim)])
plt.plot(plotsimutime[:nstep*5], volt_lim[:nstep*5], 'k--')
plt.title('voltages without AI-provided actions')
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (p.u.)')
#plt.tight_layout()

plt.subplot(122)
nstep = results2['ob_lst'].shape[0] # get how many steps this simulation is run to
plt.plot(simutime[0:nstep], results2['ob_lst'][0:nstep, :(ob_dim-ac_dim)])
plt.plot(plotsimutime[:nstep*5], volt_lim[:nstep*5], 'k--')
plt.title('voltages without any actions')
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (p.u.)')
#plt.tight_layout()
plt.show()

plt.subplot(121)
nstep = results1['ob_lst'].shape[0] # get how many steps this simulation is run to
plt.plot(simutime[0:nstep], results1['ob_lst'][0:nstep, :(ob_dim-ac_dim)])
plt.plot(plotsimutime[:nstep*5], volt_lim[:nstep*5], 'k--')
plt.title('voltages without AI-provided actions')
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (p.u.)')
#plt.tight_layout()

plt.subplot(122)
nstep = results2['ob_lst'].shape[0] # get how many steps this simulation is run to
plt.plot(simutime[0:nstep], results1['ob_lst'][0:nstep, -ac_dim:])
plt.title('remaining load with AI-provided actions')
plt.xlabel('Time (sec)')
plt.ylabel('remaining load (percentage)')
#plt.tight_layout()
plt.show()

#----------remember to de-allocate the env
env1.close_env()

print ('!!!!!!!!!-----------------finished gridpack env testing')