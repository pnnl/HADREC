'''

Authors: Renke Huang, Qiuhua Huang
Contact: qiuhua.huang@pnnl.gov

'''

import sys, os, time, parser, math
import numpy as np
import gym, ray
import logz, optimizers, utils
from obserfilter import RunningStat
from policy_LSTM import *
import socket
from shared_noise import *
from GridPackPowerDynSimEnvDef_v2 import GridPackPowerDynSimEnv
from statistics import *

folder_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
print('-----------------root path of the rlgc:', folder_dir)

FAULT_BUS_CANDIDATES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
FAULT_START_TIME = 1.0
FTD_CANDIDATES = [0.00, 0.05, 0.1]
FAULT_CASES = [(FAULT_BUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) for i in range(len(FAULT_BUS_CANDIDATES))
               for j in range(len(FTD_CANDIDATES))]

GKE = True # whether the training is running on GKE
EIOC = True # set this to be true if running on GKE

b_debug_timing = True
NUM_CORES = 18

OB_DIM = 301
AC_DIM = 81

if GKE:
    simu_input_file = folder_dir + '/testData/tamu2000/input_tamu2000_step005_rmgensmall30_withob_2acloads_v33_nocsv_superlu.xml'
else:
    simu_input_file = folder_dir + '/testData/tamu2000/input_tamu2000_step005_rmgensmall30_withob_2acloads_v33_nocsv_eioc.xml'
	
rl_config_file = folder_dir + '/testData/tamu2000/json/RLGC_RL_tamu2000_loadShedding_zone3_gp.json'


# ======================================================================

@ray.remote
class SingleRolloutWorker(object):
    '''
    responsible for doing one single rollout in the env
    '''

    def __init__(self, rollout_length, policy_params):

        print('\n\n\n -----------------------Set Env=GridPackPowerDynSimEnv------------------------\n\n\n')
        self.env = GridPackPowerDynSimEnv(simu_input_file, rl_config_file, force_symmetric_continuous_action=True)
        print('\n\n\n ----------------------------Set Env Done-----------------------------\n\n\n')
        self.rollout_length = rollout_length
        self.policy_type = policy_params['type']
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'nonlinear':
            self.policy = FullyConnectedNeuralNetworkPolicy(policy_params)
        elif policy_params['type'] == 'LSTM':
            self.policy = LSTMPolicy(policy_params)
        else:
            raise NotImplementedError
        time.sleep(5)

    def single_rollout(self, fault_tuple, weights, ob_mean, ob_std):
        # one SingleRolloutWorker is only doing one fault case in an iter
        # fault_tuple = FAULT_CASES[fault_case_id]
        total_reward = 0.
        steps = 0
        self.policy.update_weights(weights)

        # us RS for collection of observation states; restart every rollout
        self.RS = RunningStat(shape=(OB_DIM,))

        t1 = time.time()
        ob = self.env.validate(case_Idx=0, fault_bus_idx=fault_tuple[0],
                               fault_start_time=fault_tuple[1], fault_duration_time=fault_tuple[2])
        t_validate = time.time() - t1

        if self.policy_type == 'LSTM':
            self.policy.reset()

        t_act = 0
        t_step = 0
        for _ in range(self.rollout_length):
            ob = np.asarray(ob, dtype=np.float64)
            self.RS.push(ob)
            normal_ob = (ob - ob_mean) / (ob_std + 1e-8)
            t3 = time.time()
            action_org = self.policy.act(normal_ob)
            t4 = time.time()
            ob, reward, done, _ = self.env.step(action_org)
            t5 = time.time()
            t_act += t4 - t3
            t_step += t5 - t4
            total_reward += reward
            steps += 1
            if done:
                break
        t2 = time.time()

        if b_debug_timing:
            return {'reward': total_reward, 'step': steps, 'time': [{'Total Time': t2 - t1},
                                                                    {'Reset Time': t_validate},
                                                                    {'Action Time': t_act},
                                                                    {'Step Time': t_step} ]}
        else:
            return {'reward': total_reward, 'step': steps, 'time': [t2 - t1, t_validate, t_act, t_step]}

    def return_RS(self):
        return self.RS

    def close_env(self):
        self.env.close_env()
        return

def run_ars_test(params):

    # set policy parameters.
    if params['policy_file'] != "":
        nonlin_policy_org = np.load(params['policy_file'], allow_pickle=True)
        nonlin_policy = nonlin_policy_org['arr_0']

        w_M = nonlin_policy[0].copy()
        w_mean = nonlin_policy[1].copy()
        w_std = nonlin_policy[2].copy()

        policy_params = {'type': params['policy_type'],
                         'policy_network_size': params['policy_network_size'],
                         'ob_dim': OB_DIM,
                         'ac_dim': AC_DIM,
                         'weights': w_M,
                         'w_mean': w_mean,
                         'w_std': w_std,
                         }
    else:
        # set policy parameters.
        policy_params = {'type': params['policy_type'],
                         'policy_network_size': params['policy_network_size'],
                         'ob_dim': OB_DIM,
                         'ac_dim': AC_DIM}

    single_rollout_workers = []
    nsingleworks = min( len(FAULT_CASES), params['onedirection_numofcasestorun'])


    #-----------just test one single worker----------------------------------------

    for i in range(nsingleworks):
        single_rollout_workers.append(SingleRolloutWorker.remote(rollout_length=150,
                                                                      policy_params=policy_params))
        time.sleep(2)

    weights_id = ray.put(w_M)
    ob_mean = ray.put(w_mean)
    ob_std = ray.put(w_std)

    # -----------just test one single worker----------------------------------------
    icase = 1
    tmp_id = single_rollout_workers[icase].single_rollout.remote(
                fault_tuple=FAULT_CASES[icase], weights=weights_id, ob_mean=ob_mean,
                ob_std=ob_std)
    tmp_result = ray.get(tmp_id)

    print ('-----------testing just one single roll out for fault tuple: ', FAULT_CASES[icase])
    print (tmp_result)
    print('----------- finished testing for just one single roll out for fault tuple: ', FAULT_CASES[icase])
    print ('\n')

    for itr in range (5):

        results_ids = []

        # -----------just test different number of works for rolling out fully distributed for each itration-----------------
        nworkers = nsingleworks//(itr+1)

        print ('---------start multiple single roll outs, itr: ', itr, ', n-workers: ', nworkers)
        print('\n')
        for i in range(nworkers):
            results_ids.append(single_rollout_workers[i].single_rollout.remote(
                fault_tuple=FAULT_CASES[i], weights=weights_id, ob_mean=ob_mean,
                ob_std=ob_std))

        results = ray.get(results_ids)
        reward_list = []
        step_list = []
        time_list = []
        for result in results:
            reward_list.append(result['reward'])
            step_list.append(result['step'])
            time_list.append(result['time'])
        reward_ave = mean(reward_list)

        print ('--------------n-works: ', nworkers, ', average reward: ', reward_ave)
        for itmp in range(len(reward_list)):
            print('Reward: ', reward_list[itmp], ', Steps: ', step_list[itmp], ', Timing: ', time_list[itmp])
        # ----- delete the weight in the global table in ray before return

        print('\n')
	
    print ('-------------------finished all testing!!!!!!!!!!!!!--------------')
    del weights_id
    del ob_mean
    del ob_std

    return

# Setting
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', '-n', type=int, default=500)  # Number of iterations
    if EIOC:  # EIOC test
        parser.add_argument('--n_directions', '-nd', type=int, default=1)
    else:  # Constance training
        parser.add_argument('--n_directions', '-nd', type=int, default=128)
    if EIOC:  # EIOC test
        parser.add_argument('--deltas_used', '-du', type=int, default=2)
    else:  # Constance training
        parser.add_argument('--deltas_used', '-du', type=int, default=64)
    parser.add_argument('--step_size', '-s', type=float, default=1)  # default=0.02
    parser.add_argument('--delta_std', '-std', type=float, default=2)  # default=.03
    parser.add_argument('--decay', type=float, default=0.995)  # decay of step_size and delta_std
    parser.add_argument('--rollout_length', '-r', type=int, default=150)
    parser.add_argument('--seed', type=int, default=589)  # Seed Number for randomization
    parser.add_argument('--policy_type', type=str, default='LSTM')
    parser.add_argument('--dir_path', type=str,
                        default='ars_tamu2000_testrayandgridpack')  # Folder Name for outputs
    if EIOC:  # EIOC test
        parser.add_argument('--save_per_iter', type=int, default=1)  # save the .npz file per x iterations
    else:  # Constance training
        parser.add_argument('--save_per_iter', type=int, default=10)  # save the .npz file per x iterations
    parser.add_argument('--policy_network_size', type=list, default=[64, 64])

    parser.add_argument('--onedirection_numofcasestorun', type=int,
                        default=18)  # len(FAULT_CASES))  # For each direction, how many cases to run, default is to run all the fault
    # please set onedirection_numofcasestorun to be the multiplie of 3, such as 6,9,12,15,18, etc

    # please specify number of cores here; it is to help determine the number of workers
    if EIOC:  # EIOC test
        parser.add_argument('--cores', type=int, default=NUM_CORES)  # how many cores available as of hardware, EIOC test
    else:  # Constance training
        parser.add_argument('--cores', type=int, default=NUM_CORES)  # how many cores available as of hardware
    parser.add_argument('--policy_file', type=str, default="training_results/1_3faultcases/nonlinear_policy_plus490.npz")

    # the parameters controlling iterations for convergence
    # n_iter is maximum number of iterations
    # tol_p is tolerance for the percentage of average reward change between iterations
    # tol_steps is the total iterations for maintaining tol_p
    parser.add_argument('--tol_p', type=float, default=0.001)
    parser.add_argument('--tol_steps', type=int, default=100)
    local_ip = socket.gethostbyname(socket.gethostname())

    # Init Ray in Cluster, use log_to_driver=True for printing messages from remote nodes
    ##!!#####################################################################################################
    # !! if you are using EIOC, use this line of code below
    if EIOC:  # EIOC test
        ray.init(address="localhost:6379", log_to_driver=False)
    else:
        # !! if you are using Constance, use this line of code below
        ray.init(temp_dir=os.environ["tmpfolder"], redis_address=os.environ["ip_head"], log_to_driver=False)
    ##!!######################################################################################################

    args = parser.parse_args()
    params = vars(args)
    t1 = time.time()
    run_ars_test(params)
    t2 = time.time()
    print('Total Time run_ars:', t2 - t1)
