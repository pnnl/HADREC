
'''

Need at least 17 cores to fully parallel the single rollouts and test this code

Authors: Renke Huang, Qiuhua Huang
Contact: qiuhua.huang@pnnl.gov

'''

import sys, os, time, parser, math
import numpy as np
import gym, ray
import socket

from PowerDynSimEnvDef_v7 import PowerDynSimEnv
from statistics import *

folder_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
print('-----------------root path of the rlgc:', folder_dir)

case_files_array = []
case_files_array.append(folder_dir + '/testData/IEEE300/IEEE300Bus_modified_noHVDC_v2.raw')
case_files_array.append(folder_dir + '/testData/IEEE300/IEEE300_dyn_cmld_zone1.dyr')
dyn_config_file = folder_dir + '/testData/IEEE300/json/IEEE300_dyn_config.json'
rl_config_file = folder_dir + '/testData/IEEE300/json/IEEE300_RL_loadShedding_zone1_continuous_LSTM_new_lessfaultbuses_moreActionBuses_multipfcases.json'
jar_file = folder_dir + '/lib/RLGCJavaServer1.0.0_rc.jar'
# This is to fix the issue of "ModuleNotFoundError" below
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

EIOC = False  # for GCP training, set this to be False

if EIOC:
    NUM_OF_GCP_CORES = 17  # Yan used 840 cores for training on Constance
else:
    NUM_OF_GCP_CORES = 17  # Yan used 840 cores for training on Constance


SELECT_PF_PER_DIRECTION = 4
POWERFLOW_CANDIDATES = [0, 1, 2, 3]  # only select 3 out of 5 base pf cases for training

SELECT_FAULT_BUS_PER_DIRECTION = 4
FAULT_BUS_CANDIDATES = [0, 1, 2, 3]

FAULT_START_TIME = 1.0
FTD_CANDIDATES = [0.1]  # [0.00, 0.05, 0.08]
ONLY_FAULT_CASES_ALL = [(FAULT_BUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) for i in
                        range(len(FAULT_BUS_CANDIDATES)) for j in range(len(FTD_CANDIDATES))]
PF_FAULT_CASES_ALL = [(POWERFLOW_CANDIDATES[k], FAULT_BUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) \
                      for k in range(len(POWERFLOW_CANDIDATES)) for i in range(len(FAULT_BUS_CANDIDATES)) \
                      for j in range(len(FTD_CANDIDATES))]

OB_DIM = 154
AC_DIM = 46
LATENT_DIM = 10  ##  not sure whether 10 is good
JAVA_PORT = 26336


# ======================================================================

@ray.remote
class SingleRolloutSlaver(object):
    '''
    responsible for doing rollout in the env
    '''

    def __init__(self, java_port, rollout_length):

        print('\n\n\n -----------------------Set Env=PowerDynSimEnv------------------------\n\n\n')
        self.env = PowerDynSimEnv(case_files_array, dyn_config_file, rl_config_file, jar_file, java_port, folder_dir,
                                  force_symmetric_continuous_action=True)
        print("----------all base cases: \n")
        print(self.env.get_base_cases())
        print('\n\n\n ----------------------------Set Env Done-----------------------------\n\n\n')
        self.rollout_length = rollout_length
        self.java_port = java_port


    def single_rollout_noact(self, fault_tuple):
        # one SingleRolloutSlaver is only doing one fault case in an iter
        # fault_tuple = PF_FAULT_CASES_ALL[pf_fault_tuple_id]
        total_reward = 0.
        steps = 0

        t1 = time.time()

        ob = self.env.validate(case_Idx=fault_tuple[0], fault_bus_idx=fault_tuple[1],
                               fault_start_time=fault_tuple[2], fault_duation_time=fault_tuple[3])

        ob_list = []
        for _ in range(self.rollout_length):


            act_lst = [1 for i in range(AC_DIM)]

            action_org = np.array(act_lst)
            ob, reward, done, _ = self.env.step(action_org)

            ob_list.append(np.array(ob))

            total_reward += reward
            steps += 1
            if done:
                break

        t2 = time.time()

        return {'obs': np.array(ob_list), 'reward': total_reward, 'step': steps, 'time': t2-t1}

    def get_base_cases(self):

        return self.env.get_base_cases()

    def close_java_connection(self):
        self.env.close_connection()
        print('connection with Ipss Server is closed')
        return True

def run_test_java_ray(params):

    single_rollout_workers = []
    nsingleworks = min(len(PF_FAULT_CASES_ALL), params['onedirection_numofcasestorun'])

    print(' nsingleworks: ', nsingleworks, )
    print (' Num of fault tuples: ', len(PF_FAULT_CASES_ALL))
    print(' PF_FAULT_CASES_ALL:')
    print(PF_FAULT_CASES_ALL)

    for i in range(nsingleworks):
        single_rollout_workers.append(SingleRolloutSlaver.remote(java_port=JAVA_PORT+i+1, rollout_length=150))
        time.sleep(2)

    time_start = time.time()
    results_ids = []
    for i in range(nsingleworks):
        results_ids.append(single_rollout_workers[i].single_rollout_noact.remote(
            fault_tuple=PF_FAULT_CASES_ALL[i]) )

    results = ray.get(results_ids)
    reward_list = []
    step_list = []
    time_list = []
    for result in results:
        reward_list.append(result['reward'])
        step_list.append(result['step'])
        time_list.append(result['time'])

    reward_ave = mean(reward_list)
    time_end = time.time()

    print('--------------n-works: ', nsingleworks, ', average reward: ', reward_ave, ', total_time for single roll out: ', time_end-time_start)
    for itmp in range(len(reward_list)):
        print('Fault tuple: ' , PF_FAULT_CASES_ALL[itmp], 'Reward: ', reward_list[itmp], ', Steps: ', step_list[itmp], ', Timing: ', time_list[itmp])

    results_ids_close = []
    for i in range(nsingleworks):
        results_ids.append(single_rollout_workers[i].close_java_connection.remote() )

    results_close = ray.get(results_ids_close)

    print ('-----------all finished-----------------')

    return

# Setting
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # select_faultbus_num = 3  # how many fault buses to select each iteration of ARS
    casestorunperdirct = SELECT_PF_PER_DIRECTION * len(FTD_CANDIDATES) * SELECT_FAULT_BUS_PER_DIRECTION
    parser.add_argument('--onedirection_numofcasestorun', type=int, default=casestorunperdirct)

    parser.add_argument('--cores', type=int,
                            default=NUM_OF_GCP_CORES)  # how many cores available as of hardware, EIOC test

    local_ip = socket.gethostbyname(socket.gethostname())

    # Init Ray in Cluster, use log_to_driver=True for printing messages from remote nodes
    ##!!#####################################################################################################
    # !! if you are using EIOC, use this line of code below
    if EIOC:  # EIOC test
        ray.init(redis_address="localhost:6379", log_to_driver=False)
    else:
        # !! if you are using GCP, use this line of code below
        # ray.init(temp_dir=os.environ["tmpfolder"], redis_address=os.environ["ip_head"], log_to_driver=False)
        ray.init(address="localhost:6379", log_to_driver=False)
    ##!!######################################################################################################
    args = parser.parse_args()
    params = vars(args)
    run_test_java_ray(params)
