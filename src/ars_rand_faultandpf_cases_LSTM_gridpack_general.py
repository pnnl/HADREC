'''
This version of code is compatible to both EIOC and Constance
To run on a chosen platform, please set the '--cores' and ray.init() according to your system specifics
('--cores' and ray.init() and be found at the bottom of this code)

!!!!!!!!!!!!!!

MAKE SURE you set the cores >= onedirection_numofcasestorun !!!!!!! very critical
onedirection_numofcasestorun = SELECT_PF_PER_DIRECTION * len(FTD_CANDIDATES) * SELECT_FAULT_BUS_PER_DIRECTION

!!!!!!!!!!!!!!

The original version of ARS code is from https://github.com/modestyachts/ARS/tree/master/code
We modify the original ARS code to make it a two-level parallel implementation suitable for power grid control application

Authors: Renke Huang, Yujiao Chen, Qiuhua Huang, Tianzhixi Yin, Xinya Li, Ang Li, Wenhao Yu, Jie Tan
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
from statistics import *

folder_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
print('-----------------root path of the rlgc:', folder_dir)

EIOC = False #True, if you want to train on Constance or other HPC, set this flag as False, this will set different ray start method,
             #as well as more cores 
b_debug_timing = False #True

SYSTEM = 2000 # can choose 39, 300 or 2000 bus system

if SYSTEM == 300:  # for 300-bus system, seperate self-zone and other-zone, so use v10 of the env definition
    from GridPackPowerDynSimEnvDef_v10 import GridPackPowerDynSimEnv
else:
    from GridPackPowerDynSimEnvDef_v4 import GridPackPowerDynSimEnv
                                                               
if SYSTEM == 300:
    simu_input_file = folder_dir + '/testData/IEEE300/input_ieee300_genrou_exdc1_tgov1_zone1ac.xml'
    rl_config_file = folder_dir + '/testData/IEEE300/json/IEEE300_RL_loadShedding_zone1_continuous_LSTM_multipfcases_3ftdur_training_gp.json'

    OB_DIM = 142
    AC_DIM = 34
    # the following codes provide flexibility to define the power flow cases for training,
    if EIOC:
        SELECT_PF_PER_DIRECTION = 1
        POWERFLOW_CANDIDATES = list(range(1))
    else:
        SELECT_PF_PER_DIRECTION = 1
        POWERFLOW_CANDIDATES = list(range(1))

    # the following codes provide flexibility to define the total fault buses for training,
    # as well as how many fault buses to be selected from the total fault bus pool for each iteration of ARS
    if EIOC:
        SELECT_FAULT_BUS_PER_DIRECTION = 7
        FAULT_BUS_CANDIDATES = list(range(8)) #[0, 3]  #list(range(2))
    else:
        SELECT_FAULT_BUS_PER_DIRECTION = 7
        FAULT_BUS_CANDIDATES = list(range(8))

    FAULT_START_TIME = 1.0

    if EIOC:
        FTD_CANDIDATES = [0.1]
    else:
        FTD_CANDIDATES = [0.1]

elif SYSTEM == 2000:
    simu_input_file = folder_dir + '/testData/tamu2000/input_tamu2000_17pfcases_withob_v23_nocsv.xml'
    rl_config_file = folder_dir + '/testData/tamu2000/json/RLGC_RL_tamu2000_loadShedding_zone3_gp.json'
    OB_DIM = 301
    AC_DIM = 81
    # the following codes provide flexibility to define the power flow cases for training,
    if EIOC:
        SELECT_PF_PER_DIRECTION = 3
        POWERFLOW_CANDIDATES = list(range(3))
    else:
        SELECT_PF_PER_DIRECTION = 3
        POWERFLOW_CANDIDATES = list(range(3))

    # the following codes provide flexibility to define the total fault buses for training,
    # as well as how many fault buses to be selected from the total fault bus pool for each iteration of ARS
    if EIOC:
        SELECT_FAULT_BUS_PER_DIRECTION = 7
        #SELECT_FAULT_BUS_MSO = 3
        FAULT_BUS_CANDIDATES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        SELECT_FAULT_BUS_PER_DIRECTION = 7
        #SELECT_FAULT_BUS_MSO = 9
        FAULT_BUS_CANDIDATES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    FAULT_START_TIME = 1.0
    #FTD_CANDIDATES = [0.00, 0.05, 0.1]
    FTD_CANDIDATES = [0.10]

elif SYSTEM == 39:

    # the following is for a case that only loads at 3 buses could be shed
    simu_input_file = folder_dir + '/testData/IEEE39/input_39bus_step005_training_v33_newacloadperc43_multipf.xml'
    rl_config_file = folder_dir + '/testData/IEEE39/json/IEEE39_RL_loadShedding_3motor_5ft_gp_lstm.json'
    OB_DIM = 7
    AC_DIM = 3
	
    # the following codes provide flexibility to define the power flow cases for training,
    if EIOC:
        SELECT_PF_PER_DIRECTION = 1
        POWERFLOW_CANDIDATES = list(range(1)) #[0]
    else:
        SELECT_PF_PER_DIRECTION = 1
        POWERFLOW_CANDIDATES = list(range(1))

    # the following codes provide flexibility to define the total fault buses for training,
    # as well as how many fault buses to be selected from the total fault bus pool for each iteration of ARS
    if EIOC:
        SELECT_FAULT_BUS_PER_DIRECTION = 5
        FAULT_BUS_CANDIDATES = list(range(5))

    else:
        SELECT_FAULT_BUS_PER_DIRECTION = 5
        FAULT_BUS_CANDIDATES = list(range(5))

    FAULT_START_TIME = 1.0
    FTD_CANDIDATES = [0.08]

else:
    print ('----------!!! error in system definition, training will exit')
    sys.exit()

ONLY_FAULT_CASES_ALL = [(FAULT_BUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) 
                        for i in range(len(FAULT_BUS_CANDIDATES)) for j in range(len(FTD_CANDIDATES))]
						
PF_FAULT_CASES_ALL = [(POWERFLOW_CANDIDATES[k], FAULT_BUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) 
                for k in range(len(POWERFLOW_CANDIDATES))
                for i in range(len(FAULT_BUS_CANDIDATES)) 
				for j in range(len(FTD_CANDIDATES))]

# ======================================================================

@ray.remote
class SingleRolloutSlaver(object):
    '''
    responsible for doing one single rollout in the env
    '''
    
    def __init__(self, rollout_length, policy_params):

        #print('\n\n\n -----------------------Set Env=GridPackPowerDynSimEnv------------------------\n\n\n')
        '''
        if SYSTEM == 2000:
            # for 2000-bus system, some advanced tech such as other-zones with UVLS, action masks, and voltage filter can
            # be activated
            self.env = GridPackPowerDynSimEnv(simu_input_file, rl_config_file,
                                              force_symmetric_continuous_action=True,
                                              apply_uvls_otherzones=True,
                                              apply_action_mask=True,
                                              apply_volt_filter=True
                                              )
        else:
            self.env = GridPackPowerDynSimEnv(simu_input_file, rl_config_file, force_symmetric_continuous_action=True)
        '''

             
        self.env = GridPackPowerDynSimEnv(simu_input_file, rl_config_file, force_symmetric_continuous_action=True)
        
        #print('\n\n\n ----------------------------Set Env Done-----------------------------\n\n\n')
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
        # one SingleRolloutSlaver is only doing one fault case in an iter
        # fault_tuple = PF_FAULT_CASES_ALL[fault_case_id]
        total_reward = 0.
        steps = 0												
        self.policy.update_weights(weights)
        
        # us RS for collection of observation states; restart every rollout
        self.RS = RunningStat(shape=(OB_DIM,))

        t1 = time.time()											  
        ob = self.env.validate(case_Idx=fault_tuple[0], fault_bus_idx=fault_tuple[1],
                               fault_start_time=fault_tuple[2], fault_duration_time=fault_tuple[3])

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
            return {'reward': total_reward, 'step': steps,'time': [t2-t1 , t_act, t_step, t1, t2]}
        else:
            return {'reward': total_reward, 'step': steps}

    def return_RS(self):
        return self.RS

    def close_env(self):
        self.env.close_env()
        return


@ray.remote
class OneDirectionWorker(object):
    """
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 policy_params=None,
                 deltas=None,
                 rollout_length=None,
                 delta_std=None,
                 num_cases_torun=None):

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        self.delta_std = delta_std
        self.rollout_length = rollout_length

        if num_cases_torun == None:
            self.num_slaves = len(PF_FAULT_CASES_ALL)
        else:
            self.num_slaves = num_cases_torun

        # explore positive and negative delta at the same time for parallel execution
        '''		   
        self.single_rollout_slavers = [SingleRolloutSlaver.remote(rollout_length=rollout_length,
                                            policy_params=self.policy_params,
                                            java_port=self.java_port + i + 1) \
                           for i in range(self.num_slaves)]
        '''
        self.single_rollout_slavers = []
        for i in range(self.num_slaves):
            self.single_rollout_slavers.append(SingleRolloutSlaver.remote(rollout_length=rollout_length,
                                            policy_params=self.policy_params))
            time.sleep(2)

    def update_delta_std(self, delta_std):
        self.delta_std = delta_std

    def onedirction_rollout_multi_single_slavers(self, w_policy, ob_mean, ob_std, select_fault_cases_tuples, evaluate=False):
        rollout_rewards, deltas_idx = [], []
        steps = 0											

        # use RS to collect observation states from SingleRolloutSlavers; restart each iter
        self.RS = RunningStat(shape=(OB_DIM,))

        if evaluate:
            deltas_idx.append(-1)
            weights_id = ray.put(w_policy)

            reward_ids = []	 
														
            for i in range(len(select_fault_cases_tuples)):
                fault_tuple_tmp=select_fault_cases_tuples[i]
                #fault_tuple_idx=PF_FAULT_CASES_ALL.index(fault_tuple_tmp)
                reward_ids.append( self.single_rollout_slavers[i].single_rollout.remote(fault_tuple=fault_tuple_tmp,
                                    weights=weights_id, ob_mean=ob_mean, ob_std=ob_std) )

            results = ray.get(reward_ids)
            reward_list = []
            step_list = []
            for result in results:
                reward_list.append(result['reward'])
                step_list.append(result['step'])
            reward_ave = mean(reward_list)
            steps_max = max(step_list)			

			# ----- delete the weight in the global table in ray before return
            del weights_id	
			
            return {'reward_ave': reward_ave, 'reward_list': reward_list}

        else:
            idx, delta = self.deltas.get_delta(w_policy.size)
            delta = (self.delta_std * delta).reshape(w_policy.shape)
            deltas_idx.append(idx)

            pos_rewards_list = []
            pos_steps_list = []
            neg_rewards_list = []
            neg_steps_list = []
            time_list = []

            t1 = time.time()
            # compute reward and number of timesteps used for positive perturbation rollout    
            w_pos_id = ray.put(w_policy + delta)
            pos_list_ids = [self.single_rollout_slavers[i].single_rollout.remote(
                fault_tuple=select_fault_cases_tuples[i], weights=w_pos_id, ob_mean=ob_mean, ob_std=ob_std) \
                for i in range(self.num_slaves)]
            
            pos_results  = ray.get(pos_list_ids)
            for result in pos_results:
                pos_rewards_list.append(result['reward'])
                pos_steps_list.append(result['step'])
                if b_debug_timing:
                    time_list.append(result['time'])
					
            pos_reward = mean(pos_rewards_list)
            pos_steps = max(pos_steps_list)
            # get RS of SingleRolloutSlavers
            pos_RS_ids = [self.single_rollout_slavers[i].return_RS.remote() for i in range(self.num_slaves)]
            pos_RSs = ray.get(pos_RS_ids)

            # compute reward and number of timesteps used for negative pertubation rollout             
            w_neg_id = ray.put(w_policy - delta)
            neg_list_ids = [self.single_rollout_slavers[i].single_rollout.remote(
                fault_tuple=select_fault_cases_tuples[i], weights=w_neg_id, ob_mean=ob_mean, ob_std=ob_std) \
                for i in range(self.num_slaves)]
            neg_results  = ray.get(neg_list_ids)
            for result in neg_results:
                neg_rewards_list.append(result['reward'])
                neg_steps_list.append(result['step'])
                if b_debug_timing:
                    time_list.append(result['time'])
					
            neg_reward = mean(neg_rewards_list)
            neg_steps = max(neg_steps_list)												
            neg_RS_ids = [self.single_rollout_slavers[i].return_RS.remote() for i in range(self.num_slaves)]
            neg_RSs = ray.get(neg_RS_ids)
            t2 = time.time()
			
            # update RS from SingleRolloutSlavers
            for pos_RS in pos_RSs:
                self.RS.update(pos_RS)
            for neg_RS in neg_RSs:
                self.RS.update(neg_RS)

            steps += pos_steps + neg_steps
            rollout_rewards.append([pos_reward, neg_reward])
			# ----- delete the positive and negative weights in the global table in ray before return
            del w_pos_id
            del w_neg_id
			
            t3 = time.time()
			
            if b_debug_timing:			
                return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps, 'time': t2-t1, 't1': t1, 't2': t2, 't3': t3,\
                    'slavetime': time_list, 'slavestep': pos_steps_list+neg_steps_list}
            else:			
                return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps, 'time': t2-t1}
                    #'slavetime': time_list, 'slavestep': pos_steps_list+neg_steps_list}

    def get_filter(self):
        return self.RS
        

    def close_env(self):
        close_ids = [slave.close_env.remote() for slave in self.single_rollout_slavers]
        ray.get(close_ids)
        return


class ARSLearner(object):
    """
    Object class implementing the ARS algorithm.
    """

    def __init__(self,
                 policy_params=None,
                 num_deltas=None,
                 deltas_used=None,
                 delta_std=None,
                 logdir=None,
                 rollout_length=None,
                 step_size=None,
                 params=None,
                 seed=None,
                 decay=1,
                 onedirection_numofcasestorun=None,
                 cores=None,
                 policy_file = None):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        self.timesteps = 0
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.decay = decay
        self.RS = RunningStat(shape=(OB_DIM,))
        if policy_file != "":
            self.RS._M = np.zeros(OB_DIM, dtype = np.float64)
            self.RS._S = np.zeros(OB_DIM, dtype = np.float64)
            w_mean_temp = policy_params['p_M'].copy()
            w_std_temp = policy_params['p_S'].copy()
            self.RS._M[:] = w_mean_temp[:]
            self.RS._S[:] = w_std_temp[:]
            self.RS._n = policy_params['p_n']							 
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id))
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing multidirection workers.')
		
        self.onedirection_numofcasestorun = min(onedirection_numofcasestorun, len(PF_FAULT_CASES_ALL))

        if cores >= num_deltas * self.onedirection_numofcasestorun:
            self.num_workers = num_deltas
            self.repeat = 1
            self.remain = 0
        else:
            self.num_workers = cores // self.onedirection_numofcasestorun
            self.repeat = num_deltas // self.num_workers
            self.remain = num_deltas % self.num_workers												  
        #self.num_workers = num_deltas

        print('\n------------!!!! workers allocation: \n')	
        print('------------!!!! total cores:', cores, ' , total directions:', self.num_deltas, ',  onedirection_numofcasestorun: ', self.onedirection_numofcasestorun) 		
        print('------------!!!! num_workers:', self.num_workers, ',  repeat: ', self.repeat, ', remain:', self.remain)
        print('\n------------!!!!  \n')
			
        self.multi_direction_workers = [OneDirectionWorker.remote(seed + 7 * i,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std,
                                      num_cases_torun = self.onedirection_numofcasestorun,																						  
                                      ) for i in range(self.num_workers)]
        # initialize policy
        print('Initializing policy.')
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'nonlinear':
            self.policy = FullyConnectedNeuralNetworkPolicy(policy_params, seed)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'LSTM':
            self.policy = LSTMPolicy(policy_params)
            self.w_policy = self.policy.get_weights()													 																		
        else:
            raise NotImplementedError

        # initialize optimization algorithm
        print('Initializing optimizer.')
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, evaluate=False):
        """
        Aggregate update step from rollouts generated in parallel.
        """

        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)
        ob_mean = ray.put(self.RS.mean)
        ob_std = ray.put(self.RS.std)

        if evaluate:
			
            # note that for evaluation, we should evaluate all the fault cases, but we only have slaves for the num of selected cases to run, so 
            # we need to repeat call workers to evaluate all the fault cases	
			
            full_repeat = len(PF_FAULT_CASES_ALL) // (self.num_workers * self.onedirection_numofcasestorun)
            full_cases_remain = len(PF_FAULT_CASES_ALL) % (self.num_workers * self.onedirection_numofcasestorun)

            repeat_num = full_cases_remain // self.onedirection_numofcasestorun
            remain_num = full_cases_remain % self.onedirection_numofcasestorun

            rollout_ids = []
            for repeat in range(full_repeat):
                for iworker in range(self.num_workers):
                    rollout_ids += [self.multi_direction_workers[iworker].onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std,
                             select_fault_cases_tuples=PF_FAULT_CASES_ALL[repeat * self.onedirection_numofcasestorun*self.num_workers
                                                                          + iworker*self.onedirection_numofcasestorun:
                                                                          repeat * self.onedirection_numofcasestorun * self.num_workers
                                                                          +(iworker + 1) * self.onedirection_numofcasestorun],
                                                                           evaluate=True)]

            for repeat in range(repeat_num):
                rollout_ids += [self.multi_direction_workers[repeat].onedirction_rollout_multi_single_slavers.remote(
                                policy_id, ob_mean, ob_std,
                                select_fault_cases_tuples=PF_FAULT_CASES_ALL[full_repeat * self.onedirection_numofcasestorun*self.num_workers
                                                                            +repeat * self.onedirection_numofcasestorun:
                                                                             full_repeat * self.onedirection_numofcasestorun * self.num_workers
                                                                            +(repeat + 1) * self.onedirection_numofcasestorun],
                                                                            evaluate=True)]

            if remain_num != 0:
                rollout_ids += [self.multi_direction_workers[repeat_num].onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std,
                    select_fault_cases_tuples = PF_FAULT_CASES_ALL[-remain_num:],
                    evaluate=True )]

            # gather results
            results_list = ray.get(rollout_ids)
			
            rewards_list = []
            for result in results_list:
                rewards_list += result['reward_list']

            reward_eval_avr = mean(rewards_list)			
			
            del policy_id
            del ob_mean
            del ob_std
			
            return reward_eval_avr, rewards_list

        t1 = time.time()

        select_faultbus_num = SELECT_FAULT_BUS_PER_DIRECTION
        select_faultbuses_id = np.random.choice(FAULT_BUS_CANDIDATES, size=select_faultbus_num, replace=False)
        print ("select_faultbuses_id:  ", select_faultbuses_id)

        select_pf_num = SELECT_PF_PER_DIRECTION
        select_pfcases_id = np.random.choice(POWERFLOW_CANDIDATES, size=select_pf_num, replace=False)
        print ("select_pfcases_id:  ", select_pfcases_id)

        select_fault_cases_tuples = [(select_pfcases_id[k], select_faultbuses_id[i], FAULT_START_TIME, FTD_CANDIDATES[j]) \
                                        for k in range(len(select_pfcases_id))
                                        for i in range(len(select_faultbuses_id)) for j in range(len(FTD_CANDIDATES))]
										
        select_fault_cases_tuples_id = ray.put(select_fault_cases_tuples)
        print ("select_fault_cases_tuples:  ", select_fault_cases_tuples) 
		
        rollout_id_list = []
        for repeat in range(self.repeat):										 																						  
            rollout_id_list += [worker.onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std, select_fault_cases_tuples_id,
                                                      evaluate=False) for worker in self.multi_direction_workers]
													  
        rollout_id_list += [worker.onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std, select_fault_cases_tuples_id, \
                            evaluate=False) for worker in self.multi_direction_workers[:self.remain]]											   											
		
        # gather results
        results_list = ray.get(rollout_id_list)
        rollout_rewards, deltas_idx = [], []
		
        t3 = time.time()
        
        iworktmp = 0
        for result in results_list:
            self.timesteps += result["steps"]										  
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            if b_debug_timing:
               print('-----Worker No: ', iworktmp, ', Time on worker to compute +- rollouts, t1, t2, t3: ', result['time'], result['t1'], result['t2'], result['t3'] )
               for slaveT in result['slavetime']:
                   print(slaveT)
               for slaveS in result['slavestep']:
                   print(slaveS) 
            iworktmp += 1				   

        print("rollout_rewards shape:", np.asarray(rollout_rewards).shape)
        print("deltas_idx shape:", np.asarray(deltas_idx).shape)

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()
        
        print('---Time to print rollouts results: ', t2 - t3)
        print('---Full Time to generate rollouts: ', t2 - t1)
        if b_debug_timing:
            print('---t1, t2, t3 for function aggregate_rollouts: ', t1, t2, t3)
        
        # update RS from all workers
        for j in range(self.num_workers):
            self.RS.update(ray.get(self.multi_direction_workers[j].get_filter.remote()))

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis=1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100 * ( 
                    1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        if np.std(rollout_rewards) > 1:
            rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size=500)
        g_hat /= deltas_idx.size
        # del the policy weights, ob_mean and ob_std in the object store
        del policy_id
        del ob_mean
        del ob_std
        del select_fault_cases_tuples_id																	
        t2 = time.time()
        print('\n----time to aggregate rollouts: ', t2 - t1)
        return g_hat

    def train_step(self):
        """
        Perform one update step of the policy weights.
        """

        g_hat = self.aggregate_rollouts()
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat, self.step_size).reshape(self.w_policy.shape)
        print('g_hat shape, w_policy shape:',np.asarray(g_hat).shape,self.w_policy.shape)
        return

    def train(self, num_iter, save_per_iter, tol_p, tol_steps):

        rewardlist = []
        last_reward = 0.0 
        n           = int(tol_steps/save_per_iter)
        start = time.time()
        for i in range(num_iter):
            self.step_size *= self.decay
            self.delta_std *= self.decay

            for worker in self.multi_direction_workers:
                worker.update_delta_std.remote(self.delta_std)

            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)
            print('iter ', i, ' done')

            t1 = time.time()
            # record statistics every x iterations
            if (i == 0) or ((i + 1) % save_per_iter == 0):
                reward, reward_list = self.aggregate_rollouts(evaluate=True)
                w = [self.w_policy, self.RS.mean, self.RS.std, self.RS._n, self.RS._M, self.RS._S, self.step_size, self.delta_std]
                np.savez(self.logdir + "/nonlinear_policy_plus" + str(i + 1), w)
                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", reward)
                for f_index in range(len(PF_FAULT_CASES_ALL)):
                    logz.log_tabular("reward {}:".format(f_index), reward_list[f_index])
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()

                if last_reward != 0.0:
                    tol = abs((reward - last_reward)/last_reward) <= tol_p
                    rewardlist.append(tol)
                    if len(rewardlist) >= n and all(rewardlist[-5:]):
                        break
                last_reward = reward
             
            t2 = time.time()
            print('total time of save:', t2 - t1)           															 

        #close_ids = [worker.close_env.remote() for worker in self.multi_direction_workers]
        #ray.get(close_ids)
        return


def run_ars(params):

    t1 = time.time()
    logdir = params['dir_path']
    version_num = 0
    while os.path.exists(logdir + "_v" + str(version_num)):
        version_num += 1
    logdir = logdir + "_v" + str(version_num)
    os.makedirs(logdir)

    # set policy parameters.     
    if params['policy_file'] != "":
	
        print ( '\n')
        print ( '------------ policy file is not empty, initialize weights with:', params['policy_file'])
        print ( '\n')
 
        nonlin_policy_org = np.load(params['policy_file'], allow_pickle = True)
        nonlin_policy = nonlin_policy_org['arr_0']

        w_M = nonlin_policy[0].copy()
        w_mean = nonlin_policy[1].copy()
        w_std = nonlin_policy[2].copy()
        p_n = nonlin_policy[3]
        p_M = nonlin_policy[4].copy()
        p_S = nonlin_policy[5].copy()
        p_step_size = nonlin_policy[6]
        p_delta_std = nonlin_policy[7]
        
        policy_params = {'type': params['policy_type'],
                     'policy_network_size': params['policy_network_size'],
                     'ob_dim': OB_DIM,
                     'ac_dim': AC_DIM,
                     'weights': w_M,
                     'p_n':p_n,
                     'p_M':p_M,
                     'p_S':p_S,
                     'p_step_size':p_step_size,
                     'p_delta_std':p_delta_std}
    else:
        # set policy parameters. 
        policy_params = {'type': params['policy_type'],
                         'policy_network_size': params['policy_network_size'],
                         'ob_dim': OB_DIM,
                         'ac_dim': AC_DIM}

    ARS = ARSLearner(policy_params=policy_params,
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'],
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     params=params,
                     seed=params['seed'],
                     decay=params['decay'],
                     onedirection_numofcasestorun=params['onedirection_numofcasestorun'],
                     cores=params['cores'], 
                     policy_file =params['policy_file'])

    t2 = time.time()
    print('Total Time Initialize ARS:', t2-t1)
    ARS.train(params['n_iter'], params['save_per_iter'], params['tol_p'], params['tol_steps'])
    return


# Setting
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    if SYSTEM == 300:
        parser.add_argument('--n_iter', '-n', type=int, default=700)  # Number of iterations
        if EIOC: # EIOC test
            parser.add_argument('--n_directions', '-nd', type=int, default=128)
        else:    # Constance training
            parser.add_argument('--n_directions', '-nd', type=int, default=128)

        if EIOC: # EIOC test
            parser.add_argument('--deltas_used', '-du', type=int, default=64)
        else:    # Constance training
            parser.add_argument('--deltas_used', '-du', type=int, default=64)

        if EIOC:
            parser.add_argument('--policy_network_size', type=list, default=[64, 64])
        else:
            parser.add_argument('--policy_network_size', type=list, default=[64, 64])

    elif SYSTEM == 2000:
        parser.add_argument('--n_iter', '-n', type=int, default=600)  # Number of iterations
        if EIOC: # EIOC test
            parser.add_argument('--n_directions', '-nd', type=int, default=128)
        else:    # Constance training
            parser.add_argument('--n_directions', '-nd', type=int, default=128)
        if EIOC: # EIOC test
            parser.add_argument('--deltas_used', '-du', type=int, default=64)
        else:    # Constance training
            parser.add_argument('--deltas_used', '-du', type=int, default=64)
        parser.add_argument('--policy_network_size', type=list, default=[64, 64])

    elif SYSTEM == 39:
        parser.add_argument('--n_iter', '-n', type=int, default=500)  # Number of iterations
        if EIOC:  # EIOC test
            parser.add_argument('--n_directions', '-nd', type=int, default=32)
        else:  # Constance training
            parser.add_argument('--n_directions', '-nd', type=int, default=32)
        if EIOC:  # EIOC test
            parser.add_argument('--deltas_used', '-du', type=int, default=16)
        else:  # Constance training
            parser.add_argument('--deltas_used', '-du', type=int, default=16)

        parser.add_argument('--policy_network_size', type=list, default=[32, 32])

    else:
        print('----------!!! error in system definition, training will exit')
        sys.exit()

    parser.add_argument('--step_size', '-s', type=float, default=1)  # default=0.02
    parser.add_argument('--delta_std', '-std', type=float, default=2)  # default=.03
    parser.add_argument('--decay', type=float, default=0.9985)  # decay of step_size and delta_std
    parser.add_argument('--rollout_length', '-r', type=int, default=90)
    parser.add_argument('--seed', type=int, default=589)  # Seed Number for randomization
    parser.add_argument('--policy_type', type=str, default='LSTM')
    parser.add_argument('--dir_path', type=str,
                        default='outputs_training/ars_%d_bussys_%d_pf_%d_faultbus_%d_dur_lstm_gridpack'%(SYSTEM,
                        SELECT_PF_PER_DIRECTION, SELECT_FAULT_BUS_PER_DIRECTION, len(FTD_CANDIDATES)))  # Folder Name for outputs
    if EIOC: # EIOC test
        if SYSTEM == 2000:
            parser.add_argument('--save_per_iter', type=int, default=1)  # save the .npz file per x iterations
        elif SYSTEM == 39:
            parser.add_argument('--save_per_iter', type=int, default=10)  # save the .npz file per x iterations
        else:
            parser.add_argument('--save_per_iter', type=int, default=1)  # save the .npz file per x iterations
    else:    # Constance training
        parser.add_argument('--save_per_iter', type=int, default=10)  # save the .npz file per x iterations

    # onedirection_numofcasestorun: For each direction, how many cases to run, default is to run all the PF_FAULT_CASES_ALL,
    # but we could also randonmly select part of the PF_FAULT_CASES_ALL,
    # this para defines how many cases to randomly select from PF_FAULT_CASES_ALL
    casestorunperdirct = SELECT_PF_PER_DIRECTION * len(FTD_CANDIDATES) * SELECT_FAULT_BUS_PER_DIRECTION
    parser.add_argument('--onedirection_numofcasestorun', type=int, default=casestorunperdirct)
    
    # please specify number of cores here; it is to help determine the number of workers
    if EIOC: # EIOC test
        parser.add_argument('--cores', type=int, default=30) # how many cores available as of hardware, EIOC test
    else:    # Constance training
        parser.add_argument('--cores', type=int, default=23*22) # how many cores available as of hardware
    parser.add_argument('--policy_file', type=str, default="")
    
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
    if EIOC: # EIOC test
        ray.init(redis_address="localhost:6379", log_to_driver=False)
    else:
    # !! if you are using Constance, use this line of code below															 																									  
        ray.init(temp_dir=os.environ["tmpfolder"], redis_address=os.environ["ip_head"], log_to_driver=False)
##!!######################################################################################################

    args = parser.parse_args()
    params = vars(args)
    t1 = time.time()
    run_ars(params)
    t2 = time.time()
    print('Total Time run_ars:', t2-t1)    
