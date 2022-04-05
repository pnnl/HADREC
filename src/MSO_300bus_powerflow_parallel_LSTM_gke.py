'''

This version of code is compatible to both EIOC and Constance
To run on a chosen platform, please set the '--cores' and ray.init() according to your system specifics
('--cores' and ray.init() and be found at the bottom of this code)

!!!!!!!!!!!!!!

MAKE SURE you set the cores >= onedirection_numofcasestorun !!!!!!! very critical
onedirection_numofcasestorun = SELECT_PF_PER_DIRECTION * len(FTD_CANDIDATES) * SELECT_FAULT_BUS_PER_DIRECTION

!!!!!!!!!!!!!!

The original version of ARS code is from https://github.com/modestyachts/ARS/tree/master/code
We modify the original ARS code to make it a two-level parallel implementation suitable for power grid control application, and we also add the meta optimization code

Authors: Renke Huang, Yujiao Chen, Qiuhua Huang, Tianzhixi Yin, Xinya Li, Ang Li, Wenhao Yu, Jie Tan
Contact: qiuhua.huang@pnnl.gov

'''

import sys, os, time, parser, math
import numpy as np
import gym, ray
import logz, optimizers, utils
from obserfilter import RunningStat
from policy_MSO_LSTM import *
import socket
from shared_noise import *
from PowerDynSimEnvDef_v7 import PowerDynSimEnv
from statistics import *
#from up_optimizer_new import UPOptimizer
from bayesian_optimization import *

folder_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
print('-----------------root path of the rlgc:', folder_dir)

case_files_array = []
case_files_array.append(folder_dir + '/testData/IEEE300/IEEE300Bus_modified_noHVDC_v2.raw')
case_files_array.append(folder_dir + '/testData/IEEE300/IEEE300_dyn_cmld_zone1.dyr')
dyn_config_file = folder_dir + '/testData/IEEE300/json/IEEE300_dyn_config.json'
rl_config_file =  folder_dir + '/testData/IEEE300/json/IEEE300_RL_loadShedding_zone1_continuous_LSTM_new_lessfaultbuses_moreActionBuses_multipfcases.json'
jar_file = folder_dir + '/lib/RLGCJavaServer1.0.0_rc.jar'
# This is to fix the issue of "ModuleNotFoundError" below
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

EIOC = False  # for Constance training, set this to be False

if EIOC:
    NUM_OF_GCP_CORES = 27 # Yan used 840 cores for training on Constance
else:
    NUM_OF_GCP_CORES = 840 # Yan used 840 cores for training on Constance

if EIOC:
    MSO_OPT_MAX_ITER = 3    # max internal iteration number for one bayesian optimization
else:
    MSO_OPT_MAX_ITER = 16

if EIOC:
    SELECT_PF_PER_DIRECTION = 2
    POWERFLOW_CANDIDATES = [0, 1]
else:
    SELECT_PF_PER_DIRECTION = 2
    POWERFLOW_CANDIDATES = [0, 1, 2, 3] # only select 3 out of 5 base pf cases for training

if EIOC:
    SELECT_FAULT_BUS_PER_DIRECTION = 4
    NUM_HARD_FAULT_ADDED = 2
    START_HARD_CASE_ITR = 1
    FAULT_BUS_CANDIDATES = [0, 1, 2, 3]
else:
    SELECT_FAULT_BUS_PER_DIRECTION = 6
    NUM_HARD_FAULT_ADDED = 3
    START_HARD_CASE_ITR = 100
    FAULT_BUS_CANDIDATES = [0, 1, 2, 3, 4, 5]
FAULT_START_TIME = 1.0
FTD_CANDIDATES = [0.1]  #[0.00, 0.05, 0.08]
ONLY_FAULT_CASES_ALL = [(FAULT_BUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) for i in range(len(FAULT_BUS_CANDIDATES)) for j in range(len(FTD_CANDIDATES))]
PF_FAULT_CASES_ALL = [(POWERFLOW_CANDIDATES[k], FAULT_BUS_CANDIDATES[i], FAULT_START_TIME, FTD_CANDIDATES[j]) \
                  for k in range(len(POWERFLOW_CANDIDATES)) for i in range(len(FAULT_BUS_CANDIDATES)) \
                  for j in range(len(FTD_CANDIDATES))]


OB_DIM = 154
AC_DIM = 46
LATENT_DIM = 10  ##  not sure whether 10 is good
JAVA_PORT = 26336
# ======================================================================

# ------------------------------------------------------------------------
class UPOptimizer:
    def __init__(self, single_rollout_slavers=None, pf_cases_idx=None, weight=None, latent_dim=None, verbose=True,
                 terminate_threshold=-np.inf, bayesian_opt=True,
                 max_rollout_length=None, ob_mean=None, ob_std=None, pfcases=None, faultcases=None):
        self.single_rollout_slavers = single_rollout_slavers
        self.pf_cases_idx = pf_cases_idx
        self.weight = weight
        self.dim = latent_dim
        self.sample_num = 0
        self.rollout_num = 0
        self.verbose = verbose
        self.terminate_threshold = terminate_threshold

        self.solution_history = []
        self.sample_num_history = []
        self.best_f = 10000000.0
        self.best_x = None
        self.best_meanrollout_length = 0
        self.max_rollout_length = max_rollout_length

        self.bayesian_opt = bayesian_opt
        self.ob_mean = ob_mean
        self.ob_std = ob_std

        self.pfcases = pfcases
        self.faultcases = faultcases

    def reset(self):
        self.sample_num = 0
        self.rollout_num = 0

        self.best_f = 10000000.0
        self.best_x = None
        self.best_meanrollout_length = 0
        self.solution_history = []
        self.sample_num_history = []
        self.max_steps = 200000

    def fitness(self, x):
        '''
        case = self.case
        app = np.copy(x)
        avg_perf = []
        rollout_len = []

        o = self.env.validate(0,case[0], case[1], case[2])
        d = False
        rollout_rew = 0
        rollout_len.append(0)
        while not d:
            o = np.asarray(o, dtype=np.float64)
            o = (o - self.ob_mean) / (self.ob_std + 1e-8)
            a = self.policy.act(o, app)
            o, r, d, _ = self.env.step(a)
            rollout_rew += r
            self.sample_num += 1
            rollout_len[-1] += 1
            if self.max_rollout_length is not None:
                if rollout_len[-1] >= self.max_rollout_length:
                    break
        self.rollout_num += 1
        avg_perf.append(rollout_rew)

        if -rollout_rew < self.best_f:
            self.best_x = np.copy(x)
            self.best_f = -np.mean(avg_perf)
            self.best_meanrollout_length = np.mean(rollout_len)
        # print('Sampled perf: ', np.mean(avg_perf))
        return -rollout_rew

        '''
        # ------------------renke write new--------------------
        # case = self.case
        app = np.copy(x)
        avg_perf = []
        '''
        reward_ids = [self.single_rollout_slavers[i].rollout_pfidx_faultidx_1d.remote(pf_case_id=self.pf_cases_idx, \
                                                                          fault_case_id=i, weights=self.weight,
                                                                          ob_mean=self.ob_mean, ob_std=self.ob_std,
                                                                          latent=app) \
                      for i in range(len(self.faultcases))]
        '''

        reward_ids = []
        for i in range(len(self.faultcases)):
            fault_case = self.faultcases[i]
            pf_ft_tuple_tmp = (self.pfcases[ self.pf_cases_idx ], fault_case[0], fault_case[1], fault_case[2])
            reward_ids.append(self.single_rollout_slavers[i].single_rollout.remote(fault_tuple=pf_ft_tuple_tmp,
                                                                                   weights=self.weight,
                                                                                   ob_mean=self.ob_mean,
                                                                                   ob_std=self.ob_std,
                                                                                   latent=app))


        results_list = ray.get(reward_ids)

        reward_list = []
        rollout_len = []
        for result in results_list:
            reward_list.append(result['reward'])
            rollout_len.append(result['step'])

        rollout_rew = np.mean(reward_list)

        self.rollout_num += 1
        avg_perf.append(rollout_rew)

        if -rollout_rew < self.best_f:
            self.best_x = np.copy(x)
            self.best_f = -np.mean(avg_perf)
            self.best_meanrollout_length = np.mean(rollout_len)
        # print('Sampled perf: ', np.mean(avg_perf))
        return -rollout_rew

    '''
    def cames_callback(self, es):
        self.solution_history.append(self.best_x)
        self.sample_num_history.append(self.sample_num)
        if self.verbose:
            logger.record_tabular('CurrentBest', repr(self.best_x))
            logger.record_tabular('EpRewMean', self.best_f)
            logger.record_tabular("TimestepsSoFar", self.sample_num)
            logger.record_tabular("EpisodesSoFar", self.rollout_num)
            logger.record_tabular("EpLenMean", self.best_meanrollout_length)
            logger.dump_tabular()
        return self.sample_num
    '''

    def termination_callback(self, es):
        if self.sample_num > self.max_steps:  # stop if average length is over 900
            return True
        return False

    def optimize(self, maxiter=10, max_steps=100, custom_bound=None):
        if self.dim > 1 or self.bayesian_opt:
            self.max_steps = max_steps

            if custom_bound is None:
                init_guess = [0.0] * self.dim
                init_std = 0.5
                bound = [0.0, 1.0]
            else:
                init_guess = [0.5 * (custom_bound[0] + custom_bound[1])] * self.dim
                init_std = abs(0.5 * (custom_bound[0] - custom_bound[1]))
                bound = [custom_bound[0], custom_bound[1]]

            if self.bayesian_opt:
                xs, ys, _, _, _ = bayesian_optimisation(maxiter, self.fitness,
                                                        bounds=np.array([bound] * self.dim),
                                                        max_steps=max_steps,
                                                        random_search=1000)
                # callback=self.cames_callback)
                xopt = xs[np.argmin(ys)]
            else:
                xopt, es = cma.fmin2(self.fitness, init_guess, init_std, options={'bounds': bound, 'maxiter': maxiter,
                                                                                  'ftarget': self.terminate_threshold,
                                                                                  'termination_callback': self.termination_callback
                                                                                  },
                                     callback=self.cames_callback)

            print('optimized: ', repr(xopt))
        else:
            # 1d case, not used
            candidates = np.arange(-1, 1, 0.05)
            fitnesses = [self.fitness([candidate]) for candidate in candidates]
            xopt = [candidates[np.argmin(fitnesses)]]

        return xopt

# ======================================================================

@ray.remote
class SingleRolloutSlaver(object):
    '''
    responsible for doing rollout in the env
    '''
    
    def __init__(self, java_port, rollout_length, policy_params):

        print('\n\n\n -----------------------Set Env=PowerDynSimEnv------------------------\n\n\n')
        self.env = PowerDynSimEnv(case_files_array, dyn_config_file, rl_config_file, jar_file, java_port, folder_dir,
                                  force_symmetric_continuous_action=True)
        print("----------all base cases: \n")
        print(self.env.get_base_cases())
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


    def single_rollout(self, fault_tuple, weights, ob_mean, ob_std, latent):
        # one SingleRolloutSlaver is only doing one fault case in an iter
        #fault_tuple = PF_FAULT_CASES_ALL[pf_fault_tuple_id]
        total_reward = 0.
        steps = 0
        self.policy.update_weights(weights)
        
        # us RS for collection of observation states; restart every rollout
        self.RS = RunningStat(shape=(OB_DIM,))

        t1 = time.time()

        ob = self.env.validate(case_Idx=fault_tuple[0], fault_bus_idx=fault_tuple[1],
                               fault_start_time=fault_tuple[2], fault_duation_time=fault_tuple[3])

        if self.policy_type == 'LSTM':
            self.policy.reset()

        for _ in range(self.rollout_length):
            ob = np.asarray(ob, dtype=np.float64)
            self.RS.push(ob)
            normal_ob = (ob - ob_mean) / (ob_std + 1e-8)
            ob_input = np.concatenate([normal_ob, np.array(latent)])
            #action_org = self.policy.act(normal_ob)
            action_org = self.policy.act(ob_input)
            ob, reward, done, _ = self.env.step(action_org)
            total_reward += reward
            steps += 1
            if done:
                break
        
        t2 = time.time()

        return {'reward': total_reward, 'step': steps}

    def get_base_cases(self):

        return self.env.get_base_cases()
		
    '''	
    def find_c(self, case_id, weight, ob_mean, ob_std):
        # update latent var of one case
        case = FAULT_CASES_ALL[case_id]
        self.env.validate(0, case[0], case[1], case[2])
        self.policy.update_weights(weight)
		
        if self.policy_type == 'LSTM':
            self.policy.reset()
        
        self.latent_optimizer = UPOptimizer(env=self.env, case=case, policy=self.policy, latent_dim=LATENT_DIM, verbose=False,
                                            bayesian_opt=True, ob_mean=ob_mean, ob_std=ob_std)

        if EIOC:        
            self.latent_optimizer.optimize(maxiter=3, max_steps=200000, custom_bound=[-1.0, 1.0])
        else:        
            self.latent_optimizer.optimize(maxiter=20, max_steps=200000, custom_bound=[-1.0, 1.0])
        optimized_embedding = self.latent_optimizer.best_x

        #return (case_id, optimized_embedding)
        return {'case_id': case_id, 'optimized_embedding': optimized_embedding}
    '''

    def return_RS(self):
        return self.RS

    def close_java_connection(self):
        self.env.close_connection()
        print('connection with Ipss Server is closed')
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
                 num_cases_torun=None,
                 java_port=JAVA_PORT):

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        self.delta_std = delta_std
        self.rollout_length = rollout_length
        self.java_port = java_port
        # number of slaves is the same as number of fault-cases
        if num_cases_torun == None:
            self.num_slaves = len(PF_FAULT_CASES_ALL)
        else:
            self.num_slaves = num_cases_torun #round(num_cases_torun/3)*3  # This makes sure the cases during the traing to be
                                                          # selecte number of fault buses * 3 (3 different fault time duration)

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
                                            policy_params=self.policy_params,
                                            java_port=self.java_port + i + 1))
            time.sleep(5)
    def update_delta_std(self, delta_std):
        self.delta_std = delta_std

    def onedirction_rollout_multi_single_slavers(self, w_policy, ob_mean, ob_std, select_fault_cases_tuples, latent_array, evaluate=False):
        rollout_rewards, deltas_idx = [], []
        steps = 0
        
		# the ob_mean and ob_std is already id in the ray global table, no need to put it again, renke modified
        #ob_mean = ray.1put(ob_mean) 
        #ob_std = ray.1put(ob_std)
		
        # use RS to collect observation states from SingleRolloutSlavers; restart each iter
        self.RS = RunningStat(shape=(OB_DIM,))

        if evaluate:
            deltas_idx.append(-1)
            weights_id = ray.put(w_policy)

            #repeat_num = len(FAULT_CASES) // self.num_slaves
            #remain_num = len(FAULT_CASES) % self.num_slaves

            reward_ids = []
            for i in range(self.num_slaves):
                fault_tuple_tmp=select_fault_cases_tuples[i]
                #fault_tuple_idx=PF_FAULT_CASES_ALL.index(fault_tuple_tmp)
                reward_ids.append( self.single_rollout_slavers[i].single_rollout.remote(fault_tuple=fault_tuple_tmp,
                                    weights=weights_id, ob_mean=ob_mean, ob_std=ob_std,
                                    latent=latent_array[ fault_tuple_tmp[0] ] ) )

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


            t1 = time.time()

            # compute reward and number of timesteps used for positive perturbation rollout    
            w_pos_id = ray.put(w_policy + delta)
            '''
            pos_list_ids = [self.single_rollout_slavers[i].single_rollout.remote(
                fault_tuple=select_fault_cases_tuples[i], weights=w_pos_id, ob_mean=ob_mean, ob_std=ob_std) \
                for i in range(self.num_slaves)]
            '''				
            pos_list_ids = []
            for i in range(self.num_slaves):
                fault_tuple_tmp=select_fault_cases_tuples[i]
                #fault_tuple_idx=PF_FAULT_CASES_ALL.index(fault_tuple_tmp)
                pos_list_ids.append( self.single_rollout_slavers[i].single_rollout.remote(fault_tuple=fault_tuple_tmp,
                                     weights=w_pos_id, ob_mean=ob_mean, ob_std=ob_std,
                                     latent=latent_array[fault_tuple_tmp[0]] ) )
            
            pos_results  = ray.get(pos_list_ids)
            for result in pos_results:
                pos_rewards_list.append(result['reward'])
                pos_steps_list.append(result['step'])
            pos_reward = mean(pos_rewards_list)
            pos_steps = max(pos_steps_list)

            # get RS of SingleRolloutSlavers
            pos_RS_ids = [self.single_rollout_slavers[i].return_RS.remote() for i in range(self.num_slaves)]
            pos_RSs = ray.get(pos_RS_ids)

            # compute reward and number of timesteps used for negative pertubation rollout 
           
            w_neg_id = ray.put(w_policy - delta)
            ''' 
            neg_list_ids = [self.single_rollout_slavers[i].single_rollout.remote(
                fault_tuple=select_fault_cases_tuples[i], weights=w_neg_id, ob_mean=ob_mean, ob_std=ob_std) \
                for i in range(self.num_slaves)]
            ''' 				
            neg_list_ids = []
            for i in range(self.num_slaves):
                fault_tuple_tmp=select_fault_cases_tuples[i]
                #fault_tuple_idx=PF_FAULT_CASES_ALL.index(fault_tuple_tmp)
                neg_list_ids.append( self.single_rollout_slavers[i].single_rollout.remote(fault_tuple=fault_tuple_tmp,
                                     weights=w_neg_id, ob_mean=ob_mean, ob_std=ob_std,
                                     latent=latent_array[fault_tuple_tmp[0]] ) )
            
            neg_results  = ray.get(neg_list_ids)
            for result in neg_results:
                neg_rewards_list.append(result['reward'])
                neg_steps_list.append(result['step'])
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

            return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps, 'time': t2-t1}

    def calculate_c(self, pf_cases_idx, weight, ob_mean, ob_std):

        # --------------new written by Renke-------------------------------------
        if self.num_slaves < len(ONLY_FAULT_CASES_ALL):
            print ('\n !!!!!---------------attention, the latent optimization could not evaluate all fault cases for one power flow !!!!\n')

        self.latent_optimizer = UPOptimizer(single_rollout_slavers=self.single_rollout_slavers, pf_cases_idx=pf_cases_idx, weight=weight,
                                            latent_dim=LATENT_DIM, verbose=False,
                                            bayesian_opt=True, ob_mean=ob_mean, ob_std=ob_std,
                                            pfcases=POWERFLOW_CANDIDATES, faultcases=ONLY_FAULT_CASES_ALL)

        self.latent_optimizer.optimize(maxiter=MSO_OPT_MAX_ITER, max_steps=200000, custom_bound=[-1.0, 1.0])

        optimized_embedding = self.latent_optimizer.best_x

        return (pf_cases_idx, optimized_embedding)

    def get_base_cases(self):

        basecasesid = self.single_rollout_slavers[0].get_base_cases.remote()
        basecases = ray.get(basecasesid)

        return basecases


    def get_filter(self):
        return self.RS
        

    def close_java_connection(self):
        close_ids = [slave.close_java_connection.remote() for slave in self.single_rollout_slavers]
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
                 inner_iter=None,
                 update_cases=None,
                 onedirection_numofcasestorun=None,
                 cores=None,
                 policy_file = None,
                 save_per_iter = None):

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
        self.inner_iter = inner_iter
        self.latent_array = [np.zeros(LATENT_DIM) for _ in range(len(POWERFLOW_CANDIDATES))]
        self.num_c_update = min(update_cases, len(POWERFLOW_CANDIDATES))
        self.difficult_fault_buses_idx = []
        self.difficult_fault_buses_idx_dict = {}
        self.start_from_exist_policy = False
        self.current_ars_iter_num = 0
        self.save_per_iter = save_per_iter
        
        if policy_file != "":
            self.RS._M = np.zeros(OB_DIM, dtype = np.float64)
            self.RS._S = np.zeros(OB_DIM, dtype = np.float64)
            w_mean_temp = policy_params['p_M'].copy()
            w_std_temp = policy_params['p_S'].copy()
            self.RS._M[:] = w_mean_temp[:]
            self.RS._S[:] = w_std_temp[:]
            self.RS._n = policy_params['p_n']
            
            for itmp in range(len(self.latent_array)):
                self.latent_array[itmp][:] = policy_params['p_latent_array'][itmp,:]
				
            print ('\n----------------------updated latent_array: \n') 
            print (self.latent_array) 
            print ('\n--------------------------------------------\n')

            self.start_from_exist_policy = True

        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id))
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.')
		
        self.onedirection_numofcasestorun = min(onedirection_numofcasestorun, len(PF_FAULT_CASES_ALL))
        #self.onedirection_numofcasestorun = round(self.onedirection_numofcasestorun/3)*3 # This makes sure the cases during the traing to be
                                                          # selecte number of fault buses * 3 (3 different fault time duration) 
        if cores >= num_deltas * self.onedirection_numofcasestorun:
            self.num_direction_workers = num_deltas
            self.repeat = 1
            self.remain = 0
        else:
            self.num_direction_workers = cores // self.onedirection_numofcasestorun
            self.repeat = num_deltas // self.num_direction_workers
            self.remain = num_deltas % self.num_direction_workers												  
        #self.num_direction_workers = num_deltas
        self.workers = [OneDirectionWorker.remote(seed + 7 * i,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std,
                                      num_cases_torun = self.onedirection_numofcasestorun,
                                      java_port=JAVA_PORT + len(PF_FAULT_CASES_ALL) * i,
                                      ) for i in range(self.num_direction_workers)]

        time.sleep(60.0)

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

        basecases_id = self.workers[0].get_base_cases.remote()

        base_cases_lst = ray.get(basecases_id)

        print ('\n------- base power flow cases -------\n')
        print(base_cases_lst)
        print ('\n----------------------------------\n')


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
        latent_arr_id = ray.put(self.latent_array)

        if evaluate:

            repeat_num = len(PF_FAULT_CASES_ALL) // self.onedirection_numofcasestorun
            remain_num = len(PF_FAULT_CASES_ALL) % self.onedirection_numofcasestorun

            rollout_ids = []
            for repeat in range(repeat_num):
                rollout_ids += [self.workers[repeat].onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std,
                    select_fault_cases_tuples = PF_FAULT_CASES_ALL[repeat * self.onedirection_numofcasestorun: (repeat + 1) * self.onedirection_numofcasestorun:],
                    latent_array = latent_arr_id, evaluate=True )]

            if remain_num != 0:
                rollout_ids += [self.workers[repeat_num].onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std,
                    select_fault_cases_tuples = PF_FAULT_CASES_ALL[repeat_num * self.onedirection_numofcasestorun:],
                    latent_array = latent_arr_id,  evaluate=True )]

            #reward, reward_list = ray.get(rollout_ids)

            # gather results
            results_list = ray.get(rollout_ids)
			
            rewards_list = []
            for result in results_list:
                rewards_list += result['reward_list']

            reward_eval_avr = mean(rewards_list)
			
            del policy_id
            del ob_mean
            del ob_std
            del latent_arr_id
			
            return reward_eval_avr, rewards_list

        t1 = time.time()
		
        # randonmly sample the buses from FAULT_BUS_CANDIDATES
        # based on Jie's suggestion, we randomly select the same cases to run for each direction at each iteration
        # which means, for each iteration, the randomly selected cases are the same for all the directions,
        # but for different iterations, we will select different cases

        select_faultbus_num = SELECT_FAULT_BUS_PER_DIRECTION
        select_faultbuses_id = np.random.choice(FAULT_BUS_CANDIDATES, size=select_faultbus_num, replace=False)
        print("select_faultbuses_id:  ", select_faultbuses_id)

        # ----------this is the tentative code to enforcen the difficult bus faults to be learned for the ARS
        # ----------by adding those difficult buses directly to the select_faultbuses_id list
        # ----------self.difficult_fault_buses_idx is updated when Evaluation is true

        if self.start_from_exist_policy:
            if self.current_ars_iter_num > self.save_per_iter:
                print("difficult_fault_buses_idx_dict:  ", self.difficult_fault_buses_idx_dict)
                print("---------------------------------------------")
                print("difficult_fault_buses_idx:  ", self.difficult_fault_buses_idx)
                print("---------------------------------------------")
                # it_tmp = 0
                for busidx in self.difficult_fault_buses_idx:
                    if (busidx not in select_faultbuses_id):
                        for it_tmp in range(select_faultbus_num):
                            if select_faultbuses_id[it_tmp] not in self.difficult_fault_buses_idx:
                                select_faultbuses_id[it_tmp] = busidx
                                break

                print("select_faultbuses_id after enforcing difficult buses:  ", select_faultbuses_id)
        else:
            if self.current_ars_iter_num >= START_HARD_CASE_ITR:
                print("difficult_fault_buses_idx_dict:  ", self.difficult_fault_buses_idx_dict)
                print("---------------------------------------------")
                print("difficult_fault_buses_idx:  ", self.difficult_fault_buses_idx)
                print("---------------------------------------------")
                # it_tmp = 0
                for busidx in self.difficult_fault_buses_idx:
                    if (busidx not in select_faultbuses_id):
                        for it_tmp in range(select_faultbus_num):
                            if select_faultbuses_id[it_tmp] not in self.difficult_fault_buses_idx:
                                select_faultbuses_id[it_tmp] = busidx
                                break

                print("select_faultbuses_id after enforcing difficult buses:  ", select_faultbuses_id)

        select_pf_num = SELECT_PF_PER_DIRECTION
        select_pfcases_id = np.random.choice(POWERFLOW_CANDIDATES, size=select_pf_num, replace=False)
        print("select_pfcases_id:  ", select_pfcases_id)

        select_fault_cases_tuples = [(select_pfcases_id[k], select_faultbuses_id[i], FAULT_START_TIME, FTD_CANDIDATES[j])
                                        for k in range(len(select_pfcases_id))
                                        for i in range(len(select_faultbuses_id))
                                        for j in range(len(FTD_CANDIDATES))]
        select_fault_cases_tuples_id = ray.put(select_fault_cases_tuples)
        print ("select_fault_cases_tuples:  ", select_fault_cases_tuples) 
		
        rollout_id_list = []
        for repeat in range(self.repeat):										 																						  
            rollout_id_list += [worker.onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std, select_fault_cases_tuples_id, \
                                                      latent_arr_id, evaluate=evaluate) for worker in self.workers]
													  
        rollout_id_list += [worker.onedirction_rollout_multi_single_slavers.remote(policy_id, ob_mean, ob_std, select_fault_cases_tuples_id, \
                            latent_arr_id, evaluate=evaluate) for worker in self.workers[:self.remain]]											   											
        # gather results
        results_list = ray.get(rollout_id_list)
        rollout_rewards, deltas_idx = [], []

        for result in results_list:
            self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()
        print('Time to generate rollouts:', t2 - t1)
        
        # update RS from all workers
        for j in range(self.num_direction_workers):
            self.RS.update(ray.get(self.workers[j].get_filter.remote()))

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
        del latent_arr_id
        del select_fault_cases_tuples_id
		
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat

    def train_step(self):
        """
        Perform one update step of the policy weights.
        """

        g_hat = self.aggregate_rollouts()
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat, self.step_size).reshape(self.w_policy.shape)
        return

    def train(self, num_iter, save_per_iter, tol_p, tol_steps):

        rewardlist = []
        last_reward = 0.0 
        n           = int(tol_steps/save_per_iter)
       
        start = time.time()
        for i in range(num_iter):

            self.current_ars_iter_num = i
            
            self.step_size *= self.decay
            self.delta_std *= self.decay

            for worker in self.workers:
                worker.update_delta_std.remote(self.delta_std)

#----------------------latent optimization starts here----------------------------------

            if (i + 1) % self.inner_iter == 0:
                # update latent var of randomly selected few cases
                t1 = time.time()
                policy_id = ray.put(self.w_policy)
                ob_mean = ray.put(self.RS.mean)
                ob_std = ray.put(self.RS.std)

                all_powerflow_cases_idx = list(range(len(POWERFLOW_CANDIDATES)))
                pf_cases_idx = np.random.choice(all_powerflow_cases_idx, size=self.num_c_update, replace=False)
                print("\n ################################# \n")
                print("power flow cases to update latent var", pf_cases_idx)
                print("\n ################################# \n")
                # update_c_id = self.workers[0].calculate_c.remote(pf_cases_id, policy_id, ob_mean, ob_std)

                update_c_id = []
                latent_work_repeat = len(pf_cases_idx) // self.num_direction_workers
                latent_work_remain = len(pf_cases_idx) % self.num_direction_workers
                # pfidx = 0
                for repeat in range(latent_work_repeat):
                    # update_c_id += [self.workers[pfidx].calculate_c.remote(pf_cases_idx[pfidx], policy_id, ob_mean, ob_std) for pfidx in range(len(pf_cases_idx))]

                    update_c_id += [
                        worker.calculate_c.remote(pf_cases_idx[repeat * self.num_direction_workers + idx], policy_id, ob_mean,
                                                  ob_std) for idx, worker in enumerate(self.workers)]
                    # pfidx += 1

                update_c_id += [
                    worker.calculate_c.remote(pf_cases_idx[latent_work_repeat * self.num_direction_workers + idx], policy_id,
                                              ob_mean, ob_std) for idx, worker in enumerate(self.workers[:latent_work_remain])]

                update_d = ray.get(update_c_id)
                print("\n ==============================================\n") 
                print('-------------length of update_d: ', len(update_d) )	
                print("\n ==============================================\n")
                print('-------------update_d: ')					
                print(update_d)
                print("\n ============================================== \n")
				
                for case, new_c in update_d:
                    self.latent_array[case] = new_c

                del policy_id 
                del ob_mean 
                del ob_std 
					
                t2 = time.time()
                print('total time of one time of latent optimization', t2 - t1)
                #print('total time of one time of latent optimization', t2 - t1)

#----------------------latent optimization ends here----------------------------------

            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)
            print('iter ', i, ' done')
            

            # record statistics every x iterations
            if (i + 1) % save_per_iter == 0:
                reward, reward_list = self.aggregate_rollouts(evaluate=True)
                #print('\n---------!!! evaluation, reward list: \n')
                #print(reward_list)
                #print('\n----------------------------------------\n')
                w = [self.w_policy, self.RS.mean, self.RS.std, self.RS._n, self.RS._M, self.RS._S, self.step_size, self.delta_std]
                np.savez(self.logdir + "/nonlinear_policy_plus" + str(i + 1), w)
                np.savez(self.logdir + "/latent" + str(i + 1), self.latent_array)
                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", reward)
                for f_index in range(len(PF_FAULT_CASES_ALL)):
                    logz.log_tabular("reward {}:".format(f_index), reward_list[f_index])
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()

                self.difficult_fault_buses_idx_dict = {}
                self.difficult_fault_buses_idx = []
                for i_tmp in range(len(reward_list)):
                    if reward_list[i_tmp] < -10000.0:
                        fault_idx_tmp = i_tmp % len(FAULT_BUS_CANDIDATES)
                        fault_idx_key = FAULT_BUS_CANDIDATES[fault_idx_tmp]
                        if fault_idx_key in self.difficult_fault_buses_idx_dict.keys():
                            self.difficult_fault_buses_idx_dict[fault_idx_key] = self.difficult_fault_buses_idx_dict[fault_idx_key]+1
                        else:
                            self.difficult_fault_buses_idx_dict[fault_idx_key] = 1

                # sort the dict
                if self.difficult_fault_buses_idx_dict: # if the dict is not empty

                     tmp_sort_lst= sorted(self.difficult_fault_buses_idx_dict.items(),
                                                            key = lambda x:x[1],reverse = True)
                     tmp_len = min(len(tmp_sort_lst), NUM_HARD_FAULT_ADDED)

                     for i_tmp in range(tmp_len):
                        self.difficult_fault_buses_idx.append(tmp_sort_lst[i_tmp][0])

                print('---------------After evulation, here are the difficult fault bus indices: ')
                print('---------------difficult_fault_buses_idx_dict: ')
                print(self.difficult_fault_buses_idx_dict)
                print('---------------difficult_fault_buses_idx: ')
                print(self.difficult_fault_buses_idx)

                
                '''
                if last_reward != 0.0:
                    tol = abs((reward - last_reward)/last_reward) <= tol_p
                    rewardlist.append(tol)
                    if len(rewardlist) >= n and all(rewardlist[-5:]):
                        break
                '''
                last_reward = reward
                
        # close java connection for slaves
        close_ids = [worker.close_java_connection.remote() for worker in self.workers]
        ray.get(close_ids)
        return
    

def run_ars(params):
    logdir = params['dir_path']
    version_num = 0
    while os.path.exists(logdir + "_v" + str(version_num)):
        version_num += 1
    logdir = logdir + "_v" + str(version_num)
    os.makedirs(logdir)
    
    if params['policy_file'] != "":
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
		
        if params['latent_file'] != "":
            latent_src = np.load(params['latent_file'], allow_pickle = True)
            latent_cont = latent_src['arr_0'].copy()
        
        policy_params = {'type': params['policy_type'],
                     'policy_network_size': params['policy_network_size'],
                     'ob_dim': OB_DIM,
                     'ac_dim': AC_DIM,
                     'latent_dim': LATENT_DIM,
                     'weights': w_M,
                     'p_n':p_n,
                     'p_M':p_M,
                     'p_S':p_S,
                     'p_step_size':p_step_size,
                     'p_delta_std':p_delta_std,
					 'p_latent_array': latent_cont}
    else:
        # set policy parameters. 
        policy_params = {'type': params['policy_type'],
                         'policy_network_size': params['policy_network_size'],
                         'ob_dim': OB_DIM,
                         'ac_dim': AC_DIM, 'latent_dim': LATENT_DIM}

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
                     inner_iter=params['inner_iter'],
                     update_cases=params['update_cases'],
                     onedirection_numofcasestorun=params['onedirection_numofcasestorun'],
                     cores=params['cores'], 
                     policy_file =params['policy_file'],
                     save_per_iter = params['save_per_iter'])

    ARS.train(params['n_iter'], params['save_per_iter'], params['tol_p'], params['tol_steps'])
    return


# Setting
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', '-n', type=int, default=800)  # Number of iterations
    if EIOC: # EIOC test
        parser.add_argument('--n_directions', '-nd', type=int, default=3)  
    else:    # Constance training
        parser.add_argument('--n_directions', '-nd', type=int, default=128)  
    if EIOC: # EIOC test
        parser.add_argument('--deltas_used', '-du', type=int, default=2)
    else:    # Constance training
        parser.add_argument('--deltas_used', '-du', type=int, default=64)
    parser.add_argument('--step_size', '-s', type=float, default=1)  # default=0.02
    parser.add_argument('--delta_std', '-std', type=float, default=2)  # default=.03
    parser.add_argument('--decay', type=float, default=0.999)  # decay of step_size and delta_std
    parser.add_argument('--rollout_length', '-r', type=int, default=90)
    parser.add_argument('--seed', type=int, default=374) # Seed Number for randomization
    parser.add_argument('--policy_type', type=str, default='LSTM')
    save_results_path = 'MSO_300bus_pfonly_pf%dof%d_fault%dof%d'%(SELECT_PF_PER_DIRECTION,
                        len(POWERFLOW_CANDIDATES), SELECT_FAULT_BUS_PER_DIRECTION, len(FAULT_BUS_CANDIDATES))
    parser.add_argument('--dir_path', type=str,
                        default=save_results_path)  # Folder Name for outputs
    if EIOC: # EIOC test
        parser.add_argument('--save_per_iter', type=int, default=1)  # save the .npz file per x iterations
    else:    # Constance training
        parser.add_argument('--save_per_iter', type=int, default=1)  # save the .npz file per x iterations
    parser.add_argument('--policy_network_size', type=list, default=[64, 64])
    if EIOC: # EIOC test
        parser.add_argument('--inner_iter', type=int, default=1) # update latent var every x iterations
    else:    # Constance training
        parser.add_argument('--inner_iter', type=int, default=20) # update latent var every x iterations
		
    parser.add_argument('--update_cases', type=int, default=len(POWERFLOW_CANDIDATES)) # how much cases to randomly choose for latent var update

    # onedirection_cases_torun: For each direction, how many cases to run, default is to run all the FAULT_CASES,
    # but we could also randonmly select part of the FAULT_CASES, this para defines how many cases to randomly select from FAULT_CASES

        #select_faultbus_num = 3  # how many fault buses to select each iteration of ARS
    casestorunperdirct = SELECT_PF_PER_DIRECTION*len(FTD_CANDIDATES)*SELECT_FAULT_BUS_PER_DIRECTION
    parser.add_argument('--onedirection_numofcasestorun', type=int, default=casestorunperdirct)
    
    # the parameters controlling iterations for convergence
    # n_iter is maximum number of iterations
    # tol_p is tolerance for the percentage of average reward change between iterations
    # tol_steps is the total iterations for maintaining tol_p
    parser.add_argument('--tol_p', type=float, default=0.05)
    parser.add_argument('--tol_steps', type=int, default=50)

    # please specify number of cores here; it is to help determine the number of workers
    if EIOC: # EIOC test
        parser.add_argument('--cores', type=int, default=NUM_OF_GCP_CORES) # how many cores available as of hardware, EIOC test
    else:    # Constance training
        parser.add_argument('--cores', type=int, default=NUM_OF_GCP_CORES) # how many cores available as of hardware
    
    parser.add_argument('--policy_file', type=str, default="")
    parser.add_argument('--latent_file', type=str, default="")
    
    																							  
    local_ip = socket.gethostbyname(socket.gethostname())

    # Init Ray in Cluster, use log_to_driver=True for printing messages from remote nodes
##!!#####################################################################################################
    # !! if you are using EIOC, use this line of code below
    if EIOC: # EIOC test
        ray.init(redis_address="localhost:6379", log_to_driver=False)
    else:
    # !! if you are using GCP, use this line of code below															 																									  
        #ray.init(temp_dir=os.environ["tmpfolder"], redis_address=os.environ["ip_head"], log_to_driver=False)
        ray.init(address="localhost:6379", log_to_driver=False)
##!!######################################################################################################
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)
