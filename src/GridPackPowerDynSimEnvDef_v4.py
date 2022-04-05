'''
Authors: Renke Huang, Qiuhua Huang
Contact: qiuhua.huang@pnnl.gov

'''

import sys, os
import gridpack
import time
import gridpack.hadrec
import numpy as np
import random
from gym.utils import seeding
from gym import spaces
import gym
import math
import json
import xmltodict
import collections

def xml2dict(input_xmlfile):
    file_object = open(input_xmlfile,encoding = 'utf-8')
    try:
        all_the_xmlStr = file_object.read()
    finally:
        file_object.close()
        
    #xml To dict
    convertedDict = xmltodict.parse(all_the_xmlStr)

    return convertedDict

def scale_action(action_space, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0

def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))

class GridPackPowerDynSimEnv(gym.Env):
    
    # to save the original action low and high values to help rescale them back before applying to the environment
    original_action_space = None
    is_action_space_scaled =False
    
    # number of power flow base cases
    total_case_num = 1
    
    default_output_ob_time_step = 0.02
    
    def __init__(self, simu_input_file, rl_config_file, force_symmetric_continuous_action=False, trip_line_after_fault=False, data_path=None, verbose=0):
        
        self.simu_input_file = simu_input_file
        self.rl_config_file = rl_config_file
        self.data_path = data_path
        self.force_symmetric_continuous_action = force_symmetric_continuous_action
        self.trip_line_after_fault = trip_line_after_fault
        
        #----------get base power flow cases, if data_path is provided
        self.base_pf_cases = []
        if data_path is not None:
            filelist = os.listdir(self.data_path)
            
            for file in filelist:
                if file[-4:] == ".raw":
                    self.base_pf_cases.append(file)

        if len(self.base_pf_cases) > 1:
            self.total_case_num = len(self.base_pf_cases)
        
        #----------------process xml input file, read necessary parameters from gridpack input xml file-------------------
        self.input_xml_dict = xml2dict(simu_input_file)
        self.simu_time_step = float(self.input_xml_dict["Configuration"]["Dynamic_simulation"]["timeStep"])
        self.fault_start_time = float( self.input_xml_dict["Configuration"]["Dynamic_simulation"]["faultEvents"]["faultEvent"]["beginFault"] )
        self.fault_duration_time = float( self.input_xml_dict["Configuration"]["Dynamic_simulation"]["faultEvents"]["faultEvent"]["endFault"] ) \
                                    - self.fault_start_time
        tmpstr= self.input_xml_dict["Configuration"]["Dynamic_simulation"]["faultEvents"]["faultEvent"]["faultBranch"] 
        tmpstrlist = tmpstr.split()        
        self.fault_bus_no = int(tmpstrlist[0])
        
        self.outputob_nsimustep = int(self.default_output_ob_time_step/self.simu_time_step)
        
        #----------------process json file, read necessary parameters from json RL file------------------------
        with open(rl_config_file, 'r') as f:
            self.rl_config_dict = json.loads(f.read())
            
        self.faultbus_candidates =  []
        tmpfaultbus = self.rl_config_dict["faultBusCandidates"]
        for ibus in range(len(tmpfaultbus)):
            self.faultbus_candidates.append( int(tmpfaultbus[ibus][3:]) )
            
        #print ("faultBusCandidates: ", self.faultbus_candidates)  
        
        self.action_buses =  []
        tmpbus = self.rl_config_dict["actionScopeAry"]
        for ibus in range(len(tmpbus)):
            self.action_buses.append( int(tmpbus[ibus][3:]) )
            
        #print ("action buses: ", self.action_buses) 
        
        self.faultStartTimeCandidates   = self.rl_config_dict["faultStartTimeCandidates"]
        self.faultDurationCandidates    = self.rl_config_dict["faultDurationCandidates"]
        self.env_time_step              = self.rl_config_dict["envStepTimeInSec"]
        self.unstableReward             = self.rl_config_dict["unstableReward"]
        self.actionPenalty              = self.rl_config_dict["actionPenalty"]
        self.invalidActionPenalty       = self.rl_config_dict["invalidActionPenalty"]
        self.preFaultActionPenalty      = self.rl_config_dict["preFaultActionPenalty"]
        self.observationWeight          = self.rl_config_dict["observationWeight"]
        self.minVoltRecoveryLevel       = self.rl_config_dict["minVoltRecoveryLevel"]
        self.maxVoltRecoveryTime        = self.rl_config_dict["maxVoltRecoveryTime"]
        self.observation_history_length = self.rl_config_dict["historyObservationSize"]
        self.action_ranges              = np.array(self.rl_config_dict["actionValueRanges"])
        self.action_buses_pvalue_pu     = np.array(self.rl_config_dict["actionBusesPValue"])/100.0
                
        # ------------start initialization GridPACK module-----------------------
        noprintflag = gridpack.NoPrint()
        noprintflag.setStatus(True)
        self.run_gridpack_necessary_env = gridpack.Environment()
        self.hadapp = gridpack.hadrec.Module()
        
        self.hadapp.solvePowerFlowBeforeDynSimu(simu_input_file, 0)
        self.hadapp.transferPFtoDS()
        
        busfaultlist = gridpack.dynamic_simulation.EventVector()

        self.hadapp.initializeDynSimu(busfaultlist)
        
        (self.obs_genBus, self.obs_genIDs, self.obs_loadBuses, self.obs_loadIDs, self.obs_busIDs) = self.hadapp.getObservationLists()
        
        self.nobbus = len(self.obs_busIDs)
        self.nobload = len(self.obs_loadBuses)

        #--------------update the action load value array--------------------
        itmp = 0
        for ibus in self.action_buses:
            totalloadtmp = self.hadapp.getBusTotalLoadPower(ibus)[0]
            self.action_buses_pvalue_pu[itmp] = totalloadtmp/100.0
            itmp = itmp + 1
            #print('   action load bus: %d, total load: ' % (ibus), totalloadtmp)

        #---------execute one simulation step to get observations
        self.current_simu_time = 0.0
        self.hadapp.executeDynSimuOneStep()
        self.current_simu_time = self.simu_time_step
        
        #---------the following observation space definition is only valid for load shedding action cases
        self.observation_space_dim = self.nobbus + self.nobload
        
        #---------set-up the internal action value range
        #print ('action value ranges  = ', self.action_ranges)
        low = self.action_ranges[:,0]
        high = self.action_ranges[:,1]
            
        #print ('action range low =', low, 'action range high =', high)
        #print ('low shape:', np.shape(low))
        
        self.action_space = spaces.Box(low, high, dtype=self.action_ranges.dtype)
        
        if force_symmetric_continuous_action:  # i.e., force np.abs(low) == high
                if not (np.abs(low) == high).all():
                    #print('!!Warming: the original action space is non-symmetric, convert it to [-1,1] for each action')
                    self.original_action_space = spaces.Box(low, high, dtype=self.action_ranges.dtype)
                    ones = np.ones_like(low)
                    self.action_space = spaces.Box(-ones, ones, dtype=self.action_ranges.dtype)
                    self.is_action_space_scaled = True
        #print (self.observation_history_length, self.observation_space_dim)
        
        self.observation_space = spaces.Box(-999,999,shape=(self.observation_history_length * self.observation_space_dim,)) 

        #print ('obs shape[0]',self.observation_space.shape[0])
        self.seed()

        #TOOD get the initial states
        self.ob_vals = collections.deque(maxlen=self.observation_history_length)

        self.steps_beyond_done = None
        self.restart_simulation = True
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ## This is a tentative solution of the error of additional dimension is added to the returned
        # action in OpenAI Gym DDPG
        if not self.action_space.contains(action):

            print ('-------------!!!!error, action provided by AI agent not in action space!!!!')
            print('-------------  action obtained from AI agent:')
            print (action)
            print('-------------  observations as input into the AI agent:')
            print (np.array(self.ob_vals).ravel())
            action = action[0]

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        done = False
        #----------if actually the simulation returns done at previous steps, but the step function is still called
        if (self.steps_beyond_done is not None) and (self.steps_beyond_done >= 0) :
            
            #logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
            done = True
            
            return np.array(self.ob_vals).ravel(), reward, done, {}

        actionPyAry = np.asarray(action,dtype = np.float64)

        if self.is_action_space_scaled and self.original_action_space is not None:
            #Rescale the action from [-1, 1] to [low, high]
            actionPyAry = unscale_action(self.original_action_space, actionPyAry)
        
        if len(action) != len(self.action_buses):
            print ('-----------!!!!error for action size:!!!!!!!!---------')
            
        reward = 0.0
        
        remain_load = self.ob_vals[-1][self.nobbus:(self.nobbus+self.nobload)]
        
        loadshedact = gridpack.hadrec.Action()  
        for iact in range(len(action)):
            if ( abs(actionPyAry[iact]) > 0.01 ):  # add action if shed load > 0.01%                
                loadshedact.actiontype = 0
                loadshedact.bus_number = self.action_buses[iact]
                loadshedact.componentID = "1"
                loadshedact.percentage = actionPyAry[iact]
                self.hadapp.applyAction(loadshedact)
                
                # -----------check whether the action is applied before fault_start_time
                if self.current_simu_time < self.fault_start_time:
                    reward -= self.preFaultActionPenalty
                    
                # ---------check whether the action is an invalid action, need check qiuhua for his implementation here
                #if ( remain_load[iact] + actionPyAry[iact]) > 0.01: # if it is a valid action
                if ( remain_load[iact] ) > 0.01: # if it is a valid action    
                    reward += self.actionPenalty * actionPyAry[iact] * self.action_buses_pvalue_pu[iact]
                else:
                    reward -= self.invalidActionPenalty 
                    
        invalidact_rew =  reward           
                        
        for istep in range (int(self.env_time_step/self.simu_time_step)): 

            if ( 0.0<= (self.current_simu_time-(self.fault_start_time+self.fault_duration_time)) <=self.simu_time_step ):
                # need to apply the line trippin here
                if self.trip_line_after_fault:
                    linetripact = gridpack.hadrec.Action()
                    linetripact.actiontype = 1
                    linetripact.bus_number = self.fault_bus_no
                    linetripact.componentID = "1"
                    linetripact.percentage = 0.0
                    self.hadapp.applyAction(linetripact)

            self.hadapp.executeDynSimuOneStep()
            self.current_simu_time += self.simu_time_step
                        
            if istep%self.outputob_nsimustep == 1:
                ob_vals_full = self.hadapp.getObservations()
                ob_vals_tmp = list(range( self.nobbus + self.nobload ))
                ob_vals_tmp[0:self.nobbus] = ob_vals_full[0:self.nobbus]
                if self.nobload > 0:
                    ob_vals_tmp[self.nobbus:(self.nobbus+self.nobload)] = ob_vals_full[-self.nobload:]
                self.ob_vals.append(ob_vals_tmp)
                #after_getob_time = time.time()    
                #total_dataconv_time += (after_getob_time - before_getob_time)
                
            done = self.hadapp.isDynSimuDone()
            if done:
                break
            
        #--------compute the voltage deviation part for the reward    
        ob_volt_tmp = self.ob_vals[-1][0:self.nobbus]
        fault_end_time = self.fault_start_time + self.fault_duration_time
        
        for ivoltob in range (self.nobbus):
            if self.fault_start_time <= self.current_simu_time < fault_end_time:
                volt_penalty = 0.0
                
            elif (fault_end_time) <= self.current_simu_time < (fault_end_time + 0.33) :
                volt_penalty = min(0.0, ob_volt_tmp[ivoltob]-0.7)
                
            elif (self.current_simu_time < (fault_end_time + 0.5)) and (self.current_simu_time >= (fault_end_time + 0.33) ):
                volt_penalty = min(0.0, ob_volt_tmp[ivoltob]-0.8)
                
            elif (self.current_simu_time < (fault_end_time + 1.5)) and (self.current_simu_time >= (fault_end_time + 0.5) ):
                volt_penalty = min(0.0, ob_volt_tmp[ivoltob]-0.9)
                
            else:
                volt_penalty = min(0.0, ob_volt_tmp[ivoltob]- self.minVoltRecoveryLevel)
                
            reward += self.observationWeight * volt_penalty
        
        volt_rew = reward - invalidact_rew
        
        # --------- if the voltage could not be back at minVoltRecoveryLevel after fault_end_time + maxVoltRecoveryTime, done
        unstable_rew = 0.0
        for ivoltob in range (self.nobbus): # check with Qiuhua for his implementation     
            if (self.current_simu_time > (fault_end_time + self.maxVoltRecoveryTime)) and (ob_volt_tmp[ivoltob] < self.minVoltRecoveryLevel):   
                reward += self.unstableReward
                unstable_rew += self.unstableReward 
                done = True 
                #print ("------bus %d voltage still not recoverd to %f till %f second after fault clear, simulation done-------"%(self.obs_busIDs[ivoltob], self.minVoltRecoveryLevel, self.maxVoltRecoveryTime))
                break
        #---------------compute reward here-------------------    
        if done and self.steps_beyond_done is None:
            self.steps_beyond_done = 0

        return np.array(self.ob_vals).ravel(), reward, done, {}

    def reset(self):

        # reset need to randomize the operation state and fault location, and fault time

        case_Idx = np.random.randint(0, self.total_case_num) # an integer
        
        total_fault_buses = len(self.faultbus_candidates)

       
        fault_bus_idx = np.random.randint(0, total_fault_buses)# an integer, in the range of [0, total_bus_num-1]
        
        #fault_bus_idx = 3 # an integer, in the range of [0, total_bus_num-1]
        #fault_start_time =random.uniform(0.99, 1.01) # a double number, in the range of [0.2, 1]
        self.fault_start_time = self.faultStartTimeCandidates[np.random.randint(0, len(self.faultStartTimeCandidates))]
        
        self.fault_duration_time = self.faultDurationCandidates[np.random.randint(0, len(self.faultDurationCandidates))] # a double number, in the range of [0.08, 0.4]
        self.fault_bus_no = self.faultbus_candidates[fault_bus_idx]
        
        # ------reinitialize the hadrec module
        busfault = gridpack.dynamic_simulation.Event()
        busfault.start = self.fault_start_time
        busfault.end = self.fault_start_time + self.fault_duration_time
        busfault.step = self.simu_time_step
        busfault.isBus = True
        busfault.bus_idx = self.faultbus_candidates[fault_bus_idx]
    
        busfaultlist = gridpack.dynamic_simulation.EventVector([busfault])

        self.hadapp.solvePowerFlowBeforeDynSimu(self.simu_input_file, case_Idx)
        self.hadapp.transferPFtoDS()
        self.hadapp.initializeDynSimu(busfaultlist)

        #--------------update the action load value array--------------------
        itmp = 0
        for ibus in self.action_buses:
            totalloadtmp = self.hadapp.getBusTotalLoadPower(ibus)[0]
            self.action_buses_pvalue_pu[itmp] = totalloadtmp/100.0
            itmp = itmp + 1
            #print('   action load bus: %d, total load: ' % (ibus), totalloadtmp)
        
        #---------execute one simulation step to get observations
        self.current_simu_time = 0.0
        self.hadapp.executeDynSimuOneStep()
        self.current_simu_time = self.simu_time_step
        
        ob_vals_full = self.hadapp.getObservations()
        ob_tmp = list(range( self.nobbus + self.nobload ))
        ob_tmp[0:self.nobbus] = ob_vals_full[0:self.nobbus]
        if self.nobload > 0:
            ob_tmp[self.nobbus:(self.nobbus+self.nobload)] = ob_vals_full[-self.nobload:]
        
        self.ob_vals.clear()
        for itmp in range(self.observation_history_length):
            self.ob_vals.append(ob_tmp)

        self.steps_beyond_done = None
        self.restart_simulation = True

        return np.array(self.ob_vals).ravel()

    #---------------initialize the system with a specific state and fault
    def validate(self, case_Idx, fault_bus_idx, fault_start_time, fault_duration_time):

        # ------reinitialize the hadrec module
        self.fault_start_time = fault_start_time
        self.fault_duration_time = fault_duration_time
        self.fault_bus_no = self.faultbus_candidates[fault_bus_idx]
        
        busfault = gridpack.dynamic_simulation.Event()
        busfault.start = fault_start_time
        busfault.end = fault_start_time + fault_duration_time
        busfault.step = self.simu_time_step
        busfault.isBus = True
        busfault.bus_idx = self.faultbus_candidates[fault_bus_idx]
    
        busfaultlist = gridpack.dynamic_simulation.EventVector([busfault])

        #print('------debug, validate, before solvePowerFlowBeforeDynSimu, case_Idx = %d'%(case_Idx))
        self.hadapp.solvePowerFlowBeforeDynSimu(self.simu_input_file, case_Idx)

        #print ('------debug, validate, after solvePowerFlowBeforeDynSimu')
        self.hadapp.transferPFtoDS()
        #print('------debug, validate, after transferPFtoDS')
        self.hadapp.initializeDynSimu(busfaultlist)

        #--------------update the action load value array--------------------
        itmp = 0
        for ibus in self.action_buses:
            totalloadtmp = self.hadapp.getBusTotalLoadPower(ibus)[0]
            self.action_buses_pvalue_pu[itmp] = totalloadtmp/100.0
            itmp = itmp + 1
            #print('   action load bus: %d, total load: ' % (ibus), totalloadtmp)

        #--------execute one simulation step to get observations
        self.current_simu_time = 0.0
        self.hadapp.executeDynSimuOneStep()
        self.current_simu_time = self.simu_time_step
        
        ob_vals_full = self.hadapp.getObservations()
        ob_tmp = list(range( self.nobbus + self.nobload ))
        ob_tmp[0:self.nobbus] = ob_vals_full[0:self.nobbus]
        if self.nobload > 0:
            ob_tmp[self.nobbus:(self.nobbus+self.nobload)] = ob_vals_full[-self.nobload:]
        
        self.ob_vals.clear()
        for itmp in range(self.observation_history_length):
            self.ob_vals.append(ob_tmp)

        self.steps_beyond_done = None
        self.restart_simulation = True

        return np.array(self.ob_vals).ravel()
    
    def close_env(self):
        self.hadapp = None
        #self.run_gridpack_necessary_env = None
        print("--------- GridPACK HADREC APP MODULE deallocated ----------")

    def get_base_cases(self):
            
        return self.base_pf_cases   
        
        
        
    
    
    
    
    
    
    
    
    