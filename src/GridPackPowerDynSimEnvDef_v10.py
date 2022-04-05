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
    
    def __init__(self, simu_input_file, rl_config_file, force_symmetric_continuous_action=False,
                 apply_uvls_otherzones = True, apply_action_mask=True, apply_volt_filter=False,
                 allow_loadshed_before_fault=False, return_reward_details = False,
                 use_zone_info_as_ob = False, use_fault_bus_num = False, trip_line_after_fault=False,
                 data_path=None, selfzone_uvls=False, verbose=0):
        
        self.simu_input_file = simu_input_file
        self.rl_config_file = rl_config_file
        self.data_path = data_path
        self.force_symmetric_continuous_action = force_symmetric_continuous_action
        self.trip_line_after_fault = trip_line_after_fault
        self.apply_action_mask = apply_action_mask
        self.apply_uvls_otherzones = apply_uvls_otherzones
        self.use_fault_bus_num = use_fault_bus_num
        self.apply_volt_filter = apply_volt_filter
        self.use_zone_info_as_ob = use_zone_info_as_ob
        self.selfzone_uvls = selfzone_uvls
        self.allow_loadshed_before_fault = allow_loadshed_before_fault
        self.return_reward_details = return_reward_details

        self.loadshed_before_fault = 0 # 0, there is no load shed before fault for the simulation
                                       # 1, there exists load shed before fault for the simulation
        self.total_loadshed = 0.0
        self.total_loadshed_selfuvls = 0.0
        self.total_volt_vio = 0.0
        self.total_invalid_act_before = 0.0
        self.total_invalid_act = 0.0

        self.zone_load_p = []
        self.zone_load_q = []
        self.zone_num_info = []
        self.zone_generation_p = []
        self.zone_generation_q = []
        
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
        self.ObservationSelfZoneVoltNum     = self.rl_config_dict["ObservationSelfZoneVoltNum"]
        self.ObservationSelfZoneLowVoltNum = self.rl_config_dict["ObservationSelfZoneLowVoltNum"]
        self.ObservationOtherZoneVoltNum    = self.rl_config_dict["ObservationOtherZoneVoltNum"]
        self.actionLoadSelfZoneNum          = self.rl_config_dict["actionLoadSelfZoneNum"]
        self.actionLoadOtherZoneNum         = self.rl_config_dict["actionLoadOtherZoneNum"]

        self.action_buses_pvalue_pu     = np.zeros((len(self.action_buses)))
        #self.action_buses_pvalue_pu     = np.array(self.rl_config_dict["actionBusesPValue"])/100.0
                
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
        self.nob_load = len(self.obs_loadBuses)
        self.nob_volt = self.ObservationSelfZoneVoltNum - self.ObservationSelfZoneLowVoltNum

        # construct the voltage observation bus number to index of observation array dictionary
        self.ob_volt_busno_idx_dict = {self.obs_busIDs[i]: i for i in range(len(self.obs_busIDs))}

        #--------------update the action load value array--------------------
        itmp = 0
        for ibus in self.action_buses:
            totalloadtmp = self.hadapp.getBusTotalLoadPower(ibus)[0]
            self.action_buses_pvalue_pu[itmp] = totalloadtmp/100.0
            itmp = itmp + 1
            #print('   action load bus: %d, total load: ' % (ibus), totalloadtmp)

        self.selfzone_action_buses = self.action_buses[0:self.actionLoadSelfZoneNum]
        self.selfzone_action_ranges = self.action_ranges[0:self.actionLoadSelfZoneNum,:]
        self.selfzone_action_buses_pvalue_pu = self.action_buses_pvalue_pu[0:self.actionLoadSelfZoneNum]

        self.otherzones_action_buses = self.action_buses[self.actionLoadSelfZoneNum:]
        self.otherzones_action_ranges = self.action_ranges[self.actionLoadSelfZoneNum:,:]
        self.otherzones_action_buses_pvalue_pu = self.action_buses_pvalue_pu[self.actionLoadSelfZoneNum:]

        #---------execute one simulation step to get observations
        self.current_simu_time = 0.0
        self.hadapp.executeDynSimuOneStep()
        self.current_simu_time = self.simu_time_step
        self.current_env_steps = 0
        
        #---------the following observation space definition is only valid for load shedding action cases
        self.observation_space_dim = self.nob_volt + self.nob_load
        
        #---------set-up the internal action value range
        #print ('action value ranges  = ', self.action_ranges)
        low = self.selfzone_action_ranges[:,0]
        high = self.selfzone_action_ranges[:,1]
            
        #print ('action range low =', low, 'action range high =', high)
        #print ('low shape:', np.shape(low))
        
        self.action_space = spaces.Box(low, high, dtype=self.selfzone_action_ranges.dtype)
        
        if force_symmetric_continuous_action:  # i.e., force np.abs(low) == high
                if not (np.abs(low) == high).all():
                    #print('!!Warming: the original action space is non-symmetric, convert it to [-1,1] for each action')
                    self.original_action_space = spaces.Box(low, high, dtype=self.selfzone_action_ranges.dtype)
                    ones = np.ones_like(low)
                    self.action_space = spaces.Box(-ones, ones, dtype=self.selfzone_action_ranges.dtype)
                    self.is_action_space_scaled = True
        #print (self.observation_history_length, self.observation_space_dim)
        
        self.observation_space = spaces.Box(-999,999,shape=(self.observation_history_length * self.observation_space_dim,)) 

        #print ('obs shape[0]',self.observation_space.shape[0])
        self.seed()

        #TOOD get the initial states
        self.ob_vals = collections.deque(maxlen=self.observation_history_length)
        self.full_ob_vals = collections.deque(maxlen=self.observation_history_length)

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
        
        if len(action) != self.actionLoadSelfZoneNum: #len(self.action_buses):
            print ('-----------!!!!error for action size:!!!!!!!!---------')
            
        reward = 0.0
        
        remain_load = self.ob_vals[-1][self.nob_volt:self.nob_volt+self.nob_load]
        ob_volt_tmp = self.full_ob_vals[-1][0:self.nobbus]
        
        loadshedact = gridpack.hadrec.Action()
        fault_end_time = self.fault_start_time + self.fault_duration_time

        for iact in range(len(action)):

            if ( abs(actionPyAry[iact]) > 0.01 ):  # add action if shed load > 0.01%                
                loadshedact.actiontype = 0
                loadshedact.bus_number = self.selfzone_action_buses[iact]
                loadshedact.componentID = "1"
                loadshedact.percentage = actionPyAry[iact]

                # apply the load shedding mask
                if self.apply_action_mask:
                    volt_tmp = ob_volt_tmp[self.ob_volt_busno_idx_dict[self.selfzone_action_buses[iact]]]

                    if volt_tmp > 0.8 and (fault_end_time<self.current_simu_time <= 0.33+fault_end_time):
                        #print ('bus %d ac motor load shed hit mask 1'%(self.selfzone_action_buses[iact]))
                        loadshedact.percentage = 0.0
                        #reward -= self.invalidActionPenalty

                    if volt_tmp > 0.9 and (0.33+fault_end_time<self.current_simu_time <= 0.5+fault_end_time):
                        #print('bus %d ac motor load shed hit mask 2' % (self.selfzone_action_buses[iact]))
                        loadshedact.percentage = 0.0
                        #reward -= self.invalidActionPenalty

                    if volt_tmp > 0.95 and (self.current_simu_time > 0.5+fault_end_time):
                        #print('bus %d ac motor load shed hit mask 3' % (self.selfzone_action_buses[iact]))
                        loadshedact.percentage = 0.0
                        #reward -= self.invalidActionPenalty

                    if self.current_simu_time < fault_end_time + 0.15:
                        loadshedact.percentage = 0.0

                self.hadapp.applyAction(loadshedact)

                # -----------check whether the action is applied before fault_start_time
                if self.current_simu_time < self.fault_start_time:
                    self.loadshed_before_fault = 1
                    if not self.allow_loadshed_before_fault:
                        reward -= self.preFaultActionPenalty
                        self.total_invalid_act_before -= self.preFaultActionPenalty
                    
                # ---------check whether the action is an invalid action, need check qiuhua for his implementation here
                #if ( remain_load[iact] + actionPyAry[iact]) > 0.01: # if it is a valid action
                if ( remain_load[iact] ) > 0.01: # if it is a valid action    
                    #reward += self.actionPenalty * actionPyAry[iact] * self.selfzone_action_buses_pvalue_pu[iact]
                    reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[iact]
                    self.total_loadshed += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[iact]
                    #print('---test code, step: %d, iload: %d, load shed per: %f')
                else:
                    if not self.allow_loadshed_before_fault:
                        reward -= self.invalidActionPenalty
                        self.total_invalid_act -= self.invalidActionPenalty
                    
        invalidact_rew =  reward

        #------uvls for self-zone loads without AI action but with low voltage
        total_loadshed_selfuvls_previous = self.total_loadshed_selfuvls
        delimer = 1.0
        if self.selfzone_uvls:
            for iact in range(len(action)):

                volt_tmp = ob_volt_tmp[self.ob_volt_busno_idx_dict[self.selfzone_action_buses[iact]]]

                loadshedact.actiontype = 0
                loadshedact.bus_number = self.selfzone_action_buses[iact]
                loadshedact.componentID = "1"
                loadshedact.percentage = 0.0

                if (abs(actionPyAry[iact]) < 0.01):  # add action if shed load < 0.01%
                    #if volt_tmp < 0.7 and (0.0 <= (self.current_simu_time - 0.33 - fault_end_time) <= self.simu_time_step):
                    if volt_tmp < 0.7 and (self.current_env_steps == int((fault_end_time+0.33/delimer)/self.env_time_step)):
                        '''
                        print('in condition, simu time: %f, bus %d voltage, %f' % (self.current_simu_time,
                                                                                   self.selfzone_action_buses[iact], volt_tmp))
                        '''

                        #print('bus %d, volt, %f, load shed hit mask 0 0.7 at env_time_steps: %d' %
                        #      (self.selfzone_action_buses[iact], volt_tmp, self.current_env_steps))
                        loadshedact.percentage = self.selfzone_action_ranges[iact, 0]
                        self.hadapp.applyAction(loadshedact)
                        reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        self.total_loadshed_selfuvls += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        continue

                    #if volt_tmp < 0.8 and (0.0 <= (self.current_simu_time - 0.5 - fault_end_time) <= self.simu_time_step):
                    if volt_tmp < 0.8 and (self.current_env_steps == int((fault_end_time + 0.5 / delimer) / self.env_time_step)):
                        #print('bus %d, volt, %f, load shed hit mask 1 0.8 at env_time_steps: %d' %
                        #     (self.selfzone_action_buses[iact], volt_tmp, self.current_env_steps))
                        loadshedact.percentage = self.selfzone_action_ranges[iact, 0]
                        self.hadapp.applyAction(loadshedact)
                        reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        self.total_loadshed_selfuvls += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        continue

                    #if volt_tmp < 0.9 and (0.0 <= (self.current_simu_time - 1.5 - fault_end_time) <= self.simu_time_step):
                    if volt_tmp < 0.9 and (self.current_env_steps == int((fault_end_time + 1.5 / delimer) / self.env_time_step)):
                        #print('bus %d, volt, %f, load shed hit mask 2 0.9 at env_time_steps: %d' %
                        #     (self.selfzone_action_buses[iact], volt_tmp, self.current_env_steps))
                        loadshedact.percentage = self.selfzone_action_ranges[iact, 0]
                        self.hadapp.applyAction(loadshedact)
                        reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        self.total_loadshed_selfuvls += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        continue

                    #if volt_tmp < 0.9 and (0.0 <= (self.current_simu_time - 2.0 - fault_end_time) <= self.simu_time_step):
                    if volt_tmp < 0.9 and (self.current_env_steps == int((fault_end_time + 2.0 / delimer) / self.env_time_step)):
                        #print('bus %d, volt, %f, load shed hit mask 3 0.9 at env_time_steps: %d' %
                        #      (self.selfzone_action_buses[iact], volt_tmp, self.current_env_steps))
                        loadshedact.percentage = self.selfzone_action_ranges[iact, 0]
                        self.hadapp.applyAction(loadshedact)
                        reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        self.total_loadshed_selfuvls += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        continue


                    #if volt_tmp < 0.9 and (0.0 <= (self.current_simu_time - 2.5 - fault_end_time) <= self.simu_time_step):
                    if volt_tmp < 0.9 and (self.current_env_steps == int((fault_end_time + 2.5 / delimer) / self.env_time_step)):
                        #print('bus %d, volt, %f, load shed hit mask 4 0.9 at env_time_steps: %d' %
                        #     (self.selfzone_action_buses[iact], volt_tmp, self.current_env_steps))
                        loadshedact.percentage = self.selfzone_action_ranges[iact, 0]
                        self.hadapp.applyAction(loadshedact)
                        reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        self.total_loadshed_selfuvls += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        continue


                    #if volt_tmp < 0.9 and (0.0 <= (self.current_simu_time - 3.0 - fault_end_time) <= self.simu_time_step):
                    if volt_tmp < 0.9 and (self.current_env_steps == int((fault_end_time + 3.0 / delimer) / self.env_time_step)):
                        #print('bus %d, volt, %f, load shed hit mask 5 0.9 at env_time_steps: %d' %
                        #      (self.selfzone_action_buses[iact], volt_tmp, self.current_env_steps))
                        loadshedact.percentage = self.selfzone_action_ranges[iact, 0]
                        self.hadapp.applyAction(loadshedact)
                        reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        self.total_loadshed_selfuvls += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        continue


                    #if volt_tmp < 0.9 and (0.0 <= (self.current_simu_time - 3.5 - fault_end_time) <= self.simu_time_step):
                    if volt_tmp < 0.9 and (self.current_env_steps == int((fault_end_time + 3.5 / delimer) / self.env_time_step)):
                        #print('bus %d, volt, %f, load shed hit mask 6 0.9 at env_time_steps: %d' %
                        #     (self.selfzone_action_buses[iact], volt_tmp, self.current_env_steps))
                        loadshedact.percentage = self.selfzone_action_ranges[iact, 0]
                        self.hadapp.applyAction(loadshedact)
                        reward += self.actionPenalty * loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        self.total_loadshed_selfuvls += loadshedact.percentage * self.selfzone_action_buses_pvalue_pu[
                            iact]
                        continue

        self.total_loadshed += (self.total_loadshed_selfuvls-total_loadshed_selfuvls_previous)
        #------------------apply UVLS for other zones ac motor loads
        other_zones_action = np.zeros((self.actionLoadOtherZoneNum))
        if self.apply_uvls_otherzones:
            for idxtmp in range(len(self.otherzones_action_buses)):
                ibus = self.otherzones_action_buses[idxtmp]
                volt_tmp = ob_volt_tmp[self.ob_volt_busno_idx_dict[ibus]]

                if volt_tmp < 0.7 and ( 0.0 <(self.current_simu_time-0.33-fault_end_time)<self.simu_time_step):
                    loadshedact.actiontype = 0
                    loadshedact.bus_number = ibus
                    loadshedact.componentID = "1"
                    loadshedact.percentage = self.otherzones_action_ranges[idxtmp, 0]
                    self.hadapp.applyAction(loadshedact)
                    continue

                if volt_tmp < 0.8 and ( 0.0 <(self.current_simu_time-0.5-fault_end_time)<self.simu_time_step):
                    loadshedact.actiontype = 0
                    loadshedact.bus_number = ibus
                    loadshedact.componentID = "1"
                    loadshedact.percentage = self.otherzones_action_ranges[idxtmp, 0]
                    self.hadapp.applyAction(loadshedact)
                    continue

                if volt_tmp < 0.9 and ( 0.0 <(self.current_simu_time-1.5-fault_end_time)<self.simu_time_step):
                    loadshedact.actiontype = 0
                    loadshedact.bus_number = ibus
                    loadshedact.componentID = "1"
                    loadshedact.percentage = self.otherzones_action_ranges[idxtmp, 0]
                    self.hadapp.applyAction(loadshedact)
                    continue

                if volt_tmp < 0.9 and ( 0.0 <(self.current_simu_time-2.0-fault_end_time)<self.simu_time_step):
                    loadshedact.actiontype = 0
                    loadshedact.bus_number = ibus
                    loadshedact.componentID = "1"
                    loadshedact.percentage = self.otherzones_action_ranges[idxtmp, 0]
                    self.hadapp.applyAction(loadshedact)
                    continue

                #'''
                if volt_tmp < 0.9 and ( 0.0 <(self.current_simu_time-2.5-fault_end_time)<self.simu_time_step):
                    loadshedact.actiontype = 0
                    loadshedact.bus_number = ibus
                    loadshedact.componentID = "1"
                    loadshedact.percentage = self.otherzones_action_ranges[idxtmp, 0]
                    self.hadapp.applyAction(loadshedact)
                    continue
                #'''

                if volt_tmp < 0.9 and ( 0.0 <(self.current_simu_time-3.0-fault_end_time)<self.simu_time_step):
                    loadshedact.actiontype = 0
                    loadshedact.bus_number = ibus
                    loadshedact.componentID = "1"
                    loadshedact.percentage = self.otherzones_action_ranges[idxtmp, 0]
                    self.hadapp.applyAction(loadshedact)
                    continue

                #'''
                if volt_tmp < 0.9 and ( 0.0 <(self.current_simu_time-3.5-fault_end_time)<self.simu_time_step):
                    loadshedact.actiontype = 0
                    loadshedact.bus_number = ibus
                    loadshedact.componentID = "1"
                    loadshedact.percentage = self.otherzones_action_ranges[idxtmp, 0]
                    self.hadapp.applyAction(loadshedact)
                    continue
                #'''

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
                ob_vals_tmp = list(range(self.nob_volt+self.nob_load ))
                ob_vals_tmp[0:self.nob_volt] = \
                    ob_vals_full[0:self.nob_volt]
                if self.nob_load > 0:
                    ob_vals_tmp[self.nob_volt:(self.nob_volt+self.nob_load)] = \
                        ob_vals_full[-self.nob_load:]

                self.ob_vals.append(ob_vals_tmp)
                self.full_ob_vals.append(ob_vals_full)
                #after_getob_time = time.time()    
                #total_dataconv_time += (after_getob_time - before_getob_time)
                
            done = self.hadapp.isDynSimuDone()
            if done:
                break

        self.current_env_steps = self.current_env_steps+1
            
        #--------compute the voltage deviation part for the reward    
        ob_volt_tmp = self.ob_vals[-1][0:self.nob_volt]
        fault_end_time = self.fault_start_time + self.fault_duration_time
        
        for ivoltob in range (self.nob_volt):
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
            self.total_volt_vio += volt_penalty
        
        volt_rew = reward - invalidact_rew
        
        # --------- if the voltage could not be back at minVoltRecoveryLevel after fault_end_time + maxVoltRecoveryTime, done
        unstable_rew = 0.0
        for itmp in range (len(self.ob_volt_for_reward_index)): # check with Qiuhua for his implementation
            idxtmp = self.ob_volt_for_reward_index[itmp]
            volt_cut_tmp= ob_volt_tmp[idxtmp]
            if (self.current_simu_time > (fault_end_time + self.maxVoltRecoveryTime)) and (volt_cut_tmp < self.minVoltRecoveryLevel):
                reward += self.unstableReward
                unstable_rew += self.unstableReward 
                done = True 
                #print ("------bus %d voltage still not recoverd to %f till %f second after fault clear, simulation done-------"%(self.obs_busIDs[ivoltob], self.minVoltRecoveryLevel, self.maxVoltRecoveryTime))
                break
        #---------------compute reward here-------------------    
        if done and self.steps_beyond_done is None:
            self.steps_beyond_done = 0

        reward_details = []
        reward_details.append(self.total_loadshed)
        reward_details.append(self.total_volt_vio)
        reward_details.append(self.total_loadshed_selfuvls)
        reward_details.append(self.total_invalid_act)
        reward_details.append(self.total_invalid_act_before)

        if self.return_reward_details:
            return np.array(self.ob_vals).ravel(), reward, reward_details, done, {}
        else:
            return np.array(self.ob_vals).ravel(), reward, done, {}

    def reset(self):

        # reset need to randomize the operation state and fault location, and fault time
        self.loadshed_before_fault = 0
        self.total_loadshed = 0.0
        self.total_loadshed_selfuvls = 0.0
        self.total_volt_vio = 0.0
        self.total_invalid_act_before = 0.0
        self.total_invalid_act = 0.0

        case_Idx = np.random.randint(0, self.total_case_num) # an integer
        
        total_fault_buses = len(self.faultbus_candidates)

       
        fault_bus_idx = np.random.randint(0, total_fault_buses)# an integer, in the range of [0, total_bus_num-1]
        
        #fault_bus_idx = 3 # an integer, in the range of [0, total_bus_num-1]
        #fault_start_time =random.uniform(0.99, 1.01) # a double number, in the range of [0.2, 1]
        self.fault_start_time = self.faultStartTimeCandidates[np.random.randint(0, len(self.faultStartTimeCandidates))]
        
        self.fault_duration_time = self.faultDurationCandidates[np.random.randint(0, len(self.faultDurationCandidates))] # a double number, in the range of [0.08, 0.4]
        self.fault_bus_no = self.faultbus_candidates[fault_bus_idx]

        #print ('---reset function, fault tuple is: ', case_Idx, fault_bus_idx, self.fault_start_time, self.fault_duration_time)
        
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

        if self.use_zone_info_as_ob:
            tmp_tuple = self.hadapp.getZoneLoads()
            self.zone_load_p = tmp_tuple[0].copy
            self.zone_load_q = tmp_tuple[1].copy
            self.zone_num_info = tmp_tuple[2].copy

            tmp_tuple = self.hadapp.getZoneGeneratorPower()
            self.zone_generation_p = tmp_tuple[0].copy
            self.zone_generation_q = tmp_tuple[1].copy

        #--------------update the action load value array--------------------
        itmp = 0
        for ibus in self.action_buses:
            totalloadtmp = self.hadapp.getBusTotalLoadPower(ibus)[0]
            self.action_buses_pvalue_pu[itmp] = totalloadtmp/100.0
            itmp = itmp + 1
            #print('   action load bus: %d, total load: ' % (ibus), totalloadtmp)
        self.selzone_action_buses_pvalue_pu = self.action_buses_pvalue_pu[0:self.actionLoadSelfZoneNum]
        self.otherzones_action_buses_pvalue_pu = self.action_buses_pvalue_pu[self.actionLoadSelfZoneNum:]

        #---------execute one simulation step to get observations
        self.current_simu_time = 0.0
        self.hadapp.executeDynSimuOneStep()
        self.current_simu_time = self.simu_time_step
        self.current_env_steps = 0
        
        ob_vals_full = self.hadapp.getObservations()
        ob_vals_tmp = list(range(self.nob_volt + self.nob_load))
        ob_vals_tmp[0:self.nob_volt] = ob_vals_full[0:self.nob_volt]
        if self.nob_load > 0:
            ob_vals_tmp[self.nob_volt:(self.nob_volt + self.nob_load)] = ob_vals_full[-self.nob_load:]
        self.ob_vals.clear()
        self.full_ob_vals.clear()
        for itmp in range(self.observation_history_length):
            self.ob_vals.append(ob_vals_tmp)
            self.full_ob_vals.append(ob_vals_full)

        #---------set voltage observations for reward computation index array, the reason is that,
        #---------there may be voltages at some buses very low at the steady state. Reward computation needs to exclude
        # those bus voltages
        self.ob_volt_for_reward_index = []
        if self.apply_volt_filter:
            for itmp in range(self.nob_volt):
                if ob_vals_tmp[itmp] > self.minVoltRecoveryLevel:
                    self.ob_volt_for_reward_index.append(itmp)
        else:
            self.ob_volt_for_reward_index = list(range(self.nob_volt))

        self.steps_beyond_done = None
        self.restart_simulation = True

        return np.array(self.ob_vals).ravel()

    #---------------initialize the system with a specific state and fault
    def validate(self, case_Idx, fault_bus_idx, fault_start_time, fault_duration_time):

        # ------reinitialize the hadrec module
        self.loadshed_before_fault = 0
        self.total_loadshed = 0.0
        self.total_loadshed_selfuvls = 0.0
        self.total_volt_vio = 0.0
        self.total_invalid_act_before = 0.0
        self.total_invalid_act = 0.0

        self.fault_start_time = fault_start_time
        self.fault_duration_time = fault_duration_time
        if self.use_fault_bus_num:
            self.fault_bus_no = fault_bus_idx
        else:
            self.fault_bus_no = self.faultbus_candidates[fault_bus_idx]
        
        busfault = gridpack.dynamic_simulation.Event()
        busfault.start = fault_start_time
        busfault.end = fault_start_time + fault_duration_time
        busfault.step = self.simu_time_step
        busfault.isBus = True
        busfault.bus_idx = self.fault_bus_no #self.faultbus_candidates[fault_bus_idx]
    
        busfaultlist = gridpack.dynamic_simulation.EventVector([busfault])

        #print('------debug, validate, before solvePowerFlowBeforeDynSimu, case_Idx = %d'%(case_Idx))
        self.hadapp.solvePowerFlowBeforeDynSimu(self.simu_input_file, case_Idx)

        #print ('------debug, validate, after solvePowerFlowBeforeDynSimu')
        self.hadapp.transferPFtoDS()
        #print('------debug, validate, after transferPFtoDS')
        self.hadapp.initializeDynSimu(busfaultlist)

        if self.use_zone_info_as_ob:
            tmp_tuple = self.hadapp.getZoneLoads()
            self.zone_load_p = tmp_tuple[0].copy
            self.zone_load_q = tmp_tuple[1].copy
            self.zone_num_info = tmp_tuple[2].copy

            tmp_tuple = self.hadapp.getZoneGeneratorPower()
            self.zone_generation_p = tmp_tuple[0].copy
            self.zone_generation_q = tmp_tuple[1].copy

        #--------------update the action load value array--------------------
        itmp = 0
        for ibus in self.action_buses:
            totalloadtmp = self.hadapp.getBusTotalLoadPower(ibus)[0]
            self.action_buses_pvalue_pu[itmp] = totalloadtmp/100.0
            itmp = itmp + 1
            #print('   action load bus: %d, total load: ' % (ibus), totalloadtmp)
        self.selzone_action_buses_pvalue_pu = self.action_buses_pvalue_pu[0:self.actionLoadSelfZoneNum]
        self.otherzones_action_buses_pvalue_pu = self.action_buses_pvalue_pu[self.actionLoadSelfZoneNum:]

        #--------execute one simulation step to get observations
        self.current_simu_time = 0.0
        self.hadapp.executeDynSimuOneStep()
        self.current_simu_time = self.simu_time_step
        self.current_env_steps = 0
        
        ob_vals_full = self.hadapp.getObservations()
        ob_vals_tmp = list(range(self.nob_volt + self.nob_load))
        ob_vals_tmp[0:self.nob_volt] = ob_vals_full[0:self.nob_volt]
        if self.nob_load > 0:
            ob_vals_tmp[self.nob_volt:(self.nob_volt + self.nob_load)] = ob_vals_full[-self.nob_load:]
        self.ob_vals.clear()
        self.full_ob_vals.clear()
        for itmp in range(self.observation_history_length):
            self.ob_vals.append(ob_vals_tmp)
            self.full_ob_vals.append(ob_vals_full)

        #---------set voltage observations for reward computation index array, the reason is that,
        #---------there may be voltages at some buses very low at the steady state. Reward computation needs to exclude
        # those bus voltages
        self.ob_volt_for_reward_index = []
        if self.apply_volt_filter:
            for itmp in range(self.nob_volt):
                if ob_vals_tmp[itmp] > self.minVoltRecoveryLevel:
                    self.ob_volt_for_reward_index.append(itmp)
        else:
            self.ob_volt_for_reward_index = list(range(self.nob_volt))

        self.steps_beyond_done = None
        self.restart_simulation = True

        return np.array(self.ob_vals).ravel()
    
    def close_env(self):
        self.hadapp = None
        #self.run_gridpack_necessary_env = None
        print("--------- GridPACK HADREC APP MODULE deallocated ----------")

    def get_base_cases(self):
            
        return self.base_pf_cases

    def trip_one_branch(self, frombus, tobus, ckt):

        linetrip = gridpack.hadrec.Action()
        linetrip.actiontype = 1
        linetrip.brch_from_bus_number = frombus
        linetrip.brch_to_bus_number = tobus
        linetrip.branch_ckt = ckt

        self.hadapp.applyAction(linetrip)

        
        
        
    
    
    
    
    
    
    
    
    