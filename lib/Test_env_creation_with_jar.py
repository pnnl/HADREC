import os.path, time, sys
from py4j.java_gateway import (JavaGateway, GatewayParameters)
from subprocess import call, Popen, PIPE
java_port = 25334
from pathlib import Path

folder_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

print(folder_dir)
# This is to fix the issue of "ModuleNotFoundError" below
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

case_files_array = []
case_files_array.append(folder_dir + '/testData/IEEE39/IEEE39bus_multiloads_xfmr4_smallX_v30.raw')
case_files_array.append(folder_dir + '/testData/IEEE39/IEEE39bus_3AC.dyr')

dyn_config_file = folder_dir + '/testData/IEEE39/json/IEEE39_dyn_config.json'
rl_config_file = folder_dir + '/testData/IEEE39/json/IEEE39_RL_loadShedding_3motor_continuous.json'

jar_file = "RLGCJavaServer0.93.jar"
from PowerDynSimEnvDef_v7 import PowerDynSimEnv
env = PowerDynSimEnv(case_files_array,dyn_config_file,rl_config_file,jar_file,java_port)

#-------------------just above here you could already use the env in ARS training

for i in range(15):
    results = env.step([-.5,-0.3,-0.1]) # no action is applied
    print('step reward =', results[1])

print('test completed')
env.close_connection()
print('connection with Ipss Server is closed')



