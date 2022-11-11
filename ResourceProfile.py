import collections
from sys import flags
import threading
import psutil
import os
import json
from pynvml import *
import time
import pickle

global Resource_flag
Resource_flag = False
data = collections.defaultdict(list)

pid = os.getpid()
print("RP" + str(pid))
p = psutil.Process(pid)


def get_cpu_mem(rd):
    cpu_percent = p.cpu_percent() / psutil.cpu_count()
    mem_percent = p.memory_percent()
    mem_rss = p.memory_info()
    # m = threading.Timer(1.0, get_cpu_mem)
    # m.start()

    rd["cpu_percent"].append(cpu_percent)
    rd["mem_percent"].append(mem_percent)
    rd["mem_rss"].append(mem_rss)
    return cpu_percent, mem_percent, mem_rss[0]


def get_GPU(rd):
 
    nvmlInit()

    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    occupied_percent = info.used / info.total

    rd["gpu_name"] = info
    rd["gpu_occupied"].append(info.used)
    rd["gpu_occupied_percent"].append(occupied_percent)
    # t=threading.Timer(1.0,get_GPU)
    # t.start()
    nvmlShutdown()
    # print(data)
    return info.used, occupied_percent


def run_new(useGpu , rd):
    cpu = get_cpu_mem(rd)
    if useGpu:
        gpu = get_GPU(rd)
    # print('data_len', len(rd["mem_rss"]))


def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def ResourceProfile_new( useGpu , resource_data, filename):
    while True:
        global  Resource_flag
        if Resource_flag:
            save(filename, resource_data)
            print("save resource %s"%filename)
            break

        run_new(useGpu , resource_data)
        time.sleep(1)


def RP(useGPU, filename):
    global  Resource_flag
    Resource_flag = False
    resource_data = collections.defaultdict(list)
    p1 = threading.Thread(target=ResourceProfile_new, args=(useGPU ,resource_data, filename,))
    # p1.setDaemon(True)
    p1.start()

def stop_RP():
    global  Resource_flag
    Resource_flag = True




if __name__ == "__main__":
    RP( useGpu=True , filename = 'resource_log.pkl')

    t1 = time.time()
    i = 0
    while True:
        time.sleep(1)
        i += 1
        if i >= 5:
            stop_RP()
            break

    # test
    file = open('resource_log.pkl', 'rb')
    data = pickle.load(file)
    print(data)
