#!/usr/bin/env python3
results = []
def task_a():
    res = {}
    res['task_a'] = 1 + 1
    results.append(res)
def task_b():
    res = {}
    res['task_b'] = 2 + 2
    results.append(res)
    
def task_c(name):
    res = {}
    res['task_c'] = "Name: %s"%(name)
    results.append(res)

def run_parallel(*functions):
    '''
    Run functions in parallel
    '''
    from multiprocessing import Process
    processes = []
    for function in functions:
        proc = Process(target=function)
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

if __name__ == '__main__':    
	run_parallel(task_a(),task_b(),task_c(name="konstantinos"))
	print(results)