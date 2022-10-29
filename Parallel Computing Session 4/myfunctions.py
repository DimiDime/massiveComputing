#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
# Due a bug in the multiprocessing module implemented for Windows Operating System, the functions which will be executed in the parallel threads MUST be implemented in a separated file, and import them in the main programs.
# 
# In order to be loaded in you own program, you have to write your own functions here, and export to a python ".py" file, to be imported in  the main script.
# 
# To export to a python file, select in the *File*  menu, the option *Download as* and save as *Python .py* file.

# ## Functions needed for FirstParallel notebook
import time
import numpy as np
# In[ ]:


def f(x):
    return x*x


def square_vector(idx):
    start_time = time.time()
    result = []
    
    #16->625000
    #32->312500
    #64->156250
    #128->78125
    NUM_ELEMENTS=78125 #DO NOT FORGET CHANGE THIS VALUE WHEN CHANGE THE NUMBER OF PARALLEL PROCESES
    data = np.random.rand(NUM_ELEMENTS)
    for d in data:
        result.append(f(d))
    total_time=time.time()-start_time
    print("Hi, i am the task index {0}, time {1}".format(idx,total_time))
    return total_time

def dot(a,b):
    r=a*b
    return r


# ## Functions need for Benchmark notebook

# In[2]:

"""
This function, who just execute several times a mathematical operation, to consume CPU time, will be used in the benchmark notebook to measure the speedup when launch tasks in paralallel
"""

def work(task):
    """
    Some amount of work that will take time
    
    Parameters
    ----------
    task : tuple
        Contains number, loop, and number processors
    """
    number, loop = task
    b = 2. * number - 1.
    for i in range(loop):
        a, b = b * i, number * i + b
    return a, b



#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

    
def dott(tup): # Dot product function
    time_s = time.time() #get the time
    res = np.dot(tup[0],tup[1]) #run the dot product of both values in the tuple 
    
    return time.time() - time_s # return the time
