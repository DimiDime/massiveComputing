#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
#
import numpy as np


# In[ ]:




from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes



# In[ ]:


#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

#this function creates the instance of Value data type and initialize it to 0
def dot_init(g_A):
    global A 
    A = g_A #We create a variable of type "double"           
    
    
def shared_dot_1(V):
    #This code is wrong!!!
    for f in V:
        A.value += f[0]*f[1]
    
def shared_dot_2(V):
    #This code is wrong!!!
    with A.get_lock():
        for f in V:
            A.value += f[0]*f[1]
    
def shared_dot_3(V):
    #This code is wrong!!!
    a=0
    for f in V:
        a += f[0]*f[1]
    with A.get_lock():
        A.value += a
    


#This function initialize the global shared memory data

def pool_init(shared_array_,srcimg, imgfilter):
    #shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
    #srcimg: is the original image
    #imgfilter is the filter which will be applied to the image and stor the results in the shared memory array
    
    #We defines the local process memory reference for shared memory space
    global shared_space
    #Here we define the numpy matrix handler
    global shared_matrix
    
    #Here, we will define the readonly memory data as global (the scope of this global variables is the local module)
    global image
    global my_filter
    
    #here, we initialize the global read only memory data
    image=srcimg
    my_filter=imgfilter
    size = image.shape
    
    #Assign the shared memory  to the local reference
    shared_space = shared_array_
    #Defines the numpy matrix reference to handle data, which will uses the shared memory buffer
    shared_matrix = tonumpyarray(shared_space).reshape(size)


# In[ ]:


#this function just copy the original image to the global r/w shared  memory 
def parallel_shared_imagecopy(row):
    global image
    global my_filter
    global shared_space    
    # with this instruction we lock the shared memory space, avoidin other parallel processes tries to write on it
    with shared_space.get_lock():
        #while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[row,:,:]=image[row,:,:]
    return


# In[ ]:


def edge_filter(row):
    global image
    global my_filter
    global shared_space

    (rows,cols,depth) = image.shape
    #fetch the r row from the original image
    srow=image[row,:,:]
    if ( row>0 ):
        prow=image[row-1,:,:]
    else:
        prow=image[row,:,:]

    if ( row == (rows-1)):
        nrow=image[row,:,:]
    else:
        nrow=image[row+1,:,:]
    
    #defines the result vector, and set the initial value to 0
    frow=np.zeros((cols,depth))
    
     for c in range (cols): # Create a for loop to iterate through all the columns  
        if ( c>0 ): # Check if the current index is bigger than zero
            pcol = c-1 # Set pcol to the previous index
        else:
            pcol= c # otherwise set the pcol to the current index
        if ( c == (cols-1)): # if we have reached the last index of the image
            ncol= c # then next column becommes that index
        else:
            ncol= c+1 # otherwise next column becommes the following index
        
        for d in range(depth): # for loop to go over the 3 levels of depth of the image
            # Here we apply the filter over each and every position of the image
            # from index [0,0] to [2,2]
            frow[c,d] = srow[c,d] * my_filter[1,1] + prow[pcol, d] * my_filter[0,0] + prow[c, d] * my_filter[0,1]
            frow[c,d] += prow[ncol, d] * my_filter[0,2] + srow[pcol, d] * my_filter[1,0] + srow[ncol, d] * my_filter[1,2]
            frow[c,d] += nrow[pcol, d] * my_filter[2,0] + nrow[c, d] * my_filter[2,1] + nrow[ncol,d] * my_filter[2,2]
    
    with shared_space.get_lock():
        # If the Lock is set in the beginning of the edge_filter function then it will block the function itself 
        # and will not be able to execute properly.
        # This is why the instruction had to be moved to the end of the function and ensure that no ones, except this execution thread, 
        # can write in the shared memory.
            shared_matrix[row,:,:]=frow
    return    


# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

