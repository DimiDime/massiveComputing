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
    
def parallel_filtering_image(r):
    # r: is the image row to filter
    # image is the global memory array
    # my_filter is the filter shape to apply to the image
    
    global image
    global my_filter
    
    #from the global variable, gets the image shape
    (rows,cols,depth) = image.shape
    #Then, we have to go through the three layers and through all the columns
    #in order to compute the average filter in every pixel
    #fetch the r row from the original image
    frow=image[r,:,:]
    for i in range(depth):
      #with this loop we are fetching the row taking into account the margins, that is 
      #for every layer, so we have the next row (nrow) and the previous row (prow)        #for every different case
            #srow is the row in which the pixel is
          srow=image[r,:,i]
          if (r>0):
              prow = image[r-1,:,i]
          else:
              prow = image[r,:,i]
          if (r == (rows-1)):
              nrow = image[r,:,i]
          else:
              nrow = image[r+1,:,i]
            #now in each row and in each layer we move among the columns 
            #this way we can assign variables to all the pixels surrounding the correspondent pixel
            #allowing us to compute the mean, which will be the pixel of the filtered image
          for c in range(cols):
                #the eight different pixels surrounding p0
              p0 = srow[c] 
              p3 = prow[c] 
              p6 = nrow[c]
                #we have to take again the lateral margins
              if (c>0):
                  p1 = srow[c-1]
                  p4 = prow[c-1]
                  p7 = nrow[c-1]
              else:
                  p1 = srow[c]
                  p4 = prow[c]
                  p7 = nrow[c]
              if (c==(cols-1)):
                  p2 = srow[c]
                  p5 = prow[c]
                  p8 = nrow[c]
              else:
                  p2 = srow[c+1]
                  p5 = prow[c+1]
                  p8 = nrow[c+1]
              #now, we compute the mean amongst all the pixels
              p = 1/9*int(p0) + 1/9*int(p1) + 1/9*int(p2) + 1/9*int(p3) + 1/9*int(p4) + 1/9*int(p5) + 1/9*int(p6) + 1/9*int(p7) + 1/9*int(p8)
              #Finally we allocate the new value of the pixel in it's correspondent position
              #in the filtered row, for each row introduced in the function and for each layer and 
              #column indicated in the loop
              
              frow[c,i] = p
                
    #return the filtered row
    return frow

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


def edge_filter(r):
    global image
    global my_filter
    global shared_space
    (rows,cols,depth) = image.shape

    frow=image[r,:,:]
    for i in range(depth):
      #with this loop we are fetching the row taking into account the margins, that is 
      #for every layer, so we have the next row (nrow) and the previous row (prow)        #for every different case
            #srow is the row in which the pixel is
          srow=image[r,:,i]
          if (r>0):
              prow = image[r-1,:,i]
          else:
              prow = image[r,:,i]
          if (r == (rows-1)):
              nrow = image[r,:,i]
          else:
              nrow = image[r+1,:,i]
            #now in each row and in each layer we move among the columns 
            #this way we can assign variables to all the pixels surrounding the correspondent pixel
            #allowing us to compute the mean, which will be the pixel of the filtered image
          for c in range(cols):
                #the eight different pixels surrounding p0
              p0 = srow[c] 
              p3 = prow[c] 
              p6 = nrow[c]
                #we have to take again the lateral margins
              if (c>0):
                  p1 = srow[c-1]
                  p4 = prow[c-1]
                  p7 = nrow[c-1]
              else:
                  p1 = srow[c]
                  p4 = prow[c]
                  p7 = nrow[c]
              if (c==(cols-1)):
                  p2 = srow[c]
                  p5 = prow[c]
                  p8 = nrow[c]
              else:
                  p2 = srow[c+1]
                  p5 = prow[c+1]
                  p8 = nrow[c+1]
                #now, we compute the mean amongst all the pixels
              p = 1/9*int(p0) + 1/9*int(p1) + 1/9*int(p2) + 1/9*int(p3) + 1/9*int(p4) + 1/9*int(p5) + 1/9*int(p6) + 1/9*int(p7) + 1/9*int(p8)
                #Finally we allocate the new value of the pixel in it's correspondent position
                #in the filtered row, for each row introduced in the function and for each layer and 
                #column indicated in the loop
              frow[c,i] = p

    with shared_space.get_lock():
        #while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[r,:,:]=frow

    return frow


# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

