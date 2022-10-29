from re import I
import numpy as np
import multiprocessing as mp
import math




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


#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

def image_filter( image,#: numpy array,
filter_mask, #: numpy array 2D,
 numprocessors, #: int
 filtered_image ): #: multiprocessing.Array) 
    rows=range(image.shape[0]) # Get the range of the image rows
    # Using the Pool object we can define how we want to implement the filter on the image
    with mp.Pool(processes=numprocessors,initializer=pool_init,initargs=[filtered_image,image,filter_mask]) as p:
        e = p.map(f1,rows) # USing the map function we can use the filter to iterate over the whole length of the image array
    return e # And then return the result

# Function for the application of the filter
def f1(row):
    global image # Original image
    global my_filter # Filter definition
    global shared_space # Shared memory variable
    
    with shared_space.get_lock(): # Create a lock for the shared memory
        (rows,cols,depth) = image.shape # Create a tuple with each index of the number of corresponding elements
        try: # Try block for the filtered rows and columns of the image
            (frows,fcols) = my_filter.shape
        except Exception as e: # Throw and exception otherwise
            return e
        
        
        rowss=[] # Fetch the r row from the original image
        halo = math.floor(frows/2) # Devide filtered rows by 2 and round up to the nearest integer to get the value of the halo/s
        
        # For loop to go over the range in the rows
        for i in range(halo,0,-1):
            if (row-i>=0 ): # check if its the first row
                rowss.append(image[row-i,:,:]) # then we append the first image row to it
            else:
                rowss.append(image[row,:,:]) # otherwise append it to the row
        rowss.append(image[row,:,:])
        # Go over the seond row
        for ii in range(halo,0,-1): 
            if ( row+ii >= (rows) ):
                rowss.append(image[row,:,:])
            else:
                rowss.append(image[row+ii,:,:])
        
        #defines the result vector, and set the initial value to 0
        frow=np.zeros((cols,depth))
        halo2=math.floor(fcols/2)
        
        
        ko = 0 # Iterate through the columns
        for c in range(cols-1):
            colss=[] # columns array
            # in the middle of the matrix
            
            for j in range(halo2,0,-1): # Go over the halo of the columns
                if ( c-j>=0 ):
                    colss.append( c-j ) # prev row
                else:
                    # On the left border of the matrix append c
                    colss.append(c)
            # Append the column to the end of it
            colss.append(c)
            for jj in range(halo2,0,-1):
                if ( c == (cols-jj)):
                    # On the right border of the matrix
                    colss.append(c) # Next one
                else:
                    # And the middle one
                    colss.append(c+jj)
                    
            # Next we iterrate over the depth  
            for d in range(depth):
                # Define variables fr and fc
                fr=0 
                fc=0
                for co in colss: # iterate through columns
                    for rr in rowss: # iterate through rows
                        try:
                            # We apply the formula from slides for each depth
                            #(m,n) = (math.floor(frows/2),math.floor(fcols/2))
                            frow[c,d] += rr[co,d] * my_filter[fr,fc]
                            fr+=1
                        except Exception as e:
                            return e
                    fc+=1
                    fr=0
            
            # While we are in this code block no one, except this execution thread, can write in the shared memory
            shared_matrix[row,:,:]=frow

    return 0