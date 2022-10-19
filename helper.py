import numpy as np
import csv
import subprocess
import re
import os
from scipy.stats import chi, norm
import math
              
def create_intervals(dims, no_bins, latent_range):    
    #Below logic divides probability [0,1] into equal density intervals
    #result will be in parts which is a list of tuples
    #each tuple contains lower and upper bounds of random variable for that division
    
    #probability of each division
    partition_density = 1.0/(no_bins)
    
    #For each of the divisions, convert probability density bounds into random variable bounds
    density = 0
    parts = list()
    for i in range(no_bins+1):
        #below if condition is to take care of the cases where density = 1 + epsilon
        #i.e., density is greater than 1 by negligible amount due to 
        #density variable rhs in the below logic
        if density > 1:
            density = 1
            
        #below code converts probability bounds to random variable 
        #bounds using quantile function
        rv = norm.ppf(density)
        
        #below logic limits the random variable partitions to 
        #lie in [-partsmax, partsmax] range
        if rv == -np.inf:
            rv = latent_range[0]
        elif rv > latent_range[1]:
            rv = latent_range[1]
            
        parts.append(rv)

        density = density + partition_density
    
    intervals = np.array(parts)
    
    intervals = intervals
    #print("intervals shape {}".format(intervals.shape))
    return intervals

#acts config file required by ccmcl tool
def measure_coverage(feature_array, acts, ways=3, timeout=10, suffix="temp"):     
    #create csv files for calculating IDC
    csv_header = ""
    for i in range(feature_array.shape[1]):
        csv_header += "p"+str(i+1)
        if i < feature_array.shape[1]-1:
            csv_header += ","
    csv_file = f"CA_{suffix}.csv"
    np.savetxt(csv_file, feature_array, delimiter=",", header=csv_header, fmt='%d')
    
    #run ccmcl tool to compute coverage
    p1 = subprocess.Popen(['timeout', str(timeout), 'java', '-jar', 'ccmcl.jar', '-A', str(acts), '-I', str(csv_file), '-T', str(ways)], stdout=subprocess.PIPE)
    p1.wait()
    idc_acc_array_content, err = p1.communicate()                      

    #parse idc values using regex
    rex = re.compile('Total\s\d-way coverage:\s(\d?.\d+)')
    coverage = float(rex.search(str(idc_acc_array_content)).group(1))
    return coverage
    
#k = factors, v = levels
#dataset str name
#returns name of the generated acts file
def create_acts(k, v):
    #Create acts parameter file
    acts = f"Config/{k}params_{v}bins.txt"
    #create parameter file for ccmcl tool
    subprocess.call(['./create_acts.sh', "IDC", str(k), '1', str(v), str(acts)])
    assert os.path.exists(acts), "acts not generated"
    return acts
    
def generate_array(latent, density, no_bins=10):
    #calculate annulus boundary based on density
    radin, radout = chi.interval(density, latent.shape[1])
    
    #print("***Calculations using Chi distribution for density {} ".format(density))
    #print("inner radius = {} and outer radius = {}".format(radin, radout))
    
    latent_range = (-radout, radout)
    #print("latent bounds ", latent_range)
    
    intervals = create_intervals(latent.shape[1], no_bins, latent_range)
    #print("intervals on each dimension {}".format(intervals))
    
    #below commented out code is redundant
    #print("original latent shape ", latent.shape)
    #latent = latent[np.all(latent >= latent_range[0], axis = 1)]
    #latent = latent[np.all(latent < latent_range[1], axis = 1)]
    #print(f"Filtered latent shape {latent.shape} in latent range {latent_range}")
    
    
    x_squares = np.square(latent)
    radius_vector = np.sqrt(np.sum(x_squares, axis=1)).reshape(-1,1)
    indices1 = np.argwhere(radius_vector < radin)[:, 0]
    latent = latent[(radius_vector >= radin).reshape(-1)]
    
    x_squares = np.square(latent)
    radius_vector = np.sqrt(np.sum(x_squares, axis=1)).reshape(-1,1)
    indices2 = np.argwhere(radius_vector > radout)[:, 0]
    latent = latent[(radius_vector <= radout).reshape(-1)]
    #print(f"Filtered latent inside the shell [{round(radin, 4)}, {round(radout, 4)}] is {latent.shape}")
    
    #Finding array of test inputs mean values in latent space wrt partitions created above
    cov_array = np.digitize(latent[:, 0], intervals).reshape(-1, 1)
    #print("array shape {}".format(cov_array.shape))
    for i in range(latent.shape[1]-1):
        cov_vector = np.digitize(latent[:, i+1], intervals).reshape(-1, 1)
        cov_array = np.concatenate((cov_array, cov_vector), axis=1)
    #print("array shape for testing dataset {}".format(cov_array.shape)) 
    return cov_array, latent.shape[0], (indices1,indices2)