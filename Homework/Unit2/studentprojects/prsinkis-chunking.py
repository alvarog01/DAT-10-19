#!/usr/bin/env python
# Write two functions
# Function 1 - probe_df
# Function 2 - write_df

import numpy as np
import pandas as pd


def probe_df(file_path, chunksize=1000):
    col_names = []
    col_stats = []
    col_dict = {}
    num_rows = 0
    
    for chunk in pd.read_csv(file_path,chunksize=chunksize):
    
        # Get initial values together        
        if col_names == []:
            col_names = chunk.columns.to_list()
            for c in col_names:
                col_dict[c] = {'dtype':str(chunk[c].dtype)}
                col_stats.append([c,0,0])
                
        # Do stuff to loop across chunks
        num_rows = num_rows + chunk.index.size
        for (i,c) in enumerate(col_names):
            
            # Add up nulls
            col_stats[i][1] = col_stats[i][1] + chunk[c].isnull().sum()

            # Add up numbers to calc avarege, 
            # use np.float64 to address overflow errors as best we can   
            # need to make sure this is applied before the sum, to ensure the
            # individual elements are converted to an np.float64
            if chunk[c].dtype!='O':
                col_stats[i][2] = np.float64(col_stats[i][2]) + chunk[c].astype(np.float64).sum()
            
    # Once run through all chunks
    for (i,c) in enumerate(col_names):
        col_dict[c]['null_values'] = col_stats[i][1]
        if col_dict[c]['dtype'] != 'O':
            col_dict[c]['average'] = ( np.float64(col_stats[i][2]) / 
                                      (np.float64(num_rows) 
                                       - np.float64(col_stats[i][1]))
                                      )
    
    return col_dict


# This is a test of the probe_df function:
    
# test_output = probe_df('./data/titanic.csv')
# test_output = probe_df('https://dat-data.s3.amazonaws.com/taxi.csv')
# test_output = probe_df('./data/taxi.csv',5000) # Use taxi locally to save time

# for i in test_output:
#     print(f"{i} : {test_output[i]}")



def write_df(file_path_read, file_path_write, chunksize=1000, missing_vals = {}):
    first_pass = True    

    # Read the file down in chunks
    for chunk in pd.read_csv(file_path_read,chunksize=chunksize):

        # Make modfications to each chunk, looping through the dictionary
        for i in missing_vals:
            chunk[i] = chunk[i].replace(np.nan,missing_vals[i])
    
        # write out the file by appending 
        # First pass - include headers, and (w)rite new file
        # Furhter passes - exclude headers, and (a)ppend to file 
        # Exclude index to reduce file size
        if first_pass == True:
            first_pass = False
            chunk.to_csv(file_path_write, index=False,header=True,mode='w')
        else:
            chunk.to_csv(file_path_write, index=False,header=False,mode='a')

# print("starting test write...")
# mv = {'ORIGIN_CALL':24490.363017792035,
#       'ORIGIN_STAND':30.272381254657013,
#     }
# write_df('./data/taxi.csv','./data/taxi_out.csv', chunksize = 7000, missing_vals = mv)
# print("test write finished.")
