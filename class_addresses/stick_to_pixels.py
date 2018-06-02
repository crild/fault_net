# -------- points to pixels ---------

# import prerequisite modules
import numpy as np


# Function that creates .pts files for training data from fault stick data
def sticks_to_pixels(file_list,sample_rates,save_name):
    # file_list: list of fault stick file names
    # sample_rate: inline, xline, time sample rates
    # save_name: name of save file

    # Define individual step lengths
    inline_step = sample_rates[0]
    xline_step = sample_rates[1]
    time_step = sample_rates[2]

    # make empty array to store output
    arr = np.empty([0,3])

    for j in range(len(file_list)):
        # Upload file number i, store its length and shuffle it
        a = np.loadtxt(file_list[j], skiprows=0, usecols = [1,2,5], dtype = np.int32)

        a[:,0] = np.floor(a[:,0]/inline_step)*inline_step
        a[:,1] = np.floor(a[:,1]/xline_step)*xline_step
        a[:,2] = np.floor(a[:,2]/time_step)*time_step


        for i in range(1,len(a)):
            il_start = a[i-1,0]
            il_stop = a[i,0]

            xl_start = a[i-1,1]
            xl_stop = a[i,1]

            t_start = a[i-1,2]
            t_stop = a[i,2]

            t_range = np.arange(t_start,t_stop,time_step)

            xl = ((xl_stop - xl_start)/(t_stop-t_start))*(t_range-t_start)+xl_start
            il = ((il_stop - il_start)/(t_stop-t_start))*(t_range-t_start)+il_start

            line_segment = np.rint(np.asarray([il,xl,t_range]))
            arr = np.append(arr,line_segment.T,axis=0)

    # Save csv file
    np.savetxt(fname = save_name+'.pts',X = arr,fmt = '%i')


# Define the function parameters
samp_rt_Hoop = np.array([1,1,4])


filename = ['Fault1.pts','Fault2.pts','Fault3.pts','Fault4.pts','Fault5.pts','Fault6.pts',
            'Fault7.pts','Fault8.pts','Fault9.pts','Fault10.pts','Fault11.pts','Fault12.pts',
            'Fault13.pts','Fault14.pts']

name = 'fault_ilxl'
sticks_to_pixels(filename,samp_rt_Hoop,name)


filename = ['NoFault1.pts','NoFault2.pts','NoFault3.pts','NoFault4.pts','NoFault5.pts',
            'NoFault6.pts','NoFault7.pts']

name = 'nofault_ilxl'
sticks_to_pixels(filename,samp_rt_Hoop,name)
