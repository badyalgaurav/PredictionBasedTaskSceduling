import psutil


import time

def processcheck(seekitem):
    print(psutil.cpu_times())
    # plist = psutil.get_process_list()
    # str1=" ".join(str(x) for x in plist)
    # if seekitem in str1:
    #     print ("Requested process is running")

processcheck("System")