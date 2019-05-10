import os

a = ["1,2"]
b = ["2,3"]

for i in range(7):
    print(">>>>>>>>>>>index %d <<<<<<<<< <<<" % (i+1))
    os.system("python single_frame_estimation_hmr_LR_nonrigid.py /home/lgh/real_system_data7/data1/people3/LR%d/ %s %s" % ((i+1),a, b))
