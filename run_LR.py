import os
import util

a = "[1,2]"
b = "[1,2]"
hmr_path = "/home/lgh/real_system_data7/data1/people3"   #################

for i in range(7):                   #####################
    print(">>>>>>>>>>>index %d <<<<<<<<< <<<" % (i+1))
    os.system("python LR_model.py %s %s %s/LR%d/" % (a, b, hmr_path, (i+1)))
    os.system("cp ./run_all.json %s/LR%d/%s/run_all.json" % (hmr_path, (i+1), util.params["path"]["output_path"]))
