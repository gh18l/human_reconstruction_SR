import os
import json
str_hr = []
str_lr = []
str_hr.append("[11,28]")
str_lr.append("[0,16,32,48,64,80]")
str_hr.append("[10,27]")
str_lr.append("[0,15,30,45,60,75,90]")
str_hr.append("[11,28]")
str_lr.append("[0,16,32,48,64,80]")
str_hr.append("[1,18]")
str_lr.append("[0,17,34,51,68,85]")
str_hr.append("[10,27]")
str_lr.append("[0,17,33,49,65,81]")
str_hr.append("[16,33]")
str_lr.append("[0,17,34,51,68,85]")
str_hr.append("[5,22]")
str_lr.append("[0,17,34,51,68,85]")
str_hr.append("[11,28]")
str_lr.append("[0,17,34,51,68,85]")
hmr_path = "/home/lgh/real_system_data7/data1/people1"   #################

with open("./run_all.json", "r") as f:
    params = json.loads(f.read())

for i in range(1):                   #####################
    print(">>>>>>>>>>>index %d <<<<<<<<< <<<" % (i+1))
    os.system("python LR_refine_model.py %s %s %s/LR%d/ " % (str_hr[i], str_lr[i], hmr_path, (i+1)))
    os.system("cp ./run_all.json %s/LR%d/%s/run_all.json" % (hmr_path, (i + 1), params["path"]["refine_output_path"]))
