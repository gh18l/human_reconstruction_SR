import os

str_hr = []
str_lr = []
str_hr.append("[11,27]")
str_lr.append("[0,15,30,46,61,76]")
str_hr.append("[11,27]")
str_lr.append("[0,14,28,42,59,73,87]")
str_hr.append("[14,30]")
str_lr.append("[0,14,31,47,63,78]")
str_hr.append("[10,26]")
str_lr.append("[0,16,33,49,66,82]")
str_hr.append("[2,18]")
str_lr.append("[0,16,32,48,64,80]")
str_hr.append("[11,27]")
str_lr.append("[0,17,33,49,65,81]")
str_hr.append("[3,19]")
str_lr.append("[0,16,32,48,64,80,96]")


str_hr = []
str_lr = []
str_hr.append("[7,23]")
str_lr.append("[0,15,30,45,60,75,90]")
str_hr.append("[6,22]")
str_lr.append("[0,14,28,42,56,70,84]")
str_hr.append("[6,22]")
str_lr.append("[0,15,30,45,60,75,90]")
str_hr.append("[0,16]")
str_lr.append("[0,17,34,51,68,85]")
str_hr.append("[6,22]")
str_lr.append("[0,17,33,49,65,81]")
str_hr.append("[14,30]")
str_lr.append("[0,17,34,51,68,85]")
str_hr.append("[3,19]")
str_lr.append("[0,17,34,51,68,85]")
str_hr.append("[10,26]")
str_lr.append("[0,17,34,51,68,85]")

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

for i in range(7):
    print(">>>>>>>>>>>index %d <<<<<<<<< <<<" % (i+1))
    os.system("python refine_run.py /home/lgh/real_system_data7/data1/people3/HR/output %s %s /home/lgh/real_system_data7/data1/people3/LR%d/" % (str_hr[i], str_lr[i], (i+1)))
