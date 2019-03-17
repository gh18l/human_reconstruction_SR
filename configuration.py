hmr_path = "/home/lgh/code/SMPLify_TF/test/test_phone3/"
texture_path = "/home/lgh/code/SMPLify_TF/test/test_phone_HR/output/texture_file/"
HR_pose_path = "/home/lgh/code/SMPLify_TF/test/test_phone_HR/output/"
crop_texture = True  ###only use in small texture
index_data = 0
video = True
pedestrian_constraint = True
###dingjianLR100
#lr_points = [0, 16, 31, 47, 64, 80, 96]    ###[0, 18, 36, 54, 72]
#hr_points = [4, 20, 36]
###xiongfei
# lr_points = [0, 18, 36, 54, 72]    ###[0, 18, 36, 54, 72]
# hr_points = [1, 19, 37]
# lr_points = [0, 18]    ###[0, 18, 36, 54, 72]
# hr_points = [1, 19]
###jianing
# lr_points = [0, 16, 32, 48, 64, 80]    ###[0, 18, 36, 54, 72]
# hr_points = [11, 27, 43, 59]
###jianing2
lr_points = [0, 16, 32, 48, 64, 80]    ###[0, 18, 36, 54, 72]
hr_points = [14, 30, 46, 62]
###jianing2copy
#lr_points = [0, 16]    ###[0, 18, 36, 54, 72]
#hr_points = [14, 30]
### zhicheng
# lr_points = [0, 16, 31, 47, 63, 79, 95]    ###[0, 18, 36, 54, 72]
# hr_points = [6, 23, 39, 55, 71]
### zhicheng2
# lr_points = [0, 16, 32, 48, 64, 80, 96]    ###[0, 18, 36, 54, 72]
# hr_points = [6, 22, 38, 54, 70]
### zhicheng3
# lr_points = [0, 16, 32, 48, 64, 80]    ###[0, 18, 36, 54, 72]
# hr_points = [11, 27, 44, 60, 76]
### dingjian
# lr_points = [0, 16, 32, 49, 65, 81, 98]    ###[0, 18, 36, 54, 72]
# hr_points = [1, 17, 33]

iphone_param = {"hmr_constraint_hr" : 1000,
                "temporal_hr": 800,
                "dense_optflow_hr" : 0.02,
                "hmr_constraint_lr": 1000,
                "temporal_lr": 800,
                "dense_optflow_lr": 0.02}



