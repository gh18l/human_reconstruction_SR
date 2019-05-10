import period_new
import util
import sys
#HR_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small4/output"
#HR_path = util.HR_pose_path
period_new.refine_LR_pose(util.HR_pose_path, util.hr_points, util.lr_points)