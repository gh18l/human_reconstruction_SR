import period_new
import util
import sys
#HR_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small4/output"
#HR_path = util.HR_pose_path
HR_path = sys.argv[1]
print(util.hr_points.dtype)
period_new.refine_LR_pose(HR_path, util.hr_points, util.lr_points)