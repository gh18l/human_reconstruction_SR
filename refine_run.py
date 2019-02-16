import period_new
import util
HR_path = "/home/lgh/code/SMPLify_TF/test/test_hmr_init/HR_multi_crop_small2/output"
period_new.refine_LR_pose(HR_path, util.hr_points, util.lr_points)