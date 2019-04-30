import os
import numpy as np
import pickle
import smpl_np
import util
import hmr
import cv2
import correct_final_texture as tex
from opendr_render import render
def fig1():
    ind = 14
    path = "/home/lgh/real_system_data/data1/people3/HR/"
    HR_path = path + "output/"
    hmr_dict, data_dict = util.load_hmr_data(path)
    hmr_cams = hmr_dict["hmr_cams"]
    HR_imgs = data_dict["imgs"]
    HR_pkl_files = os.listdir(HR_path)
    HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))

    HR_pkl_path = os.path.join(HR_path, HR_pkl_files[ind])
    with open(HR_pkl_path) as f:
        param = pickle.load(f)
    pose = param['pose']
    beta = param['betas']
    tran = param['trans']
    hmr_cam = hmr_cams[ind, :].squeeze()
    smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    #template = np.load("/home/lgh/real_system_data/data1/people3/HR/output/texture_file/template.npy")
    #smpl.set_template(template)
    v = smpl.get_verts(pose, beta, tran)

    camera = render.camera(hmr_cam[0], hmr_cam[1], hmr_cam[2], np.zeros(3))
    bg_white = (np.ones_like(HR_imgs[ind]) * 255).astype(np.uint8)
    img_result_naked = camera.render_naked(v, bg_white)
    #img_result_naked = cv2.resize(img_result_naked, (300, 750))
    cv2.imshow("1", img_result_naked)
    cv2.waitKey()
    a = 1

def texture_to_mask():
    img = cv2.imread("/home/lgh/mask_CRF_zyp/mpii_0401/6_seg_00.png")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                continue
            if i>=930:
                img[i, j, :] = 0
                continue
            img[i, j, :] = 255
    cv2.imwrite("/home/lgh/mask_CRF_zyp/mpii_0401/label.png", img)

def stitch_texture():
    bg = cv2.imread("/home/lgh/real_system_data/data1/people3/HR/optimization_data/0028.png")
    render = cv2.imread("/home/lgh/real_system_data/data1/people3/HR/output_nonrigid/hmr_optimization_texture_nonrigid0014.png")
    bg_img = np.copy(bg)
    for i in range(render.shape[0]):
        for j in range(render.shape[1]):
            if render[i, j, 0] == 0 and render[i, j, 1] == 0 and render[i, j, 2] == 0:
                continue
            bg_img[i, j, :] = render[i, j, :]
    cv2.imwrite("/home/lgh/real_system_data/data1/people3/HR/output_nonrigid/result.png", bg_img)

def img_together():
    img = cv2.imread("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model1.png")
    mask = cv2.imread("/home/lgh/picture_for_paper/reconstruction_model/label.png")
    smpl = cv2.imread("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model3.png")
    nonrigid = cv2.imread("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model4.png")
    result = cv2.imread("/home/lgh/real_system_data/data1/people3/HR/output_nonrigid/hmr_optimization_texture_nonrigid0014.png")
    _mask = mask[:, :, 0]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0:
                continue
            _mask[i, j] = 255
    cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/mask.png", _mask)
    a = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=_mask)
    b = cv2.add(smpl, np.zeros(np.shape(smpl), dtype=np.uint8), mask=_mask)
    c = cv2.add(nonrigid, np.zeros(np.shape(nonrigid), dtype=np.uint8), mask=_mask)
    #d = cv2.add(result, np.zeros(np.shape(result), dtype=np.uint8), mask=_mask)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j, 0] == 0 and a[i, j, 1] == 0 and a[i, j, 2] == 0:
                a[i, j, :] = 255
                b[i, j, :] = 255
                c[i, j, :] = 255
            if result[i, j, 0] == 0 and result[i, j, 1] == 0 and result[i, j, 2] == 0:
                result[i, j, :] = 255
    cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model4_fullcrop_temp.png", result)

    cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model1_fullcrop.png", a)
    cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model2_fullcrop.png", b)
    cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model3_fullcrop.png", c)
    #cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/reconstruction_model4_fullcrop.png", d)
    result_weights = cv2.addWeighted(a, 0.5, img, 0.5, 0)
    cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/mask_weights.png", result_weights)
    final = np.zeros([750, 5000, 3], dtype=np.uint8)
    final[:, :1000] = img
    final[:, 1000:2000] = result_weights
    final[:, 2000:3000] = smpl
    final[:, 3000:4000] = nonrigid
    final[:, 4000:5000] = result
    cv2.imwrite("/home/lgh/picture_for_paper/reconstruction_model/final.png", final)

def experiment_motion():
    num = 10
    imgs = []
    refines = []
    origins= []
    width = 150
    height = 300
    final = np.zeros([height * 2, width * num, 3])
    for i in range(num):
        img = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_%04d.png" % (i+4))
        refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_%04d.png" % (i+4))
        #origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/%04d.png" % (i+33))
        if i >= 7:
            img = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_%04d.png" % (i + 6))
            refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_%04d.png" % (i + 6))
            #origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/%04d.png" % (i + 35))
        img = img[50:350, 25:175, :]
        refine = refine[50:350, 25:175, :]
        #origin = origin[50:350, 25:175, :]
        final[:height, i*width:(i+1)*width, :] = img
        final[height:height*2, i*width:(i+1)*width, :] = refine
        #final[height*2:height*3, i*width:(i+1)*width, :] = origin
    cv2.imwrite("/home/lgh/picture_for_paper/motion_anasiy/LR2/final.png", final)

    # img1 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0004.png")
    # img2 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0005.png")
    # img3 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0006.png")
    # img4 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0007.png")
    # img5 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0008.png")
    # img6 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0009.png")
    # img7 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0010.png")
    # #img8 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0011.png")
    # #img9 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0012.png")
    # img10 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0013.png")
    # img11 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0014.png")
    # img12 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/texture_bg_0015.png")
    #
    # img1refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0004.png")
    # img2refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0005.png")
    # img3refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0006.png")
    # img4refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0007.png")
    # img5refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0008.png")
    # img6refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0009.png")
    # img7refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0010.png")
    # #img8refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0011.png")
    # #img9refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0012.png")
    # img10refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0013.png")
    # img11refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0014.png")
    # img12refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/refine/texture_bg_0015.png")
    #
    # img1origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0033.png")
    # img2origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0034.png")
    # img3origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0035.png")
    # img4origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0036.png")
    # img5origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0037.png")
    # img6origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0038.png")
    # img7origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0039.png")
    # #img8origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0040.png")
    # #img9origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0041.png")
    # img10origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0042.png")
    # img11origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0043.png")
    # img12origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR2/0044.png")
    # width = 150
    # height = 300
    # final = np.zeros([1200, 2000, 3])
    # img1 = img1[50:350, 25:175, :]
    # final[:height, 0:150, :] = img1
    # img2 = img2[50:350, 25:175, :]
    # final[:height, 150:300, :] = img2
    # img3 = img3[50:350, 25:175, :]
    # final[:height, 300:450, :] = img3
    # img4 = img4[50:350, 25:175, :]
    # final[:height, 450:600, :] = img4
    # img5 = img5[50:350, 25:175, :]
    # final[:height, 600:750, :] = img5
    # img6 = img6[50:350, 25:175, :]
    # final[:height, 750:900, :] = img6
    # img7 = img7[50:350, 25:175, :]
    # final[:height, 900:1050, :] = img7
    # img10 = img10[50:350, 25:175, :]
    # final[:height, 1050:1200, :] = img10
    # img11 = img11[50:350, 25:175, :]
    # final[:height, 1200:1350, :] = img11
    # img12 = img12[50:350, 25:175, :]
    # final[:height, 1350:1500, :] = img12
    # # final[:400, 2000:2200, :] = img11
    # # final[:400, 2200:, :] = img12
    # img1refine = img1refine[50:350, 25:175, :]
    # final[height:height * 2, 0:150, :] = img1refine
    # img2refine = img2refine[50:350, 25:175, :]
    # final[height:height*2, 150:300, :] = img2refine
    # img3refine = img3refine[50:350, 25:175, :]
    # final[height:height*2, 300:450, :] = img3refine
    # img4refine = img4refine[50:350, 25:175, :]
    # final[height:height*2, 600:800, :] = img4refine
    # img5refine = img5refine[50:350, 25:175, :]
    # final[height:height*2, 800:1000, :] = img5refine
    # img6refine = img6refine[50:350, 25:175, :]
    # final[height:height*2, 1000:1200, :] = img6refine
    # img7refine = img7refine[50:350, 25:175, :]
    # final[height:height*2, 1200:1400, :] = img7refine
    # img10refine = img10refine[50:350, 25:175, :]
    # final[height:height*2, 1400:1600, :] = img10refine
    # img11refine = img11refine[50:350, 25:175, :]
    # final[height:height*2, 1600:1800, :] = img11refine
    # img12refine = img12refine[50:350, 25:175, :]
    # final[height:height*2, 1800:2000, :] = img12refine
    # # final[400:800, 2000:2200, :] = img11refine
    # # final[400:800, 2200:, :] = img12refine
    # img1origin = img1origin[50:350, 25:175, :]
    # final[height*2:height*3, 0:200, :] = img1origin
    # img2origin = img2origin[50:350, 25:175, :]
    # final[height*2:height*3, 200:400, :] = img2origin
    # img3origin = img3origin[50:350, 25:175, :]
    # final[height*2:height*3, 400:600, :] = img3origin
    # img4origin = img4origin[50:350, 25:175, :]
    # final[height*2:height*3, 600:800, :] = img4origin
    # img5origin = img5origin[50:350, 25:175, :]
    # final[height*2:height*3, 800:1000, :] = img5origin
    # img6origin = img6origin[50:350, 25:175, :]
    # final[height*2:height*3, 1000:1200, :] = img6origin
    # img7origin = img7origin[50:350, 25:175, :]
    # final[height*2:height*3, 1200:1400, :] = img7origin
    # img10origin = img10origin[50:350, 25:175, :]
    # final[height*2:height*3, 1400:1600, :] = img10origin
    # img11origin = img11origin[50:350, 25:175, :]
    # final[height*2:height*3, 1600:1800, :] = img11origin
    # img12origin = img12origin[50:350, 25:175, :]
    # final[height*2:height*3, 1800:2000, :] = img12origin
    # # final[800:1200, 2000:2200, :] = img11origin
    # # final[800:1200, 2200:, :] = img12origin
    # cv2.imwrite("/home/lgh/picture_for_paper/motion_anasiy/LR2/final.png", final)

def experiment_motion2():
    img1 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/texture_bg_0071.png")
    img2 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/texture_bg_0072.png")
    img3 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/texture_bg_0073.png")
    img4 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/texture_bg_0074.png")
    img5 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/texture_bg_0075.png")
    img6 = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/texture_bg_0076.png")
    img1refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/refine/texture_bg_0071.png")
    img2refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/refine/texture_bg_0072.png")
    img3refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/refine/texture_bg_0073.png")
    img4refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/refine/texture_bg_0074.png")
    img5refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/refine/texture_bg_0075.png")
    img6refine = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/refine/texture_bg_0076.png")
    img1origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/0071.png")
    img2origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/0072.png")
    img3origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/0073.png")
    img4origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/0074.png")
    img5origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/0075.png")
    img6origin = cv2.imread("/home/lgh/picture_for_paper/motion_anasiy/LR3/0076.png")
    final = np.zeros([1200, 1200, 3])
    final[:400, 0:200, :] = img1
    final[:400, 200:400, :] = img2
    final[:400, 400:600, :] = img3
    final[:400, 600:800, :] = img4
    final[:400, 800:1000, :] = img5
    final[:400, 1000:, :] = img6
    final[400:800, 0:200, :] = img1refine
    final[400:800, 200:400, :] = img2refine
    final[400:800, 400:600, :] = img3refine
    final[400:800, 600:800, :] = img4refine
    final[400:800, 800:1000, :] = img5refine
    final[400:800, 1000:, :] = img6refine

    final[800:1200, 0:200, :] = img1origin
    final[800:1200, 200:400, :] = img2origin
    final[800:1200, 400:600, :] = img3origin
    final[800:1200, 600:800, :] = img4origin
    final[800:1200, 800:1000, :] = img5origin
    final[800:1200, 1000:, :] = img6origin
    cv2.imwrite("/home/lgh/picture_for_paper/motion_anasiy/LR3/final.png", final)

def img_together2():
    img1 = cv2.imread("/home/lgh/picture_for_paper/extend/0028.png")
    img2 = cv2.imread("/home/lgh/picture_for_paper/extend/untitled.png")
    img2 = cv2.resize(img2, (1000, 750))
    final = np.zeros([750, 2000, 3], dtype=np.uint8)
    final[:, :1000, :] = img1
    final[:, 1000:, :] = img2
    cv2.imwrite("/home/lgh/picture_for_paper/extend/final.png", final)

def crop_img():
    img1 = cv2.imread("/media/lgh/6626-63BC/SR5/origin.png")
    img2 = cv2.imread("/media/lgh/6626-63BC/SR5/imgSR.png")
    img3 = cv2.imread("/media/lgh/6626-63BC/SR5/videoSR.png")
    img4 = cv2.imread("/media/lgh/6626-63BC/SR5/texture.png")
    img5 = cv2.imread("/media/lgh/6626-63BC/SR5/gt.jpg")
    #img6 = cv2.imread("/home/lgh/picture_for_paper/reconstruction_model4/reconstruction6.png")
    img1 = img1[100:1000, 500:1400, :]
    img2 = img2[100:1000, 500:1400, :]
    img3 = img3[100:1000, 500:1400, :]
    img4 = img4[100:1000, 500:1400, :]
    img5 = img5[100:1000, 500:1400, :]
    #img6 = img6[100:980, 700:1180, :]
    # for i in range(img3.shape[0]):
    #     for j in range(img3.shape[1]):
    #         if img3[i, j, 0] == 0 and img3[i, j, 1] == 0 and img3[i, j, 2] == 0:
    #             img3[i, j, :] = 255
    # for i in range(img5.shape[0]):
    #     for j in range(img5.shape[1]):
    #         if img5[i, j, 0] == 0 and img5[i, j, 1] == 0 and img5[i, j, 2] == 0:
    #             img5[i, j, :] = 255
    # for i in range(img6.shape[0]):
    #     for j in range(img6.shape[1]):
    #         if img6[i, j, 0] == 0 and img6[i, j, 1] == 0 and img6[i, j, 2] == 0:
    #             img6[i, j, :] = 255
    cv2.imwrite("/media/lgh/6626-63BC/SR5/output/origin5.png", img1)
    cv2.imwrite("/media/lgh/6626-63BC/SR5/output/imgSR5.png", img2)
    cv2.imwrite("/media/lgh/6626-63BC/SR5/output/videoSR5.png", img3)
    cv2.imwrite("/media/lgh/6626-63BC/SR5/output/our5.png", img4)
    cv2.imwrite("/media/lgh/6626-63BC/SR5/output/gt5.png", img5)
    #cv2.imwrite("/media/lgh/6626-63BC/SR1/output/gt", img6)
def delete_bg():
    img = cv2.imread("/home/lgh/picture_for_paper/extend/correspondence2.png")
    img = cv2.resize(img, (1000, 750))
    mask = cv2.imread("/home/lgh/picture_for_paper/extend/label.png")
    _mask = mask[:, :, 0]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0:
                continue
            _mask[i, j] = 255
    crop = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=_mask)
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            if crop[i, j, 0] == 0 and crop[i, j, 1] == 0 and crop[i, j, 2] == 0:
                crop[i, j, :] = 255
    crop = crop[100:650, 500:750]
    cv2.imwrite("/home/lgh/picture_for_paper/extend/correspondence2_crop.png", crop)


def generate_new_data():
    path = "/home/lgh/real_system_data5/data1/people10/"
    HR_path = path + "HR/"
    LR_path = path + "LR1/"
    LR_output_path = LR_path + "output/"
    #LR_output_path = LR_path + "output_after_refine/"
    #hmr_dict, data_dict = util.load_hmr_data(path)
    img_files = os.listdir(LR_path + "optimization_data")
    img_files = sorted([filename for filename in img_files if (filename.endswith(".png") or filename.endswith(
        ".jpg")) and "mask" not in filename and "label" not in filename])

    imgs = []
    for ind in range(len(img_files)):
        img_file_path = os.path.join(LR_path + "optimization_data", img_files[ind])
        img = cv2.imread(img_file_path)
        imgs.append(img)

    pkl_files = os.listdir(LR_output_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    for ind, pkl_file in enumerate(pkl_files):
        LR_pkl_path = os.path.join(LR_output_path, pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        beta = param['betas']
        tran = param['trans']
        cam = param['cam']

        smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        template = np.load(HR_path + "output/texture_file/template.npy")
        smpl.set_template(template)
        v = smpl.get_verts(pose, beta, tran)

        camera = render.camera(cam[0], cam[1], cam[2], cam[3])

        ##render
        #texture_img = cv2.imread(HR_path + "output/texture_file/HR1.png")
        texture_img = cv2.imread(HR_path + "output/texture_file/HR.png")
        texture_vt = np.load(HR_path + "output/texture_file/vt.npy")
        img_result_texture = camera.render_texture(v, texture_img, texture_vt)
        # img_result_texture = tex.correct_render_small(img_result_texture, 3)
        if not os.path.exists(LR_path + "new_output"):
            os.makedirs(LR_path + "new_output")
        cv2.imwrite(LR_path + "new_output/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
        img_bg = cv2.resize(imgs[ind], (util.img_width, util.img_height))
        img_result_texture_bg = camera.render_texture_imgbg(img_result_texture, imgs[ind])
        cv2.imwrite(LR_path + "new_output/texture_bg_%04d.png" % ind,
                    img_result_texture_bg)
        img_result_naked = camera.render_naked(v, imgs[ind])
        cv2.imwrite(LR_path + "new_output/hmr_optimization_%04d.png" % ind, img_result_naked)
        img_result_naked_rotation = camera.render_naked_rotation(v, 90, imgs[ind])
        cv2.imwrite(LR_path + "new_output/hmr_optimization_rotation_%04d.png" % ind, img_result_naked_rotation)

def generate_new_data1():
    path = "/home/lgh/real_system_data/data1/people3/"
    HR_path = path + "HR/"
    LR_path = path + "LR1/"
    LR_output_path = LR_path + "output_after_refine1/"
    #LR_output_path = LR_path + "output_after_refine/"
    #hmr_dict, data_dict = util.load_hmr_data(path)
    img_files = os.listdir(LR_path + "optimization_data")
    img_files = sorted([filename for filename in img_files if (filename.endswith(".png") or filename.endswith(
        ".jpg")) and "mask" not in filename and "label" not in filename])

    imgs = []
    for ind in range(len(img_files)):
        img_file_path = os.path.join(LR_path + "optimization_data", img_files[ind])
        img = cv2.imread(img_file_path)
        imgs.append(img)

    pkl_files = os.listdir(LR_output_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    for ind, pkl_file in enumerate(pkl_files):
        LR_pkl_path = os.path.join(LR_output_path, pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        beta = param['betas']
        tran = param['trans']
        cam = param['cam']

        smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        template = np.load(HR_path + "output/texture_file/template.npy")
        smpl.set_template(template)
        v = smpl.get_verts(pose, beta, tran)

        camera = render.camera(cam[0], cam[1], cam[2], cam[3])

        ##render
        #texture_img = cv2.imread(HR_path + "output/texture_file/HR1.png")
        texture_img = cv2.imread(HR_path + "output/texture_file/HR.png")
        texture_vt = np.load(HR_path + "output/texture_file/vt.npy")
        img_result_texture = camera.render_texture(v, texture_img, texture_vt)
        # img_result_texture = tex.correct_render_small(img_result_texture, 3)
        if not os.path.exists(LR_path + "new_output"):
            os.makedirs(LR_path + "new_output")
        cv2.imwrite(LR_path + "new_output/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
        img_bg = cv2.resize(imgs[ind], (util.img_width, util.img_height))
        img_result_texture_bg = camera.render_texture_imgbg(img_result_texture, img_bg)
        cv2.imwrite(LR_path + "new_output/texture_bg_%04d.png" % ind,
                    img_result_texture_bg)
        img_result_naked = camera.render_naked(v, imgs[ind])
        cv2.imwrite(LR_path + "new_output/hmr_optimization_%04d.png" % ind, img_result_naked)
        img_result_naked_rotation = camera.render_naked_rotation(v, 90, imgs[ind])
        cv2.imwrite(LR_path + "new_output/hmr_optimization_rotation_%04d.png" % ind, img_result_naked_rotation)

def img16transformresize8():
    img = cv2.imread("/home/lgh/MPIIdatasets/img16_resize8/optimization_data/0006.png")

    with open("/home/lgh/MPIIdatasets/img16_resize/output/hmr_optimization_pose_0006.pkl") as f:
        param = pickle.load(f)
        pose = param['pose']
        beta = param['betas']
        tran = param['trans']
        cam = param['cam_LR1']

    smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    template = np.load("/home/lgh/MPIIdatasets/img16/output/texture_file/template.npy")
    smpl.set_template(template)
    v = smpl.get_verts(pose, beta, tran)

    camera = render.camera(cam[0], cam[1], cam[2], cam[3])

        ##render
    texture_img = cv2.imread("/home/lgh/MPIIdatasets/img16/output/texture_file/HR.png")
    #texture_img = cv2.imread(HR_path + "output/texture_file/HR.png")
    texture_vt = np.load("/home/lgh/MPIIdatasets/img16/output/texture_file/vt.npy")
    img_result_texture = camera.render_texture(v, texture_img, texture_vt)
    img_result_texture = tex.correct_render_small(img_result_texture)
    img_result_texture_bg = camera.render_texture_imgbg(img_result_texture, img)
    cv2.imwrite("/media/lgh/6626-63BC/SR5/texture.png",
                    img_result_texture_bg)

def crop_single_img():
    img = cv2.imread("/media/lgh/ACE-Z1/data_from_gigadata/HIT_canteen/hit_4_mp4/texture/texture.png")
    img1 = img[800:1800, :500, :]
    #img2 = img[300:, :, :]
    cv2.imwrite("/media/lgh/ACE-Z1/data_from_gigadata/HIT_canteen/hit_4_mp4/texture/texture1.png", img1)
    #cv2.imwrite("/home/lgh/picture_for_paper/motion_anasiy/LR2/motion_analysis2.png", img2)

def generate_realsystem_SR():
    img = cv2.imread("/home/lgh/real_system_data3/data1/people1/LR/new_output/hmr_optimization_texture_0044.png")
    #img = cv2.resize(img, (4000, 3000))
    #cv2.imwrite("/home/lgh/real_system_data/data1/people3/LR/output_after_refine/texture_bg_0050_compare.png", img)
    imgSR = cv2.imread("/media/lgh/6626-63BC/realsystemSR/0044image.png")
    videoSR = cv2.imread("/media/lgh/6626-63BC/realsystemSR/0044video.png")
    origin = cv2.imread("/home/lgh/real_system_data3/data1/people1/LR/optimization_data/0044.png")
    origin = cv2.resize(origin, (600, 1200))
    origin1 = np.copy(origin)
    #
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                continue
            origin1[i,j,:] = img[i,j,:]

    img = cv2.resize(origin1, (800, 1600))
    origin = cv2.resize(origin, (800, 1600))
    img = img[270:1400,150:635, :]
    imgSR = imgSR[270:1400,150:635, :]
    videoSR = videoSR[270:1400,150:635, :]
    origin = origin[270:1400,150:635, :]
    cv2.imwrite("/media/lgh/6626-63BC/SR8/origin8.png", origin)
    cv2.imwrite("/media/lgh/6626-63BC/SR8/imgSR8.png", imgSR)
    cv2.imwrite("/media/lgh/6626-63BC/SR8/videoSR8.png", videoSR)
    cv2.imwrite("/media/lgh/6626-63BC/SR8/our8.png", img)

def addGaussianNoise(image,percetage):
    G_Noiseimg = image
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(20,40)
        temp_y = np.random.randint(20,40)
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg
def make_teaser():
    bg = cv2.imread("/home/lgh/teaser/FZ9A3079_DxO.jpg")
    bg = cv2.resize(bg, (2048/8, 1365/8))
    bg = cv2.resize(bg, (2048, 1365))
    cv2.imwrite("/home/lgh/teaser/bg_ds.jpg", bg)

    mask1 = cv2.imread("/media/lgh/6626-63BC/human/1_json/label.png")
    mask2 = cv2.imread("/media/lgh/6626-63BC/human/2_json/label.png")
    mask3 = cv2.imread("/media/lgh/6626-63BC/human/3_json/label.png")
    mask4 = cv2.imread("/media/lgh/6626-63BC/human/4_json/label.png")
    mask5 = cv2.imread("/media/lgh/6626-63BC/human/5_json/label.png")
    mask6 = cv2.imread("/media/lgh/6626-63BC/human/6_json/label.png")
    mask7 = cv2.imread("/media/lgh/6626-63BC/human/7_json/label.png")
    mask8 = cv2.imread("/media/lgh/6626-63BC/human/8_json/label.png")
    mask9 = cv2.imread("/media/lgh/6626-63BC/human/9_json/label.png")
    mask1_ = mask1[:,:,0]
    mask2_ = mask2[:, :, 0]
    mask3_ = mask3[:, :, 0]
    mask4_ = mask4[:, :, 0]
    mask5_ = mask5[:, :, 0]
    mask6_ = mask6[:, :, 0]
    mask7_ = mask7[:, :, 0]
    mask8_ = mask8[:, :, 0]
    mask9_ = mask9[:, :, 0]
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j, 0] == 0 and mask1[i, j, 1] == 0 and mask1[i, j, 2] == 0:
                continue
            mask1_[i,j] = 255
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i, j, 0] == 0 and mask2[i, j, 1] == 0 and mask2[i, j, 2] == 0:
                continue
            mask2_[i,j] = 255
    for i in range(mask3.shape[0]):
        for j in range(mask3.shape[1]):
            if mask3[i, j, 0] == 0 and mask3[i, j, 1] == 0 and mask3[i, j, 2] == 0:
                continue
            mask3_[i,j] = 255
    for i in range(mask4.shape[0]):
        for j in range(mask4.shape[1]):
            if mask4[i, j, 0] == 0 and mask4[i, j, 1] == 0 and mask4[i, j, 2] == 0:
                continue
            mask4_[i,j] = 255
    for i in range(mask5.shape[0]):
        for j in range(mask5.shape[1]):
            if mask5[i, j, 0] == 0 and mask5[i, j, 1] == 0 and mask5[i, j, 2] == 0:
                continue
            mask5_[i,j] = 255
    for i in range(mask6.shape[0]):
        for j in range(mask6.shape[1]):
            if mask6[i, j, 0] == 0 and mask6[i, j, 1] == 0 and mask6[i, j, 2] == 0:
                continue
            mask6_[i,j] = 255
    for i in range(mask7.shape[0]):
        for j in range(mask7.shape[1]):
            if mask7[i, j, 0] == 0 and mask7[i, j, 1] == 0 and mask7[i, j, 2] == 0:
                continue
            mask7_[i,j] = 255
    for i in range(mask8.shape[0]):
        for j in range(mask8.shape[1]):
            if mask8[i, j, 0] == 0 and mask8[i, j, 1] == 0 and mask8[i, j, 2] == 0:
                continue
            mask8_[i,j] = 255
    for i in range(mask9.shape[0]):
        for j in range(mask9.shape[1]):
            if mask9[i, j, 0] == 0 and mask9[i, j, 1] == 0 and mask9[i, j, 2] == 0:
                continue
            mask9_[i,j] = 255
    img1 = cv2.imread("/media/lgh/6626-63BC/human/1.png")
    img2 = cv2.imread("/media/lgh/6626-63BC/human/2.png")
    img3 = cv2.imread("/media/lgh/6626-63BC/human/3.png")
    img4 = cv2.imread("/media/lgh/6626-63BC/human/4.png")
    img5 = cv2.imread("/media/lgh/6626-63BC/human/5.png")
    img6 = cv2.imread("/media/lgh/6626-63BC/human/6.png")
    img7 = cv2.imread("/media/lgh/6626-63BC/human/7.png")
    img8 = cv2.imread("/media/lgh/6626-63BC/human/8.png")
    img9 = cv2.imread("/media/lgh/6626-63BC/human/9.png")
    result1 = cv2.add(img1, np.zeros(np.shape(img1), dtype=np.uint8), mask=mask1_)
    result2 = cv2.add(img2, np.zeros(np.shape(img2), dtype=np.uint8), mask=mask2_)
    result3 = cv2.add(img3, np.zeros(np.shape(img3), dtype=np.uint8), mask=mask3_)
    result4 = cv2.add(img4, np.zeros(np.shape(img4), dtype=np.uint8), mask=mask4_)
    result5 = cv2.add(img5, np.zeros(np.shape(img5), dtype=np.uint8), mask=mask5_)
    result6 = cv2.add(img6, np.zeros(np.shape(img6), dtype=np.uint8), mask=mask6_)
    result7 = cv2.add(img7, np.zeros(np.shape(img7), dtype=np.uint8), mask=mask7_)
    result8 = cv2.add(img8, np.zeros(np.shape(img8), dtype=np.uint8), mask=mask8_)
    result9 = cv2.add(img9, np.zeros(np.shape(img9), dtype=np.uint8), mask=mask9_)
    # result1 = tex.correct_render_small(result1, 3)
    # result2 = tex.correct_render_small(result2, 3)
    # result3 = tex.correct_render_small(result3, 3)
    # result4 = tex.correct_render_small(result4, 3)
    # result5 = tex.correct_render_small(result5, 3)
    # result6 = tex.correct_render_small(result6, 3)
    # result7 = tex.correct_render_small(result7, 3)
    # result8 = tex.correct_render_small(result8, 3)
    # result9 = tex.correct_render_small(result9, 3)
    cv2.imwrite("/home/lgh/teaser/temp1.png", result1)
    cv2.imwrite("/home/lgh/teaser/temp2.png", result2)
    cv2.imwrite("/home/lgh/teaser/temp3.png", result3)
    cv2.imwrite("/home/lgh/teaser/temp4.png", result4)
    cv2.imwrite("/home/lgh/teaser/temp5.png", result5)
    cv2.imwrite("/home/lgh/teaser/temp6.png", result6)
    cv2.imwrite("/home/lgh/teaser/temp7.png", result7)
    cv2.imwrite("/home/lgh/teaser/temp8.png", result8)
    cv2.imwrite("/home/lgh/teaser/temp9.png", result9)

    for i in range(result1.shape[0]):
        for j in range(result1.shape[1]):
            if result1[i,j,0] == 0 and result1[i,j,1] == 0 and result1[i,j,2] == 0:
                continue
            bg[609+i, 1120+j,:] = result1[i, j]
    #bg[609:878, 1120:1261, :] = result
    for i in range(result2.shape[0]):
        for j in range(result2.shape[1]):
            if result2[i,j,0] == 0 and result2[i,j,1] == 0 and result2[i,j,2] == 0:
                continue
            bg[448+i, 1419+j,:] = result2[i, j]
    # for i in range(result3.shape[0]):
    #     for j in range(result4.shape[1]):
    #         if result3[i,j,0] == 0 and result3[i,j,1] == 0 and result3[i,j,2] == 0:
    #             continue
    #         bg[675+i, 1354+j,:] = result3[i, j]
    for i in range(result4.shape[0]):
        for j in range(result4.shape[1]):
            if result4[i,j,0] == 0 and result4[i,j,1] == 0 and result4[i,j,2] == 0:
                continue
            bg[675+i, 1354+j,:] = result4[i, j]
    #bg[609:878, 1120:1261, :] = result
    for i in range(result5.shape[0]):
        for j in range(result5.shape[1]):
            if result5[i,j,0] == 0 and result5[i,j,1] == 0 and result5[i,j,2] == 0:
                continue
            bg[486+i, 722+j,:] = result5[i, j]
    for i in range(result6.shape[0]):
        for j in range(result6.shape[1]):
            if result6[i,j,0] == 0 and result6[i,j,1] == 0 and result6[i,j,2] == 0:
                continue
            bg[491+i, 694+j,:] = result6[i, j]
    for i in range(result7.shape[0]):
        for j in range(result7.shape[1]):
            if result7[i,j,0] == 0 and result7[i,j,1] == 0 and result7[i,j,2] == 0:
                continue
            bg[580+i, 1766+j,:] = result7[i, j]
    #bg[609:878, 1120:1261, :] = result
    for i in range(result8.shape[0]):
        for j in range(result8.shape[1]):
            if result8[i,j,0] == 0 and result8[i,j,1] == 0 and result8[i,j,2] == 0:
                continue
            bg[509+i, 328+j,:] = result8[i, j]
    for i in range(result9.shape[0]):
        for j in range(result9.shape[1]):
            if result9[i,j,0] == 0 and result9[i,j,1] == 0 and result9[i,j,2] == 0:
                continue
            bg[511+i, 654+j,:] = result9[i, j]
    #bg[609:878, 1120:1261, :] = result
    cv2.imwrite("/home/lgh/teaser/final.png", bg)

def generate_zhengfangxingSR():
    origin = cv2.imread("/media/lgh/6626-63BC/SR8/origin8.png")
    imgSR = cv2.imread("/media/lgh/6626-63BC/SR8/imgSR8.png")
    videoSR = cv2.imread("/media/lgh/6626-63BC/SR8/videoSR8.png")
    ours = cv2.imread("/media/lgh/6626-63BC/SR8/our8.png")
    bg = cv2.imread("/home/lgh/real_system_data3/data1/people1/LR/optimization_data/0044.png")
    x = 2486*4+150
    y = 1158*4+270
    video_full_path = "/home/lgh/real_system_data3/data1/human_render/ref.avi"
    cap = cv2.VideoCapture(video_full_path)
    for i in range(47):
        success, frame = cap.read()
    success, frame = cap.read()
    frame1 = cv2.resize(frame, (16000, 8000))
    frame2 = cv2.resize(frame, (16000, 8000))
    frame3 = cv2.resize(frame, (16000, 8000))
    frame4 = cv2.resize(frame, (16000, 8000))

    frame1[y:(y+1130), x:(x+485),:] = origin
    new_origin = frame1[600*8:750*8, 1225*8:1375*8,:]
    frame2[y:(y + 1130), x:(x + 485), :] = imgSR
    new_imgSR = frame2[600 * 8:750 * 8, 1225 * 8:1375 * 8, :]
    frame3[y:(y + 1130), x:(x + 485), :] = videoSR
    new_videoSR = frame3[600 * 8:750 * 8, 1225 * 8:1375 * 8, :]
    frame4[y:(y + 1130), x:(x + 485), :] = ours
    new_ours = frame4[600 * 8:750 * 8, 1225 * 8:1375 * 8, :]
    cv2.imwrite("/media/lgh/6626-63BC/SR9/origin8.png", new_origin)
    cv2.imwrite("/media/lgh/6626-63BC/SR9/imgSR8.png", new_imgSR)
    cv2.imwrite("/media/lgh/6626-63BC/SR9/videoSR8.png", new_videoSR)
    cv2.imwrite("/media/lgh/6626-63BC/SR9/our8.png", new_ours)
    #frame = cv2.resize(frame, (2000, 1000))
    #cv2.imshow("1",frame)
    #cv2.waitKey()
def forpipeline():
    for ind in range(16):
        path = "/home/lgh/real_system_data/data1/people1/HR/"
        HR_path = path + "output/"
        hmr_dict, data_dict = util.load_hmr_data(path)
        hmr_cams = hmr_dict["hmr_cams"]
        HR_imgs = data_dict["imgs"]
        HR_pkl_files = os.listdir(HR_path)
        HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                              key=lambda d: int((d.split('_')[3]).split('.')[0]))

        HR_pkl_path = os.path.join(HR_path, HR_pkl_files[ind])
        with open(HR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        beta = param['betas']
        tran = param['trans']
        hmr_cam = hmr_cams[ind, :].squeeze()
        smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        template = np.load("/home/lgh/real_system_data/data1/people1/HR/output/texture_file/template.npy")
        smpl.set_template(template)
        v = smpl.get_verts(pose, beta, tran)

        camera = render.camera(hmr_cam[0], hmr_cam[1], hmr_cam[2], np.zeros(3))
        bg_white = (np.ones_like(HR_imgs[ind]) * 255).astype(np.uint8)
        texture_img = cv2.imread("/home/lgh/real_system_data/data1/people1/HR/output/texture_file/HR.png")
        # texture_img = cv2.imread(HR_path + "output/texture_file/HR.png")
        texture_vt = np.load("/home/lgh/real_system_data/data1/people1/HR/output/texture_file/vt.npy")
        img_result_texture = camera.render_texture(v, texture_img, texture_vt)
        cv2.imwrite("/home/lgh/real_system_data/data1/people1/HR/forpaper/%04d.png" % ind, img_result_texture)
    # img_result_naked = cv2.resize(img_result_naked, (300, 750))

def crop_img_for_nonrigid():
    img = cv2.imread("/home/lgh/real_system_data5/data1/people9/HR/optimization_data/0040.jpg")
    mask = cv2.imread("/home/lgh/real_system_data5/data1/people9/HR/optimization_data/mask_0000.png")
    label = cv2.imread("/home/lgh/real_system_data5/data1/people9/HR/optimization_data/label.png")

    img1 = img[:, 0:300,:]
    mask1 = mask[:,0:300,:]
    label1 = label[:, 0:300, :]
    cv2.imwrite("/home/lgh/real_system_data5/data1/people9/0000.jpg", img1)
    cv2.imwrite("/home/lgh/real_system_data5/data1/people9/mask_0000.png", mask1)
    cv2.imwrite("/home/lgh/real_system_data5/data1/people9/label.png", label1)

def to_render():
    mesh = np.load("/home/lgh/octopus/out/sample.npy")
    cam = np.array([1080.0, 540.0])
    camera = render.camera(cam[0], cam[1], cam[1], np.zeros(3))
    bg = cv2.imread("/home/lgh/octopus/data/sample/frames/0000.png")
    img_result_naked = camera.render_naked(mesh, bg)
    cv2.imwrite("/home/lgh/octopus/out/img_result.png", img_result_naked)
to_render()
#crop_img_for_nonrigid()
# forpipeline()
#generate_zhengfangxingSR()
#make_teaser()
#generate_realsystem_SR()
#generate_new_data()
#crop_single_img()
#img16transformresize8()
#crop_img()
#texture_to_mask()
#delete_bg()
#generate_new_data()

#crop_img()
#img_together()
#img_together2()
#experiment_motion()
#img_together()
#stitch_texture()
#fig1()

