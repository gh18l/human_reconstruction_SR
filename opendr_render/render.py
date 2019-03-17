import numpy as np
import cv2
import core
from libcpp_render import cpp_render
import os
import util
class camera():
    def __init__(self, focal_length, center_x, center_y, trans):
        self.focal = focal_length.astype(np.float32)
        self.cx = center_x.astype(np.float32)
        self.cy = center_y.astype(np.float32)
        self.trans = trans.astype(np.float32)
        self.renderer = core.SMPLRenderer(face_path="opendr_render/smpl_faces.npy")

    def render_naked(self, verts, img=None):
        ## camera_for_render [focal center_x center_y camera_t_x camera_t_y camera_t_z]
        camera_for_render = np.hstack([self.focal, self.cx, self.cy, self.trans])
        render_result = self.renderer(verts, cam=camera_for_render, img=img, do_alpha=True)
        return render_result

    def render_naked_rotation(self, verts, angle, img=None):
        ## camera_for_render [focal center_x center_y camera_t_x camera_t_y camera_t_z]
        camera_for_render = np.hstack([self.focal, self.cx, self.cy, self.trans])
        render_result = self.renderer.rotated(verts, angle, cam=camera_for_render, img_size=img.shape[:2])
        return render_result

    def generate_uv(self, verts, image=None):
        rt = np.zeros(3)
        t = self.trans
        camera_matrix = np.array([[self.focal, 0.0, self.cx],
                                  [0.0, self.focal, self.cy],
                                  [0.0, 0.0, 1.0]])
        k = np.zeros(5)

        uv_real = cv2.projectPoints(verts, rt, t, camera_matrix, k)[0].squeeze()
        width = 600.0 if image is None else float(image.shape[1])
        height = 450.0 if image is None else float(image.shape[0])
        uv_u = (uv_real[:, 0] / width).reshape((len(uv_real), 1))
        uv_v = (uv_real[:, 1] / height).reshape((len(uv_real), 1))
        uv_norm = np.hstack((uv_u, uv_v))

        return uv_real, uv_norm

    def write_camera(self, path):
        f = open(path, "w")
        f.write(str(self.focal) + " " + str(self.trans[0]) + " " + str(self.trans[1]) + " " + str(self.trans[2])
                    + " " + str(self.cx) + " " + str(self.cy))
        f.close()

    def write_obj(self, path, verts, uv=None):
        with open(path, 'w') as fp:
            fp.write("mtllib test.mtl\n")
            fp.write("\n")
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            fp.write("\n")
            if uv is not None:
                for uv in uv:
                    fp.write('vt %f %f %f\n' % (uv[0], uv[1], 0.))
            fp.write("\n")
            fp.write("o m_avg\n")
            fp.write("g m_avg\n")
            fp.write("usemtl lambert1\n")
            fp.write("s 1\n")

            for face in self.renderer.faces + 1:
                fp.write('f %d/%d %d/%d %d/%d\n' % (face[0], face[0], face[1], face[1], face[2], face[2]))

    '''
    img is texture image, arbitrary size
    '''
    def render_texture(self, verts, img, vt):
        self.write_camera("./render_temp/camera.txt")
        self.write_obj("./render_temp/model.obj", verts, vt)
        cv2.imwrite("./render_temp/HR.png", img)
        cpp_render(util.img_widthheight)
        render_result = cv2.imread("./render_temp/result.png")
        return render_result

    def render_texture_imgbg(self, render_result, bg):
        bg_img = np.copy(bg)
        for i in range(render_result.shape[0]):
            for j in range(render_result.shape[1]):
                if render_result[i, j, 0] == 0 and render_result[i, j, 1] == 0 and render_result[i, j, 2] == 0:
                    continue
                bg_img[i, j, :] = render_result[i, j, :]
        return bg_img
    '''
    save cropped texture
    '''
    def save_texture_img(self, img, mask=None):
        if mask is not None:
            img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8),
                mask=mask)
        return img

    '''
    img is texture image, arbitrary size
    '''
    def write_texture_data(self, texture_path, img, vt):
        if not os.path.exists(texture_path):
            os.makedirs(texture_path)
        np.save(texture_path + "vt.npy", vt)
        cv2.imwrite(texture_path + "HR.png", img)

def read_texture_data(texture_path):
    vt = np.load(texture_path + "vt.npy")
    img = cv2.imread(texture_path + "HR.png")
    return vt, img

def opencv2render(img):
    b, g, r = cv2.split(img)
    img_ = cv2.merge([r, g, b])
    return img_

def save_nonrigid_template(texture_path, template):
    np.save(texture_path + "template.npy", template)