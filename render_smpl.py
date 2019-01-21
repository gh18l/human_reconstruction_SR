'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python render_smpl.py


'''

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
import os
import pickle as pkl

def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


smpl_parameter_path = "/home/lgh/code/SMPLify/smplify_public/code/temp/DCToutput"
colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}
## Load SMPL model (here we load the female model)
m = load_model('../../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

## load smpl parameters
params = []
smpl_parameters_names = os.listdir(smpl_parameter_path)
smpl_parameters_names = sorted([filename for filename in smpl_parameters_names if filename.endswith(".pkl")],
                        key=lambda d: int(d.split('.')[0]))
for ind, smpl_parameters_name in enumerate(smpl_parameters_names):
    with open(os.path.join(smpl_parameter_path, smpl_parameters_name)) as f:
        param = pkl.load(f)
    params.append(param)

## Assign random pose and shape parameters
for i in range(0, len(params)):
    m.pose[:] = params[i]['pose']
    m.betas[:] = params[i]['betas']
    ## Create OpenDR renderer
    rn = ColoredRenderer()

    ## Assign attributes to renderer
    w, h = (640, 480)
    rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w, w]) / 2.,
                              c=np.array([w, h]) / 2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m, f=m.f, vc = colors['pink'], bgcolor=np.ones(3))
    ## Construct point light source
    albedo = rn.vc
    yrot = np.radians(120)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(m),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1., 1., 1.]))
    img = rn.r
    img = img * 255.0
    ## Show it using OpenCV
    import cv2

    #cv2.imshow('render_SMPL', rn.r)
    output = "/home/lgh/code/SMPLify/smplify_public/code/temp/DCToutput/DCT0/rendered_%d.png" % (i)
    cv2.imwrite(output, img)
    #cv2.waitKey(0)









## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()