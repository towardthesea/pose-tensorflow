#!/usr/bin/env python
import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
import matplotlib.pyplot as plt
from util.timer import Timer
import cv2
import yarp
import argparse
import numpy as np


def read_yarp_image(inport):

    # Create numpy array to receive the image and the YARP image wrapped around it
    img_array = np.ones((240, 320, 3), dtype=np.uint8)
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    # Read the data from the port into the image
    inport.read(yarp_image)

    return img_array, yarp_image


def write_yarp_image(outport, img_array):
    # Create the yarp image and wrap it around the array
    # img_array = img_array[:, :, (2, 1, 0)]
    yarp_img = yarp.ImageRgb()
    yarp_img.resize(320, 240)
    yarp_img.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    outport.write(yarp_img)


def draw_links(im, pose, threshold=0.5):

    num_joints = pose.shape[0]
    pose_conf = pose[:, 2]
    pose = pose.astype(int)
    im = im.copy()
    for jnt_idx in range(6, 11):
        if pose_conf[jnt_idx] >= threshold and pose_conf[jnt_idx+1] >= threshold:
            pt1 = (pose[jnt_idx, 0], pose[jnt_idx, 1])
            pt2 = (pose[jnt_idx+1, 0], pose[jnt_idx+1, 1])
            cv2.line(im, pt1, pt2, color=(255, 0, 0), thickness=3)
    pt1 = (pose[12, 0], pose[12, 1])
    pt2 = (pose[13, 0], pose[13, 1])
    cv2.line(im, pt1, pt2, color=(0, 0, 255), thickness=3)
    return im


def add_part(part_list, part_pose, part_name):
    part = yarp.Bottle()
    part.addString(part_name)
    part.addDouble(part_pose[0])
    part.addDouble(part_pose[1])
    part.addDouble(part_pose[2])

    part_list.addList().read(part)


def stream_parts(port, pose, threshold=0.5):

    all_joints_names = cfg.all_joints_names

    all_body_parts = port.prepare()
    all_body_parts.clear()

    body_parts = yarp.Bottle()
    body_parts.clear()

    part_bottle = yarp.Bottle()
    # for pidx in range(num_joints):
    part_bottle.clear()
    add_part(part_bottle, pose[13, :], 'forehead')  # forehead
    add_part(part_bottle, pose[12, :], 'chin')  # chin

    add_part(part_bottle, (170.0, 220.0, 0.9), 'Rshoulder')  # R shoulder
    add_part(part_bottle, pose[7, :], 'Relbow')  # R elbow
    add_part(part_bottle, pose[6, :], 'Rwrist')  # R wrist

    add_part(part_bottle, (150.0, 220.0, 0.9), 'Lshoulder')  # L shoulder
    add_part(part_bottle, pose[10, :], 'Lelbow')  # L elbow
    add_part(part_bottle, pose[11, :], 'Lwrist')  # L wrist

    add_part(part_bottle, pose[2, :], 'Rhip')  # R hip
    add_part(part_bottle, pose[1, :], 'Rknee')  # R knee
    add_part(part_bottle, pose[0, :], 'Rankle')  # R ankle

    add_part(part_bottle, pose[3, :], 'Lhip')  # L hip
    add_part(part_bottle, pose[4, :], 'Lknee')  # L knee
    add_part(part_bottle, pose[5, :], 'Lankle')  # L ankle

    body_parts.addList().read(part_bottle)

    all_body_parts.addList().read(body_parts)

    ts = yarp.Stamp()
    ts.update()
    port.setEnvelope(ts)
    port.write()


def im_process(sess, cfg, inputs, outputs, image, out_port, fig="preview"):

    image_batch = data_to_input(image)

    # image = image[:, :, (2, 1, 0)]    // Remove comment to get "good" image in opencv
    timer = Timer()
    timer.tic()
    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

    timer.toc()
    print('Detection took {:.3f}s'.format(timer.total_time))

    # Visualise
    # visualize.show_heatmaps(cfg, image, scmap, pose)
    # plt.figure()
    # plt.imshow(visualize.visualize_joints(image, pose))
    CONF_THRES = 0.8

    stream_parts(out_port, pose)
    image = draw_links(image, pose)
    image = visualize.visualize_joints(image, pose, threshold=CONF_THRES)
    if args.cv_show:
        cv2.imshow(fig, image)
    return image


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Pose Estimation demo')
    parser.add_argument('--src', dest='src_port', help='Yarp port of source images',
                        default='/icub/camcalib/left/out')
    parser.add_argument('--des', dest='des_port', help='Yarp port of receiver',
                        default='/leftCam')
    parser.add_argument('--cv', dest='cv_show', help='Show image on opencv fig',
                        default=False)

    args = parser.parse_args()

    return args


class skeleton2DModule(yarp.RFModule):
    """
    Description:
        Object to read yarp image and recognize the skeleton
        
    Args:
        input_port  : input port of image
        output_port : output port for streaming parts
        display_port: output port for image with skeleton
    """
    def __init__(self, input_port_name, output_port_name, display_port_name):
        yarp.RFModule.__init__(self)
        # Prepare ports
        self._input_port = yarp.Port()
        self._input_port_name = input_port_name
        self._input_port.open(self._input_port_name)
        self._output_port = yarp.BufferedPortBottle()
        self._output_port_name = output_port_name
        self._output_port.open(self._output_port_name)
        self._display_port = yarp.Port()
        self._display_port_name = display_port_name
        self._display_port.open(self._display_port_name)
        # Prepare image buffers
        # Input
        self._input_buf_image = yarp.ImageRgb()
        self._input_buf_image.resize(320, 240)
        self._input_buf_array = np.zeros((240, 320, 3), dtype=np.uint8)
        self._input_buf_image.setExternal(self._input_buf_array,
                                          self._input_buf_array.shape[1], self._input_buf_array.shape[0])
        # Output
        self._display_buf_image = yarp.ImageRgb()
        self._display_buf_image.resize(320, 240)
        self._display_buf_array = np.zeros((240, 320, 3), dtype=np.uint8)
        self._display_buf_image.setExternal(self._display_buf_array,
                                            self._display_buf_array.shape[1], self._display_buf_array.shape[0])
        self._logger = yarp.Log()

    def configure(self, rf):
        if args.cv_show:
            cv2.namedWindow(args.des_port)
        if not yarp.Network.connect(args.src_port, args.des_port):
            self._logger.error('Cannot connect to camera port!')

        return True

    def close(self):

        self._input_port.interrupt()
        self._output_port.interrupt()
        self._display_port.interrupt()
        self._input_port.close()
        self._output_port.close()
        self._display_port.close()
        if args.cv_show:
            cv2.destroyAllWindows()
        # yarp.Network.fini()

        return True

    def getPeriod(self):

        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.0

    def updateModule(self):

        # Read an image from the port
        # self._input_buf_array, _ = read_yarp_image(inport=self._input_port)
        self._input_port.read(self._input_buf_image)
        # Process the image
        self._display_buf_array = im_process(sess=sess, cfg=cfg, inputs=inputs,
                                             outputs=outputs, image=self._input_buf_array,
                                             out_port=self._output_port, fig=args.des_port)
        # Send the result to the output port
        write_yarp_image(self._display_port, self._display_buf_array)
        # self._display_port.write(self._display_buf_image)
        if args.cv_show:
            key = cv2.waitKey(20)
            if key == 27:
                self.close()
                return False

        return True


if __name__ == '__main__':
    cfg = load_config("demo/pose_cfg.yaml")
    args = parse_args()

    # Initialise YARP
    yarp.Network.init()

    post2D = skeleton2DModule(args.des_port, '/skeleton2D/bodyParts:o', '/skeleton2D/dispSkeleton:o')
    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('skeleton2D')
    rf.setDefaultContext('skeleton2D.ini')
    rf.configure(sys.argv)

    post2D.runModule(rf)

    #fourcc = cv2.VideoWriter_fourcc(*'x264')  # 'x264' doesn't work
    #out = cv2.VideoWriter('./videos/001_output_pose.avi', fourcc, 30.0, (320, 240))  # 'False' for 1-ch instead of 3-ch for color



