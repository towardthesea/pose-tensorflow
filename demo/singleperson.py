import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
from util.timer import Timer
import matplotlib.pyplot as plt

cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = "demo/image.png"
image = imread(file_name, mode='RGB')

image_batch = data_to_input(image)

timer = Timer()
timer.tic()
# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
timer.toc()
print('Detection took {:.3f}s'.format(timer.total_time))
# Visualise
visualize.show_heatmaps(cfg, image, scmap, pose)
plt.show()
