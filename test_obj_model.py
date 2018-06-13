# Copyright 2017 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imresize, imsave
import argparse
from tensorbox.train_obj_model import build_forward
from tensorbox.utils import googlenet_load
from tensorbox.utils.annolist import AnnotationLib as al
from tensorbox.utils.train_utils import add_rectangles, rescale_boxes
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics
from PIL import Image
from IPython.display import Image as Image2
import time
import scatteract_logger


def main(model_dir, image_dir, true_idl, iteration, iou_threshold, conf_threshold):

    hypes_file = '{}/hypes.json'.format(model_dir)

    with open(hypes_file, 'r') as f:
        H = json.load(f)

    model_name = model_dir.split("/")[1]
    pred_idl = './output/%s_%d_val_%s.idl' % (model_name, iteration, os.path.basename(hypes_file).replace('.json', ''))
    true_annos = al.parse(true_idl)

    tf.reset_default_graph()
    googlenet = googlenet_load.init(H)
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])

    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, '{}/save.ckpt-{}'.format(model_dir,iteration))

        annolist = al.AnnoList()
        t = time.time()
        for i in range(len(true_annos)):
            true_anno = true_annos[i]

            img = Image.open(image_dir + "/" + true_anno.imageName)
            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img,img)
            img = np.array(bg)
            img_orig = np.copy(img)

            if img.shape[0] != H["image_height"] or img.shape[1] != H["image_width"]:
                true_anno = rescale_boxes(img.shape, true_anno, H["image_height"], H["image_width"])
                img = imresize(img, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            pred_anno = al.Annotation()
            pred_anno.imageName = true_anno.imageName
            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=H['rnn_len'], min_conf=conf_threshold)
            new_img_true = np.copy(img)
            new_img_pred = np.copy(img)

            for rect_true in true_anno.rects:
                cv2.rectangle(new_img_true,(int(rect_true.x1),int(rect_true.y1)),
                              (int(rect_true.x2),int(rect_true.y2)),
                              (0,255,0),2)

            for rect_pred in rects:
                cv2.rectangle(new_img_pred,(int(rect_pred.x1),int(rect_pred.y1)),
                              (int(rect_pred.x2),int(rect_pred.y2)),
                              (0,0,255),2)

            pred_anno.rects = rects
            pred_anno = rescale_boxes(img.shape, pred_anno, img_orig.shape[0], img_orig.shape[1])
            annolist.append(pred_anno)

            if i % 10 == 0 and i < 400:
                imsave("{}/".format(model_dir)+pred_anno.imageName.split('/')[-1][:-4]+'_pred.bmp',new_img_pred)
                imsave("{}/".format(model_dir)+pred_anno.imageName.split('/')[-1][:-4]+'_true.bmp',new_img_true)
            if (i+1) % 200 == 0 or i==0:
                mylogger.debug("Number of images analyzed: {}".format(i+1))
        avg_time = (time.time() - t) / (i + 1)
        mylogger.debug('%f images/sec' % (1. / avg_time))


    annolist.save(pred_idl)
    rpc_cmd = './tensorbox/utils/annolist/doRPC.py  --minOverlap %f %s %s' % (iou_threshold, true_idl, pred_idl)
    mylogger.debug('$ %s' % rpc_cmd)
    rpc_output = subprocess.check_output(rpc_cmd, shell=True)
    mylogger.debug(rpc_output)
    txt_file = [line for line in rpc_output.split('\n') if line.strip()][-1]
    output_png = 'output/{}_{}_results.png'.format(model_name, iteration)
    plot_cmd = './tensorbox/utils/annolist/plotSimple.py %s --output %s' % (txt_file, output_png)
    mylogger.debug('$ %s' % plot_cmd)
    plot_output = subprocess.check_output(plot_cmd, shell=True)
    Image2(filename=output_png)

    df = pd.read_csv("output/rpc-{}_{}_val_hypes_overlap{}.txt".format(model_name,iteration,iou_threshold), sep=" ", names=['precision','recall','fpii','score','accuracy'])
    auc = metrics.auc(df['recall'],df['precision'])
    mylogger.info("Average Precision: {}".format(auc))


if __name__ == "__main__":

    """
    Example of command line usage:

    python test_obj_model.py --model_dir output/lstm_rezoom_plot_labels_2017_04_11_01.14 --image_dir data/plot_test --true_idl data/plot_test/labels.idl --iteration 125000
    """

    mylogger = scatteract_logger.get_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', help='Model directory (string)', required=True)
    parser.add_argument('--image_dir', help='Image directory (string)', required=True)
    parser.add_argument('--true_idl', help='True idl file (string)', required=True)
    parser.add_argument('--iteration', help='Iteration number (int)', required=True)
    parser.add_argument('--iou_threshold', help='IOU threshold (float)', default=0.5, required=False)
    parser.add_argument('--conf_threshold', help='Confidence threshold (float)', default=0.3, required=False)
    args = vars(parser.parse_args())

    main(args['model_dir'], args['image_dir'], args['true_idl'], int(args['iteration']), float(args['iou_threshold']), float(args['conf_threshold']))
