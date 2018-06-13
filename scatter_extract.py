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
import json
from scipy.misc import imresize, imsave
from tensorbox.train_obj_model import build_forward
from tensorbox.utils import googlenet_load
from tensorbox.utils.annolist import AnnotationLib as al
from tensorbox.utils.train_utils import add_rectangles, rescale_boxes
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import tesseract
from sklearn.linear_model import RANSACRegressor
import pandas as pd
import time
import argparse
import os
import scatteract_logger


def parse_coords(idl_file_name):
    """
    Function that reads the coordinate ild file and returns a dictionary of pandas dataframes.
    Inputs:
    idl_file_dir: (string) File path.
    Outputs:
    df_dict (dictionary): Dictionary of dataframes, key is the plot name, values is the pandas dataframe.
    """

    df_dict = {}
    with open(idl_file_name, 'r') as myfile:
        for line in myfile:
            file_name = line.split(':')[0]
            file_name = file_name[1:len(file_name)-1]

            df = pd.DataFrame(columns = ['X','Y'])
            coords = line.split(':')[1].split('),')
            coords = [coord.replace('(','').replace(')','').replace(';','').replace(' ','') for coord in coords]
            for j in range(len(coords)):
                df.loc[j]= coords[j].split(',')
            df = df.astype(float)

            df_dict[file_name] = df

    return df_dict


class PlotExtractor(object):
    """
    Object used to extract point coordinates from scatter plots.
    """

    def __init__(self, my_dir_dict, iteration):
        """
        Constructor
        Inputs:
        my_dir_dict (dictionary): dictionary of the directory for the three models (ticks, points and labels) required.
        iterarion: (int) Iteration number of the saved models to use.

        For example:
        my_dir_dict = {'ticks':"./output/lstm_rezoom_plot_ticks_2017_03_23_22.56",
               'labels':"./output/lstm_rezoom_plot_labels_2017_03_25_09.51",
               'points':"./output/lstm_rezoom_plot_points_2017_03_25_19.55"}
        iteration = 100000
        """

        self.my_dir_dict = my_dir_dict
        self.iteration = iteration
        self.hypes_file_dict = {key:'{}/hypes.json'.format(my_dir_dict[key]) for key in my_dir_dict}


        self.H={}
        for key in self.hypes_file_dict:
            with open(self.hypes_file_dict[key], 'r') as f:
                self.H[key] = json.load(f)


        self.models = {}
        for key in self.H:
            graph, x_in, pred_boxes, pred_logits, pred_confidences, saver = self.init_model(self.H[key])
            self.models[key] = {'graph': graph, 'x_in':x_in, 'pred_boxes':pred_boxes, 'pred_logits':pred_logits,
                                 'pred_confidences':pred_confidences,'saver':saver, 'dir': self.my_dir_dict[key]}


        color_list = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (0,128,0), (0,0,128)]
        self.color_dict = {}
        index = 0
        for key in self.H:
            self.color_dict[key] = color_list[index]
            index+=1


    def init_model(self, H):
        """
        Helper function uses to initialize the saved models.
        Inputs:
        H (dictionary): Dictionary of the loaded json hype files for each models.
        Outputs:
        graph (tensorflow graph) : Graph for the current model.
        x_in (tensorflow op) : Input image for the current model.
        pred_boxes (tensorflow op) : Output bouding box prediction.
        pred_logits(tensorflow op) : Output logits for the prediction.
        pred_confidences (tensorflow op) :  Output confidence score for the prediction.
        saver (tensorflow saver) : Saver which allowed to load pretrain weights into the current model.
        """

        graph = tf.Graph()
        with graph.as_default():
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

        return graph, x_in, pred_boxes, pred_logits, pred_confidences, saver


    def open_image(self, image_name):
        """
        Helper function which opens an image and converts it to RGB.
        Inputs:
        image_name (string): image path
        Outputs:
        img (numpy array):  Numpy matrix of the image.
        """

        img = Image.open(image_name)
        if np.shape(img)[-1]==4:
            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img,img)
            img = np.array(bg)
        elif len(np.shape(img))==2:
            img = np.array(img.convert('RGB'))
        else:
            img = np.array(img)
        return img


    def test(self,image_dir, image_output_dir, csv_output_dir , true_idl_dict = None, coord_idl = None, predict_idl = None,
             quick = False, conf_threshold = 0.3, max_dist_perc = 2.0):
        """
        Method used to test the object on a set of scatter plots.
        Inputs:
        true_idl_dict (dictionary): Dictionary where the key is the object name (ticks, points, labels) and
        the values are the path to their corresponding idl file which may or may not contain the ground truth.
        image_dir: (string): Path to the location of the images
        image_output_dir (string): Path to the location where we want to save new images with marked bounding boxes.
        csv_output_dir (string): Path to the location where we want to save the output csv for each images.
        coord_idl: (string) Optional, ground truth idl for the real coordinate values.
        Outputs:
        pred_dict: (dictionary) Dictionary of dataframes which contains the predicted coordinate values.
        """

        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
        if not os.path.exists(csv_output_dir):
            os.makedirs(csv_output_dir)

        if true_idl_dict is not None:
            true_annos_dict = {key:al.parse(true_idl_dict[key]) for key in true_idl_dict}
        else:
            true_annos_dict =  {key:al.parse(predict_idl) for key in self.my_dir_dict}

        start_time = time.time()

        pred_dict = {}
        for key in self.models:
            pred_dict[key] = self.predict_model(self.models[key], self.H[key], true_annos_dict[key], image_dir, conf_threshold)
            pred_dict[key].save('{}/model_pred_{}.idl'.format(self.models[key]['dir'],self.iteration))

        self.draw_images(image_dir, true_annos_dict, pred_dict, image_output_dir, true_idl_dict)

        pred_dict['labels'] = self.get_ocr(pred_dict['labels'], image_dir)

        pred_dict['labels'] = self.get_closest_ticks(pred_dict['labels'], pred_dict['ticks'])

        pred_labels_X, pred_labels_Y = self.split_labels_XY(pred_dict['labels'])

        self.regressor_x_dict = self.get_conversion(pred_labels_X, cat = 'x')
        self.regressor_y_dict = self.get_conversion(pred_labels_Y, cat = 'y')

        df_dict = self.predict_points(pred_dict['points'],self.regressor_x_dict, self.regressor_y_dict, csv_output_dir)

        mylogger.debug("{} images/sec".format(float(len(pred_dict['labels']))/(time.time()-start_time)))

        if coord_idl is not None:
            mylogger.debug("Now using ground truth to test ...")
            self.get_metrics(df_dict, coord_idl, csv_output_dir, quick = quick, max_dist_perc = max_dist_perc)

        return pred_dict


    def predict_model(self,model, H, true_annos, image_dir, conf_threshold):
        """
        Method uses to get the prediction from the object detection models.
        Inputs:
        model (dictionary) : Dictionary which contains all the Tensorflow object required to run the model
        H (dictionary): Loaded json hype which describes the hyperparameters of the models.
        true_annos (Annotationlist): List of annotations which contain the ground truth bounding boxes.
        image_dir: (string): Path to the location of the images
        Outputs:
        annolist (Annotationlist): List of annotations which contain the predicted bounding boxes.
        """

        annolist = al.AnnoList()
        with tf.Session(graph = model['graph']) as sess:
            sess.run(tf.initialize_all_variables())
            model['saver'].restore(sess, '{}/save.ckpt-{}'.format(model['dir'],self.iteration))

            for i in range(len(true_annos)):
                true_anno = true_annos[i]

                img = self.open_image(image_dir+true_anno.imageName)
                img_orig = np.copy(img)

                if img.shape[0] != H["image_height"] or img.shape[1] != H["image_width"]:
                    img = imresize(img, (H["image_height"], H["image_width"]), interp='cubic')

                (np_pred_boxes, np_pred_confidences) = sess.run([model['pred_boxes'], model['pred_confidences']],
                                                                feed_dict={model['x_in']: img})

                pred_anno = al.Annotation()
                pred_anno.imageName = true_anno.imageName
                new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                                use_stitching=True, rnn_len=H['rnn_len'], min_conf=conf_threshold)

                pred_anno.rects = rects
                pred_anno = rescale_boxes(img.shape, pred_anno, img_orig.shape[0], img_orig.shape[1])
                annolist.append(pred_anno)

        return annolist


    def draw_images(self, image_dir, true_annos_dict, pred_dict, image_output_dir, true_idl_dict):
        """
        Helper function uses to draw the images with marked bouding boxes.
        Inputs:
        image_dir (string): Path of the image directory.
        true_annos_dict (dictionary): Dictionary of the loaded annotation files for the bounding boxes of each object (ticks, points, labels)
        pred_dict: (dictionary): Same as the true_annos_dict but for the predictions.
        image_output_dir: (string) Path to the location where we want to save new images with marked bounding boxes.
        true_idl_dict (dict):  If not None, this means we are in test mode, so also save ground truth images.
        """

        for j in range(len(true_annos_dict[true_annos_dict.keys()[0]])):

            img = self.open_image(image_dir+true_annos_dict[true_annos_dict.keys()[0]][j].imageName)

            new_img_true = np.copy(img)
            new_img_pred = np.copy(img)

            for key in true_annos_dict:

                if true_idl_dict is not None:
                    for rect_true in true_annos_dict[key][j].rects:
                        cv2.rectangle(new_img_true,(int(rect_true.x1),int(rect_true.y1)),
                                      (int(rect_true.x2),int(rect_true.y2)),
                                      self.color_dict[key],2)

                for rect_pred in pred_dict[key][j].rects:
                    cv2.rectangle(new_img_pred,(int(rect_pred.x1),int(rect_pred.y1)),
                                  (int(rect_pred.x2),int(rect_pred.y2)),
                                   self.color_dict[key],2)

            imsave("./{}/".format(image_output_dir)+true_annos_dict['ticks'][j].imageName.split('/')[-1][:-4]+'_pred.bmp',new_img_pred)
            if true_idl_dict is not None:
                imsave("./{}/".format(image_output_dir)+true_annos_dict['ticks'][j].imageName.split('/')[-1][:-4]+'_true.bmp',new_img_true)


    def get_ocr(self,pred_labels, image_dir, pixel_extra=1):
        """
        Method apply OCR on the detected label.
        Inputs:
        pred_labels: (Annotationlist) List of annotations which contains the bounding box for each labels.
        image_dir (string): path to the directory which contains the images.
        pixel_extra: (int) Pixel to add to the bounding boxes, in case they are too tight.
        Outputs:
        pred_labels: (Annotationlist) List of annotation which have the bounding boxes of the labels, and their corresponding values after OCR.
        """

        for j in range(len(pred_labels)):

            img = self.open_image(image_dir+pred_labels[j].imageName)

            new_rects = []
            for rect in pred_labels[j].rects:
                image = Image.fromarray(np.copy(img[max(0,int(rect.y1)-pixel_extra):int(rect.y2)+pixel_extra, max(0,int(rect.x1)-pixel_extra):int(rect.x2)+pixel_extra,:]))
                label = tesseract.get_label(image)
                if label is not None:
                    rect.classID = label
                    new_rects.append(rect)

            pred_labels[j].rects = new_rects

        return pred_labels


    def get_closest_ticks(self, pred_labels, pred_ticks):
        """
        Method that matches ticks with their closest labels
        Inputs:
        pred_labels (Annotationlist): List of annotations (bounding boxes) for the predicted labels
        pred_ticks (Annotationlist): List of annotations (bounding boxes) for the predicted ticks
        Outputs
        annolist (Annotationlist): List of annotations which has the bounding boxes of the ticks, but the label values of the labels.
        """

        annolist = al.AnnoList()

        for j in range(len(pred_labels)):

            new_annot = al.Annotation()
            new_annot.imageName = pred_labels[j].imageName
            if len(pred_labels[j].rects)>0 and len(pred_ticks[j].rects)>0:

                distances = np.zeros((len(pred_labels[j].rects),len(pred_ticks[j].rects)))

                for k in range(len(pred_labels[j].rects)):
                    for i in range(len(pred_ticks[j].rects)):
                        distances[k,i] = pred_labels[j].rects[k].distance(pred_ticks[j].rects[i])

                min_index_labels = np.argmin(distances, axis=1)
                min_index_ticks = np.argmin(distances, axis=0)

                new_rects = []
                for k in range(len(pred_labels[j].rects)):
                    min_index = min_index_labels[k]
                    if min_index_ticks[min_index] == k:
                        temp_rect = pred_ticks[j].rects[min_index]
                        temp_rect.classID = pred_labels[j].rects[k].classID
                        new_rects.append(temp_rect)

                new_annot.rects = new_rects
                annolist.append(new_annot)

            else:
                new_annot.rects = []
                annolist.append(new_annot)


        return annolist


    def split_labels_XY(self,pred_labels, eps_std_div = 25.0):
        """
        Method which splits labels into X labels, and Y labels.
        Inputs:
        pred_labels: (Annotationlist)  List of annotations of the predicted labels.
        eps_std_div (float): Parameters which controls how far appart the bounding boxes are allowed to be on an axis.
        Outputs:
        pred_labels_X, pred_labels_Y (Annotationlist, Annotationlist): Annotation lists for the predicted labels on the X and Y axis.
        """

        pred_labels_X, pred_labels_Y  = al.AnnoList(), al.AnnoList()

        for j in range(len(pred_labels)):

            pred_X, pred_Y = al.Annotation(), al.Annotation()
            pred_X.imageName, pred_Y.imageName = pred_labels[j].imageName, pred_labels[j].imageName

            X, Y = [], []
            for rect in pred_labels[j].rects:
                X.append((rect.x2 + rect.x1)/2.0)
                Y.append((rect.y2 + rect.y1)/2.0)
            X, Y = np.reshape(np.array(X),(len(X),1)), np.reshape(np.array(Y),(len(Y),1))
            std_x = np.std(X)
            std_y = np.std(Y)

            if std_x>0 and std_y>0:
                db_scan_x = DBSCAN(eps=std_y/eps_std_div, min_samples = 2).fit(Y)
                db_scan_y = DBSCAN(eps=std_x/eps_std_div, min_samples = 2).fit(X)
                cluster_list_x = filter(lambda x: x!=-1, db_scan_x.labels_)
                cluster_list_y = filter(lambda x: x!=-1, db_scan_y.labels_)
                if len(cluster_list_x)>0 and len(cluster_list_y)>0:
                    cluster_label_x = max(set(cluster_list_x), key=cluster_list_x.count)
                    cluster_label_y = max(set(cluster_list_y), key=cluster_list_y.count)
                    rect_x, rect_y = [], []
                    for k in range(len(db_scan_x.labels_)):
                        if db_scan_x.labels_[k]==cluster_label_x and db_scan_y.labels_[k]!=cluster_label_y:
                            rect_x.append(pred_labels[j].rects[k])
                        elif db_scan_y.labels_[k]==cluster_label_y and db_scan_x.labels_[k]!=cluster_label_x:
                            rect_y.append(pred_labels[j].rects[k])
                    pred_X.rects, pred_Y.rects = rect_x, rect_y

            pred_labels_X.append(pred_X), pred_labels_Y.append(pred_Y)

        return pred_labels_X, pred_labels_Y


    def get_conversion(self, pred_labels, cat):
        """
        Method that uses ransac regression to find conversion rules between label coordinate and labels coordinates.
        Inputs:
        pred_labels: (Annotationlist)  List of annotations which has the bounding boxes of the ticks, but the label values of the labels.
        cat:  (string)  Axis type, either 'x' or 'y'.
        Outputs:
        regressors:  (dictionary): Dictionary in which the key is the image name, and the value is a regressor which converts to label coordinates.
        """

        regressors = {}
        for j in range(len(pred_labels)):

            positions = []
            labels = []
            if len(pred_labels[j].rects)>=1:
                for k in range(len(pred_labels[j].rects)):

                    if cat=='x':
                        positions.append((pred_labels[j].rects[k].x1 + pred_labels[j].rects[k].x2)/2.0)
                        labels.append(pred_labels[j].rects[k].classID)
                    elif cat=='y':
                        positions.append((pred_labels[j].rects[k].y1 + pred_labels[j].rects[k].y2)/2.0)
                        labels.append(pred_labels[j].rects[k].classID)

            if len(positions)>1:
                try:
                    ransac_threshold = np.median(np.abs(np.array(labels) - np.median(labels)))**2/50.0
                    reg = RANSACRegressor(random_state=0, loss = lambda x,y: np.sum((np.abs(x-y))**2,axis=1), residual_threshold = ransac_threshold)
                    reg = reg.fit(np.reshape(positions,(len(positions),1)),np.reshape(labels,(len(labels),1)))
                except ValueError:
                    reg = None
            else:
                reg = None
            regressors[pred_labels[j].imageName] = reg

        return regressors


    def predict_points(self, pred_points, regressor_x_dict, regressor_y_dict, csv_output_dir = None):
        """
        Method which uses the ransac regressor to predict label coordinates values for the points detected.
        Inputs:
        pred_points: (Annotationlist) List of annotations which contain the bounding boxes of the points detected.
        regressor_x_dict (dictionary): Dictionary of regressors for the X axis.
        regressor_y_dict (dictionary): Dictionary of regressors for the Y axis.
        csv_output_dir: (string) Output directory for the csv files which contain the final predictions.
        Outputs:
        df_dict (dictionary): Dictionnary of dataframes that contains the final predictions.
        """

        df_dict = {}

        for j in range(len(pred_points)):

            reg_x = regressor_x_dict[pred_points[j].imageName]
            reg_y = regressor_y_dict[pred_points[j].imageName]

            X_pixel, Y_pixel = [], []
            for rect in  pred_points[j].rects:
                X_pixel.append((rect.x1+rect.x2)/2.0)
                Y_pixel.append((rect.y1+rect.y2)/2.0)

            if reg_x is not None and reg_y is not None and len(X_pixel)>0:
                X_pixel = np.reshape(X_pixel,(len(X_pixel),1))
                Y_pixel = np.reshape(Y_pixel,(len(Y_pixel),1))
                X_label = list(np.ravel(reg_x.predict(X_pixel)))
                Y_label = list(np.ravel(reg_y.predict(Y_pixel)))
            else:
                X_label = []
                Y_label = []

            df = pd.DataFrame({'X':X_label,'Y':Y_label})
            df_dict[pred_points[j].imageName] = df
            if csv_output_dir is not None:
                df.to_csv(csv_output_dir + "/{}".format(pred_points[j].imageName.split('/')[-1][:-4]+'_pred.csv'), index=False)

        return df_dict


    def is_good(self, element, df_element, max_dist_perc, norm):
        """
        Helper method which says if a predicted points is close enough to a "true" point.
        Inputs:
        element (list) : Label coordinate of a predicted point [X, Y]
        df_element (Pandas dataframe): Dataframe of the ground truth
        max_dist_perc (float): Maximum tolerated percentage difference between prediction and ground truth.
        norm (float): Normalization constant which corresponds to the range of possible values from the ground truth.
        Outputs:
        (Boolean, Int):  Boolean which corresponds to weather or not there is a true element close enough,
        and an Int which points to which ground truth element is the closest.
        """

        if len(df_element)==0:
            return False, 0
        else:
            distances  = np.abs(element - df_element)
            distances_normed = distances / norm
            return ((distances_normed <= max_dist_perc/100.0).sum(axis=1)==2).sum()>=1, distances_normed.sum(axis=1).argmin()


    def get_precision_recall(self, pred, true, max_dist_perc, norm, quick = False):
        """
        Helper method which computes the precsion and recall on a particular plot.
        Inputs:
        pred: (pandas dataframe): Dataframe which contains the predictions.
        true: (pandas dataframe): Dataframe which contains the ground truth.
        max_dist_perc (float): Maximum tolerated percentage difference between prediction and ground truth.
        norm (float): Normalization constant which corresponds to the range of possible values from the ground truth.
        quick (boolean): Boolean to decide weather to approximate the precision recall quantities.
        Outputs:
        precision, recall (float, float): Precision and recall value for this particular plot.
        """

        if len(true)==0 and len(pred)==0:
            precision, recall = 1.0, 1.0
        elif (len(true)==0 and len(pred)>0) or (len(true)>0 and len(pred)==0):
            precision, recall = 0.0, 0.0
        else:
            true_positive = []
            true_copy = true.copy()
            pred_copy = pred.copy()
            keep_going_flag = True

            if not quick:
                while keep_going_flag:
                    
                    pred_copy.index = range(len(pred_copy))
                    true_copy.index = range(len(true_copy))

                    D = np.zeros((len(pred_copy),len(true_copy)))
                    D_x = np.zeros((len(pred_copy),len(true_copy)))
                    D_y = np.zeros((len(pred_copy),len(true_copy)))
                    for j in range(len(pred_copy)):
                        for k in range(len(true_copy)):
                            dist = np.abs(pred_copy.iloc[j]-true_copy.iloc[k])
                            dist = dist/norm
                            D_x[j,k] = dist.values[0]
                            D_y[j,k] = dist.values[1]
                            D[j,k] = dist.sum()

                    min_index_pred = np.argmin(D, axis=1)
                    min_index_true = np.argmin(D, axis=0)

                    new_true_positives = 0
                    true_copy_new = true_copy.copy()
                    pred_copy_new = pred_copy.copy()
                    for k in range(len(pred_copy)):
                        closest_true_index = min_index_pred[k]
                        if min_index_true[closest_true_index] == k:
                            if D_x[k,closest_true_index]<= max_dist_perc/100.0 and D_y[k,closest_true_index]<= max_dist_perc/100.0:
                                true_positive.append(pred_copy.iloc[k])
                                new_true_positives+=1
                                true_copy_new = true_copy_new[true_copy_new.index!=closest_true_index]
                                pred_copy_new = pred_copy_new[pred_copy_new.index!=k]

                    if len(pred_copy_new)==0 or len(true_copy_new)==0 or new_true_positives==0:
                        keep_going_flag = False

                    true_copy = true_copy_new
                    pred_copy = pred_copy_new
                    
            else:
                for j in range(len(pred)):
                    bool_good, pos_good = self.is_good(pred.iloc[j],true_copy,max_dist_perc, norm)
                    if bool_good:
                        true_positive.append(pred.iloc[j])
                        true_copy = true_copy[true_copy.index!=pos_good]

            precision = len(true_positive)/float(len(pred))
            recall = len(true_positive)/float(len(true))

        return precision, recall


    def get_metrics(self, df_dict_pred, coord_idl, csv_output_dir = None, max_dist_perc = 2.0, quick = False):
        """
        Method which computes the precision and recall for all the plots.
        Inputs:
        df_dict_pred (dictionary): Dictionnary of dataframes wich contains the predicted points in label coordinates.
        coord_idl (string): Path to the idl files which contain the ground truth coordinates
        max_dist_perc (float): Maximum tolerated percentage difference between prediction and ground truth.
        quick (boolean): Boolean to decide weather to approximate the precision recall quantities.

        Outputs:
        df_prec_recall : (pandas dataframe) Pandas dataframe of precisions and recalls for each plot.
        """

        df_dict_true = parse_coords(coord_idl)
        precision_list, recall_list, image_name_list = [], [], []
        count_good = 0
        count_perfect = 0
        count_bad = 0
        metrics_logger = scatteract_logger.get_logger()

        for file_name in df_dict_pred:

            df_pred = df_dict_pred[file_name]
            df_true = df_dict_true.get(file_name,None)

            if df_true is not None:
                prec, rec = self.get_precision_recall(df_pred,df_true,
                                                      max_dist_perc = max_dist_perc,
                                                      norm = (df_true.max(axis=0)-df_true.min(axis=0)), quick = quick)
                precision_list.append(prec)
                recall_list.append(rec)
                image_name_list.append(file_name)
                if rec>=0.8 and prec>=0.8:
                    count_good+=1
                if rec==1.0 and prec==1.0:
                    count_perfect+=1
                if rec<=0.1 and prec<=0.1:
                    count_bad+=1
            else:
                metrics_logger.warn("No ground truth for :" + file_name)

        metrics_logger.info("Percentage of good extraction (recall and precision above 80%): {}".format(float(count_good)/len(precision_list)))
        metrics_logger.info("Percentage of perfect extraction (recall and precision at 100%): {}".format(float(count_perfect)/len(precision_list)))
        metrics_logger.info("Percentage of bad extraction (recall and precision below 10%): {}".format(float(count_bad)/len(precision_list)))
        metrics_logger.info("Precision: {}".format(np.mean(precision_list)))
        metrics_logger.info("Recall: {}".format(np.mean(recall_list)))
        metrics_logger.info("F1 score: {}".format(2*np.mean(recall_list)*np.mean(precision_list)/(np.mean(recall_list)+np.mean(precision_list))))

        df_prec_recall = pd.DataFrame({"image_name":image_name_list,"recall":recall_list,"precision":precision_list})
        if csv_output_dir is not None:
            df_prec_recall.to_csv(csv_output_dir + "/" + "precision_recall_list.csv")
            metrics_logger.debug("Saving a csv of precisions and recalls : {}".format(csv_output_dir + "/" + "precision_recall_list.csv"))

        return df_prec_recall


if __name__ == "__main__":

    """
    Example of command-line usage:

    TEST: (requires a dict of ground truth for ticks, labels and points bounding boxes, and an idl file for the ground truth coordinate)

    python scatter_extract.py --model_dict '{"ticks":"./output/lstm_rezoom_plot_ticks_2017_04_11_19.52", "labels":"./output/lstm_rezoom_plot_labels_2017_04_11_01.14","points":"./output/lstm_rezoom_plot_points_2017_04_14_15.58"}' \
    --iteration 125000 --image_dir data/plot_test/ --true_idl_dict '{"ticks":"./data/plot_test/ticks.idl","labels":"./data/plot_test/labels.idl", "points":"./data/plot_test/points.idl"}' \
    --image_output_dir image_output --csv_output_dir csv_output --true_coord_idl ./data/plot_test/coords.idl


    PREDICT: (requires an idl file which has a list of images to test on)

    python scatter_extract.py --model_dict '{"ticks":"./output/lstm_rezoom_plot_ticks_2017_04_11_19.52", "labels":"./output/lstm_rezoom_plot_labels_2017_04_11_01.14","points":"./output/lstm_rezoom_plot_points_2017_04_14_15.58"}' \
    --iteration 125000 --image_dir data/plots_real/ --predict_idl ./data/plots_real/test_real.idl --image_output_dir image_output --csv_output_dir csv_output
    """

    mylogger = scatteract_logger.get_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dict', help='Directory for the object detection models', required=True, type=json.loads)
    parser.add_argument('--iteration', help='Iteration number for the trained models', required=True)
    parser.add_argument('--image_dir', help='Directory of the images', required=True)
    parser.add_argument('--true_idl_dict', help='Path of the ground truth idls', required=False, type=json.loads, default=None)
    parser.add_argument('--predict_idl', help='Path of an idl file which list the images to predict on', required=False, default=None)
    parser.add_argument('--image_output_dir', help='Directory to output images with bounding boxes', required=True)
    parser.add_argument('--csv_output_dir', help='Directory to output csv of results', required=True)
    parser.add_argument('--true_coord_idl', help='Idl of the ground truth coordinates', required=False, default=None)
    parser.add_argument('--conf_threshold', help='Confidence threshold', required=False, default=0.3)
    parser.add_argument('--max_dist_perc', help='Maximum percent distance to be considered true positive', required=False, default=2.0)
    args = vars(parser.parse_args())


    plt_xtr = PlotExtractor(args["model_dict"], int(args["iteration"]))

    plt_xtr.test(args["image_dir"], args["image_output_dir"], args["csv_output_dir"],true_idl_dict=args["true_idl_dict"],
                 coord_idl=args["true_coord_idl"], predict_idl = args["predict_idl"], quick=False, conf_threshold = float(args['conf_threshold']), max_dist_perc = float(args['max_dist_perc']))
