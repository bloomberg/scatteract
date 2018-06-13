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


import matplotlib
matplotlib.use('Agg') #To make sure plots are not being displayed during the generation.
import numpy as np
np.random.seed(0) #For consistent data generation.
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm
from matplotlib import font_manager
import random
import string
import os
import argparse
import scatteract_logger


### SETTING CONSTANT PARAMTERS #####

markers = np.array([".",",","o","v","^","<",">","1","2","3","4","8","s","p",
           "*","h","H","+","x","D","d","|","_"])
markers_with_full = [markers[j] for j in [0,1,2,3,4,5,6,11,12,13,14,15,16,19,20]]

linestyles = np.array(['solid', 'dashed', 'dashdot', 'dotted', '-', '--' ,'-.', ':'])

color_grid = ['b', 'g', 'r', 'k', '0.6', '0.8']
color_subtick_list = ['b','g','r','k', 'k', 'k', '0.7', '0.85']

direction_ticks = ['in','out','inout']

font_list = ['fonts/' + name for name in os.listdir('fonts')]
 

dpi_min = 85
dpi_max = 250

figsize_min = 3
figsize_max = 10

tick_size_width_min = 0
tick_size_width_max = 3
tick_size_length_min = 0
tick_size_length_max = 12

points_nb_min = 10
points_nb_max = 130
x_min_top = -2
x_max_top = 5
y_min_top = -2
y_max_top = 4
x_scale_range_max = 4
y_scale_range_max = 4

size_points_min = 3
size_points_max = 12

max_points_variations = 5

pad_min = 2
pad_max = 18

axes_label_size_min = 10
axes_label_size_max = 16
tick_label_size_min = 10
tick_label_size_max = 16
title_size_min = 14
title_size_max = 24

axes_label_length_min = 5
axes_label_length_max = 15
title_length_min = 5
title_length_max = 25

colorbg_transparant_max = 0.05

styles = plt.style.available
if 'dark_background' in styles:
    styles.remove('dark_background')

point_dist = ['uniform', 'linear', 'quadratic']


def cat_in_dict(cat,cat_dict):

    for key, cat_i in cat_dict.items():

        if cat[0]==cat_i[0] and np.all(cat[1]==cat_i[1]) and cat[2]==cat_i[2] and cat[3]==cat_i[3]:
            return key

    return False


def get_random_plot(name, direc):
    """
    Random plot generation method.
    Inputs:
    name: (string) name of the plot which will be saved.
    Outputs:
    ax : (matplotlib obj) Matplotlib object of the axes of the plot
    fig : (matplotlib obj) Matplotlib object of the figure of the plot
    x, y : (list, list) Actuall x and y coordinates of the points.
    s : (list) sizes of the points.
    categories : (list) categories of the points.
    tick_size : (list) Tick size on the plot. [width, length]
    axes_x_pos, axes_y_pos: (float, float) Position of the labels of the axis.
    """

    # PLOT STYLE
    style = random.choice(styles)
    plt.style.use(style)

    # POINT DISTRIBUTION
    distribution = random.choice(point_dist)

    # RESOLUTION AND TICK SIZE
    dpi = int(dpi_min + np.random.rand(1)[0]*(dpi_max-dpi_min))
    figsize = (figsize_min+np.random.rand(2)*(figsize_max-figsize_min)).astype(int)
    tick_size = [(tick_size_width_min+np.random.rand(1)[0]*(tick_size_width_max-tick_size_width_min)),
                 (tick_size_length_min+np.random.rand(1)[0]*(tick_size_length_max-tick_size_length_min))]
    tick_size.sort()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ACTUAL POINTS
    points_nb = int(points_nb_min + (np.random.rand(1)[0]**1.5)*(points_nb_max-points_nb_min))
    x_scale =  int(x_min_top+np.random.rand(1)[0]*(x_max_top - x_min_top))
    y_scale =  int(y_min_top+np.random.rand(1)[0]*(y_max_top - y_min_top))
    x_scale_range = x_scale + int(np.random.rand(1)[0]*x_scale_range_max)
    y_scale_range = y_scale + int(np.random.rand(1)[0]*y_scale_range_max)
    x_min = (-np.random.rand(1)[0]+np.random.rand(1)[0])*10**(x_scale)
    x_max = (-np.random.rand(1)[0]+np.random.rand(1)[0])*10**(x_scale_range)
    x_min, x_max = min(x_min,x_max), max(x_min,x_max)
    y_min = (-np.random.rand(1)[0]+np.random.rand(1)[0])*10**(y_scale)
    y_max = (-np.random.rand(1)[0]+np.random.rand(1)[0])*10**(y_scale_range)
    y_min, y_max = min(y_min,y_max), max(y_min,y_max)

    if distribution=='uniform':
        x = x_min+np.random.rand(points_nb)*(x_max-x_min)
        y = y_min+np.random.rand(points_nb)*(y_max-y_min)
    elif distribution=='linear':
        x = x_min+np.random.rand(points_nb)*(x_max-x_min)
        y = x*(max(y_max,-y_min)/(max(x_max,-x_min)))*random.choice([-1.0,1.0]) + (y_min+np.random.rand(points_nb)*(y_max-y_min))*np.random.rand(1)[0]/2.0
    elif distribution=='quadratic':
        x = x_min+np.random.rand(points_nb)*(x_max-x_min)
        y = x**2*(1.0/(max(x_max,-x_min)))**2*max(y_max,-y_min)*random.choice([-1.0,1.0]) + (y_min+np.random.rand(points_nb)*(y_max-y_min))*np.random.rand(1)[0]/2.0

    # POINTS VARIATION
    nb_points_var = 1+int(np.random.rand(1)[0]*max_points_variations)
    nb_points_var_colors =  1+int(np.random.rand(1)[0]*nb_points_var)
    nb_points_var_markers =  1+int(np.random.rand(1)[0]*(nb_points_var-nb_points_var_colors))
    nb_points_var_size =  max(1,1+nb_points_var-nb_points_var_colors-nb_points_var_markers)

    rand_color_number = np.random.rand(1)[0]
    if rand_color_number<=0.5:
        colors = cm.rainbow(np.random.rand(nb_points_var_colors))
    elif rand_color_number>0.5 and rand_color_number<=0.7:
        colors = cm.gnuplot(np.random.rand(nb_points_var_colors))
    elif rand_color_number>0.7 and rand_color_number<=0.8:
        colors = cm.copper(np.random.rand(nb_points_var_colors))
    else:
        colors = cm.gray(np.linspace(0,0.6,nb_points_var_colors))
    s_set = (size_points_min+np.random.rand(nb_points_var_size)*(size_points_max-size_points_min))**2
    markers_subset = list(np.random.choice(markers,size=nb_points_var_markers))
    markers_empty = np.random.rand(1)[0]>0.75
    markers_empty_ratio = random.choice([0.0,0.5,0.7])

    # BUILDING THE PLOT
    s = []
    categories = []
    cat_dict = {}
    index_cat = 0

    for _x, _y,  in zip(x,y):
        s_ = random.choice(s_set)
        c_ = random.choice(colors)
        m_ = random.choice(markers_subset)
        if m_ in markers_with_full and markers_empty:
            e_ = np.random.rand(1)[0]> markers_empty_ratio
        else:
            e_ = False
        cat = [s_,c_,m_, e_]

        if cat_in_dict(cat,cat_dict) is False:
            cat_dict[index_cat] = cat
            index_cat += 1
        categories.append(cat_in_dict(cat,cat_dict))
        s.append(s_)
        if e_:
            plt.scatter(_x, _y, s=s_, color = c_, marker=m_, facecolors='none')
        else:
            plt.scatter(_x, _y, s=s_, color = c_, marker=m_)

    # PAD BETWEEN TICKS AND LABELS
    pad_x = max(tick_size[1]+0.5,int(pad_min + np.random.rand(1)[0]*(pad_max-pad_min)))
    pad_y = max(tick_size[1]+0.5,int(pad_min + np.random.rand(1)[0]*(pad_max-pad_min)))
    direction_ticks_x = random.choice(direction_ticks)
    direction_ticks_y = random.choice(direction_ticks)

    # NON-DEFAULT TICKS PROB, WITH THRESHOLD OF 0.6
    weid_ticks_prob = np.random.rand(1)[0]

    # TICKS STYLE AND LOCATION (X AXIS)
    if np.random.rand(1)[0]>0.5:
        axes_x_pos = 1
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        if weid_ticks_prob >0.6:
            ax.xaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=pad_x,
                                 direction= direction_ticks_x, bottom=np.random.rand(1)[0]>0.5, top=True)
        else:
            ax.xaxis.set_tick_params(bottom=np.random.rand(1)[0]>0.5, top=True)
        if np.random.rand(1)[0]>0.5:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_tick_params(bottom=False)
            if np.random.rand(1)[0]>0.5:
                axes_x_pos = np.random.rand(1)[0]
                ax.spines['top'].set_position(('axes',axes_x_pos ))
    else:
        axes_x_pos = 0
        if weid_ticks_prob >0.6:
            ax.xaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=pad_x,
                                 direction= direction_ticks_x, bottom=True, top=np.random.rand(1)[0]>0.5)
        else:
            ax.xaxis.set_tick_params(bottom=True, top=np.random.rand(1)[0]>0.5)
        if np.random.rand(1)[0]>0.5:
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_tick_params(top=False)
            if np.random.rand(1)[0]>0.5:
                axes_x_pos = np.random.rand(1)[0]
                ax.spines['bottom'].set_position(('axes',axes_x_pos))

    # TICKS STYLE AND LOCATION (Y AXIS)
    if np.random.rand(1)[0]>0.5:
        axes_y_pos = 1
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        if weid_ticks_prob > 0.6:
            ax.yaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=pad_y,
                                 direction= direction_ticks_y, left=np.random.rand(1)[0]>0.5, right=True)
        else:
            ax.yaxis.set_tick_params(left=np.random.rand(1)[0]>0.5, right=True)
        if np.random.rand(1)[0]>0.5:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_tick_params(left=False)
            if np.random.rand(1)[0]>0.5:
                axes_y_pos = np.random.rand(1)[0]
                ax.spines['right'].set_position(('axes',axes_y_pos))
    else:
        axes_y_pos = 0
        if weid_ticks_prob >0.6:
            ax.yaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=pad_y,
                                 direction= direction_ticks_y, left=True, right=np.random.rand(1)[0]>0.5)
        else:
            ax.yaxis.set_tick_params(left=True, right=np.random.rand(1)[0]>0.5)
        if np.random.rand(1)[0]>0.5:
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_tick_params(right=False)
            if np.random.rand(1)[0]>0.5:
                axes_y_pos = np.random.rand(1)[0]
                ax.spines['left'].set_position(('axes',axes_y_pos))

    # LABEL ROTATION
    if np.random.rand(1)[0]>0.77:
        plt.xticks(rotation=int(np.random.rand(1)[0]*90))
    if np.random.rand(1)[0]>0.77:
        plt.yticks(rotation=int(np.random.rand(1)[0]*90))

    # SUB-TICKs
    if weid_ticks_prob > 0.6:
        color_subtick = random.choice(color_subtick_list)
        length_subtick = 0.75*np.random.rand(1)[0]*tick_size[1]
        if np.random.rand(1)[0]>0.7:
            minorLocator = AutoMinorLocator()
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_tick_params(which='minor', length=length_subtick, direction= direction_ticks_x, color=color_subtick,
                                     bottom= ax.spines['bottom'].get_visible(), top=ax.spines['top'].get_visible())
        if np.random.rand(1)[0]>0.7:
            minorLocator = AutoMinorLocator()
            ax.yaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_tick_params(which='minor', length=length_subtick, direction= direction_ticks_y, color=color_subtick,
                                     left= ax.spines['left'].get_visible(), right= ax.spines['right'].get_visible())


    # FONT AND SIZE FOR LABELS (tick labels, axes labels and title)
    font = random.choice(font_list)
    size_ticks = int(tick_label_size_min + np.random.rand(1)[0]*(tick_label_size_max-tick_label_size_min))
    size_axes = int(axes_label_size_min + np.random.rand(1)[0]*(axes_label_size_max-axes_label_size_min))
    size_title = int(title_size_min + np.random.rand(1)[0]*(title_size_max-title_size_min))
    ticks_font = font_manager.FontProperties(fname = font, style='normal', size=size_ticks, weight='normal', stretch='normal')
    axes_font = font_manager.FontProperties(fname = font, style='normal', size=size_axes, weight='normal', stretch='normal')
    title_font = font_manager.FontProperties(fname = font, style='normal', size=size_title, weight='normal', stretch='normal')

    # TEXTS FOR AXIS LABELS AND TITLE
    label_x_length = int(axes_label_length_min + np.random.rand(1)[0]*(axes_label_length_max-axes_label_length_min))
    label_y_length = int(axes_label_length_min + np.random.rand(1)[0]*(axes_label_length_max-axes_label_length_min))
    title_length = int(title_length_min + np.random.rand(1)[0]*(title_length_max-title_length_min))
    x_label = ("".join( [random.choice(string.ascii_letters+'       ') for i in range(label_x_length)] )).strip()
    y_label = ("".join( [random.choice(string.ascii_letters+'       ') for i in range(label_y_length)] )).strip()
    title = ("".join( [random.choice(string.ascii_letters+'       ') for i in range(title_length)] )).strip()
    plt.xlabel(x_label , fontproperties = axes_font)
    plt.ylabel(y_label , fontproperties = axes_font, color='black')
    if axes_x_pos==1:
        plt.title(title, fontproperties = title_font, color='black',y=1.1)
    else:
        plt.title(title, fontproperties = title_font, color='black')

    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    # GRID
    if np.random.rand(1)[0]>0.7:
        plt.grid(b=True, which='major', color=random.choice(color_grid), linestyle=random.choice(linestyles))

    # AXIS LIMITS
    xmin = min(x)
    xmax = max(x)
    deltax = 0.05*abs(xmax-xmin)
    plt.xlim(xmin - deltax, xmax + deltax)
    ymin = min(y)
    ymax = max(y)
    deltay = 0.05*abs(ymax-ymin)
    plt.ylim(ymin - deltay, ymax + deltay)


    # BACKGROUND AND PATCH COLORS
    if np.random.rand(1)[0]>0.75:
        color_bg = (1-colorbg_transparant_max)+colorbg_transparant_max*np.random.rand(3)
        ax.set_axis_bgcolor(color_bg)
    if np.random.rand(1)[0]>0.75:
        color_bg = (1-colorbg_transparant_max)+colorbg_transparant_max*np.random.rand(3)
        fig.patch.set_facecolor(color_bg)

    # MAKE SURE THE PLOT FITS INSIDE THE FIGURES
    plt.tight_layout()

    plt.savefig("./data/{}/".format(direc)+name, dpi='figure', facecolor=fig.get_facecolor())

    return ax, fig, x, y, s, categories, tick_size, axes_x_pos, axes_y_pos


def get_data_pixel(ax, fig, x, y, s):
    """
    Method that return the bouding box of the points.
    Inputs:
    ax : (matplotlib obj) Matplotlib object of the axes of the plot
    fig : (matplotlib obj) Matplotlib object of the figure of the plot
    x, y : (list, list) Actuall x and y coordinates of the points.
    s : (list) sizes of the points.
    Outputs:
    boxes (list) : list of bounding boxes for each points.
    """

    xy_pixels = ax.transData.transform(np.vstack([x,y]).T)
    xpix, ypix = xy_pixels.T

    boxes = []
    for x_j, y_j, s_j in zip(xpix,ypix,s):
        if s_j<25:
            s_j = 25
        box_size = fig.dpi*np.sqrt(s_j)/70.0
        x0 = x_j-box_size/2.0
        y0 = y_j - box_size/2.0
        x1 = x_j + box_size/2.0
        y1 = y_j + box_size/2.0
        boxes.append(Bbox([[x0, y0], [x1, y1]]))

    return boxes


def get_tick_pixel(ax, fig, tick_size, axes_x_pos, axes_y_pos):
    """
    Method that return the bouding box of the ticks.
    Inputs:
    ax : (matplotlib obj) Matplotlib object of the axes of the plot
    fig : (matplotlib obj) Matplotlib object of the figure of the plot
    tick_size : (list) Tick size on the plot. [width, length]
    axes_x_pos, axes_y_pos: (float, float) Position of the labels of the axis.
    Outputs:
    boxes_x, boxes_y (list, list) : list of bounding boxes for each ticks.
    """

    x_tick_pos = [ ax.transLimits.transform(textobj.get_position()) for textobj in ax.get_xticklabels() if len(textobj.get_text())>0]
    y_tick_pos = [ ax.transLimits.transform(textobj.get_position()) for textobj in ax.get_yticklabels() if len(textobj.get_text())>0]

    x_tick_pos = [ ax.transScale.transform(ax.transAxes.transform([array[0], axes_x_pos])) for array in x_tick_pos]
    y_tick_pos = [ ax.transScale.transform(ax.transAxes.transform([axes_y_pos, array[1]])) for array in y_tick_pos]

    boxes_x = []
    for x_j, y_j in x_tick_pos:
        box_size_x = fig.dpi*5/50.0
        box_size_y = fig.dpi*5/50.0 
        x0 = x_j-box_size_x/2.0
        y0 = y_j - box_size_y/2.0
        x1 = x_j + box_size_x/2.0
        y1 = y_j + box_size_y/2.0
        boxes_x.append(Bbox([[x0, y0], [x1, y1]]))

    boxes_y = []
    for x_j, y_j in y_tick_pos:
        box_size_x = fig.dpi*5/50.0 
        box_size_y = fig.dpi*5/50.0 
        x0 = x_j-box_size_x/2.0
        y0 = y_j - box_size_y/2.0
        x1 = x_j + box_size_x/2.0
        y1 = y_j + box_size_y/2.0
        boxes_y.append(Bbox([[x0, y0], [x1, y1]]))

    return boxes_x, boxes_y


def get_label_pixel(ax):
    """
    Method that return the bouding box of the labels.
    Inputs:
    ax : (matplotlib obj) Matplotlib object of the axes of the plot
    Outputs:
    x_label_bounds, y_label_bounds (list, list) : list of bounding boxes for each labels.
    """

    x_label_bounds = [ textobj.get_window_extent() for textobj in ax.get_xticklabels() if len(textobj.get_text())>0]
    y_label_bounds = [ textobj.get_window_extent() for textobj in ax.get_yticklabels() if len(textobj.get_text())>0]

    return x_label_bounds, y_label_bounds


def get_label_value(ax):
    """
    Method that return the value of the labels.
    Inputs:
    ax : (matplotlib obj) Matplotlib object of the axes of the plot
    Outputs:
    x_labels, y_labels (list, list) : list of the values for each labels.
    """

    x_labels, y_labels = [], []

    xticks_text = ax.get_xticklabels()
    xticks_numbers = ax.get_xticks()
    for j in range(len(xticks_text)):
        if len(xticks_text[j].get_text())>0:
            x_labels.append(xticks_numbers[j])

    yticks_text = ax.get_yticklabels()
    yticks_numbers = ax.get_yticks()
    for j in range(len(yticks_text)):
        if len(yticks_text[j].get_text())>0:
            y_labels.append(yticks_numbers[j])

    return x_labels, y_labels


def get_ground_truth(ax, fig, x, y, s, tick_size, axes_x_pos, axes_y_pos):
    """
    Method that return the bounding boxes and label values, essentially all the required
    ground truth for each plot.
    Inputs:
    ax : (matplotlib obj) Matplotlib object of the axes of the plot
    fig : (matplotlib obj) Matplotlib object of the figure of the plot
    x, y : (list, list) Actuall x and y coordinates of the points.
    s : (list) sizes of the points.
    tick_size : (list) Tick size on the plot. [width, length]
    axes_x_pos, axes_y_pos: (float, float) Position of the labels of the axis.
    Outputs:
    point_boxes: list of bounding boxes for each points.
    x_tick_boxes, y_tick_boxes: list of bounding boxes for each ticks.
    x_label_boxes, y_label_boxes: list of bounding boxes for each labels.
    x_labels, y_labels: list of the values for each labels.
    """

    point_boxes = get_data_pixel(ax, fig, x, y, s)

    x_tick_boxes, y_tick_boxes = get_tick_pixel(ax, fig, tick_size, axes_x_pos, axes_y_pos)

    x_label_boxes, y_label_boxes = get_label_pixel(ax)

    x_labels, y_labels = get_label_value(ax)

    return point_boxes, x_tick_boxes, y_tick_boxes, x_label_boxes, y_label_boxes, x_labels, y_labels


def write_idl(length_y, file_obj, plot_name, boxes, scores = None):
    """
    Function that writes bounding boxes into an idl file.
    Inputs:
    length_y: (int) length of the Y direction of the image (needed to convert coordinate system origin)
    file_obj: (file obj) file object of the idl file
    plot_name: (string) plotname of the plot with bounding boxes.
    boxes: (list) List of bounding boxes
    scores : (list) Optional, confidence score for each bouding boxes.
    """

    string_prep = '"{plot_name}":'.format(plot_name=plot_name)
    if scores is None:
        for box in boxes:
            string_prep += " ({}, {}, {}, {}),".format(int(np.round(box.x0)),int(length_y-np.round(box.y1)),
                          int(np.round(box.x1)),int(length_y-np.round(box.y0)))
    else:
        for box, score in zip(boxes,scores):
            string_prep += " ({}, {}, {}, {}):{},".format(int(np.round(box.x0)),int(length_y-np.round(box.y1)),
                          int(np.round(box.x1)),int(length_y-np.round(box.y0)),score)
    string_prep = string_prep[:-1]
    string_prep+=';'
    file_obj.write(string_prep)
    file_obj.write("\n")


def write_coords(file_obj, plot_name, x , y):
    """
    Function that writes coordinates into an idl file.
    Inputs:
    file_obj: (file obj) file object of the idl file
    plot_name: (string) plotname of the plot with bounding boxes.
    x,y : (list, list) List of x and y coordinates
    """

    string_prep = '"{plot_name}":'.format(plot_name=plot_name)
    for x_i, y_i in zip(x,y):
        string_prep += " ({}, {}),".format(x_i,y_i)
    string_prep = string_prep[:-1]
    string_prep+=';'
    file_obj.write(string_prep)
    file_obj.write("\n")


def write_labels(length_y, file_obj, plot_name, labels, label_box):
    """
    Function that writes label values into an idl file.
    Inputs:
    length_y: (int) length of the Y direction of the image (needed to convert coordinate system origin)
    file_obj: (file obj) file object of the idl file
    plot_name: (string) plotname of the plot with bounding boxes.
    labels: (string) Value of the labels.
    label_boxes: (list) List of bounding boxes
    """

    string_prep = '"{plot_name}":'.format(plot_name=plot_name)
    for j in range(len(labels)):
        box = label_box[j]
        string_prep += " ({}, {}, {}, {}):{},".format(int(np.round(box.x0)),int(length_y-np.round(box.y1)),
                          int(np.round(box.x1)),int(length_y-np.round(box.y0)),labels[j])
    string_prep = string_prep[:-1]
    string_prep+=';'
    file_obj.write(string_prep)
    file_obj.write("\n")


def generate_plots(n, file_name, direc):
    """
    Function that generates a random scatter plot, find the relevant bounding boxes, write those into a file,
    and then keep doing that in a loop.
    Inputs:
    n: (int) Number of plots to generate
    file_name: (string) String to use in the file_name of the idl files which will be saved.
    direc: Directory of where to save the images of the plots.
    """


    if not os.path.exists("./data/{}".format(direc)):
        os.makedirs("./data/{}".format(direc))
        os.makedirs("./data/{}/plots".format(direc))
    
    with open("./data/{}/".format(direc)+file_name+"_coords.idl",'w') as f_coords, \
         open("./data/{}/".format(direc)+file_name+"_points.idl",'w') as f_points, \
         open("./data/{}/".format(direc)+file_name+"_points_cat.idl",'w') as f_points_cat, \
         open("./data/{}/".format(direc)+file_name+"_ticks.idl",'w') as f_ticks, \
         open("./data/{}/".format(direc)+file_name+"_labels.idl",'w') as f_labels, \
         open("./data/{}/".format(direc)+file_name+"_label_values.idl",'w') as f_label_values:

        for j in range(n):
            try:
                plot_name =  'plots/{}_{}.png'.format(file_name,j+1)
                ax, fig, x, y, s, categories, tick_size, axes_x_pos, axes_y_pos = get_random_plot(plot_name, direc)
                length_y = fig.get_size_inches()[1]*fig.dpi

                point_boxes, x_tick_boxes, y_tick_boxes, x_label_boxes, y_label_boxes, x_labels, y_labels = get_ground_truth(ax, fig, x, y, s, tick_size, axes_x_pos, axes_y_pos)

                write_coords(f_coords, plot_name, x , y)
                write_labels(length_y, f_label_values, plot_name, x_labels+y_labels, x_label_boxes+y_label_boxes)
                write_idl(length_y, f_points, plot_name, point_boxes)
                write_idl(length_y, f_points_cat, plot_name, point_boxes, scores = categories)
                write_idl(length_y, f_ticks, plot_name,  x_tick_boxes+y_tick_boxes)
                write_idl(length_y, f_labels, plot_name, x_label_boxes+y_label_boxes)

                plt.close(fig)
            except ValueError:
                mylogger.warn("Error while generating plot.  This happens occasionally because of the tight-layout option.")


if __name__ == '__main__':

    """
    Example of command-line usage:

    python generate_random_scatter.py --directory plots_v1 --n_train 25000 --n_test 500
    """

    mylogger = scatteract_logger.get_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_train', help='Number of training images', required=True)
    parser.add_argument('--n_test', help='Number of test images', required=True)
    parser.add_argument('--directory', help='Directory to save the idl and images', required=True)
    args = vars(parser.parse_args())

    generate_plots(n=int(args['n_train']), file_name = "train", direc = args['directory'])
    generate_plots(n=int(args['n_test']), file_name = "test", direc = args['directory'])
