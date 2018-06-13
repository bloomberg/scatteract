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

import argparse
import os

from scipy.misc import imread, imsave
import numpy as np

from tensorbox.utils.annolist import AnnotationLib as al


def write_labels(file_name, plot_name, label):

    string_prep = '"{plot_name}"\t'.format(plot_name=plot_name)
    string_prep += str(label)
    file_name.write(string_prep)
    file_name.write("\n")


def main(directory, labels_idl):

    if not os.path.exists("{}/label_images".format(directory)):
        os.makedirs("{}/label_images".format(directory))

    with open("{}/".format(directory) + "label_image_values.tsv", 'w') as f_label_values:
        labels_annos = al.parse(labels_idl)

        for j in range(len(labels_annos)):

            labels_anno = labels_annos[j]
            img = imread(directory + "/" + labels_anno.imageName)

            index = 1
            for rect in labels_anno.rects:

                curr_label_image = np.copy(img[int(rect.y1):int(rect.y2),
                                               int(rect.x1):int(rect.x2), :])
                curr_label_value = rect.score
                if np.shape(curr_label_image)[0] > 0 and np.shape(curr_label_image)[1] > 0:
                    curr_file_name = (labels_anno.imageName.split('/')[-1][:-4] +
                                      '_{}.png'.format(index))
                    imsave("{}/label_images/".format(directory) + curr_file_name, curr_label_image)
                    write_labels(f_label_values, curr_file_name, curr_label_value)
                    index += 1


if __name__ == '__main__':

    """
    Example of command-line usage:

    python generate_test_tesseract.py --image_dir data/plots8 --label_values_idl  data/plots8/test_label_values.idl
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', help='Image directory', required=True)
    parser.add_argument('--label_values_idl',
                        help='Path of the relevant label-values idl', required=True)
    args = vars(parser.parse_args())

    main(args['image_dir'], args['label_values_idl'])
