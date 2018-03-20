from __future__ import print_function
import os
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import argparse
import pickle

from preprocess_pca import extract

def load_obj(obj):
    if not os.path.isfile(obj):
        print('file {} does not exist'.format(obj))
        return None, False
    else:
        with open(obj, 'r') as f:
            vertices = []
            for line in f:
                if 'v' in line and 'vn' not in line:
                    v = [float(x) for x in list(filter(lambda x : is_number(x), line.split()))]
                    vertices.append(v)
                if 'f' in line:
                    break
            return vertices, True


def read_weight(fname):
    with open(fname, 'r') as f :
        weight = []
        for line in f :
            w = [float(x) for x in list(filter(lambda x : is_number(x), line.split()))]
            weight.append(w)
        return weight

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _floats_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))



def _get_output_tfrecord_name(output_dir, basename, index):
    return '%s/%s_%08d.tfrecord' % (output_dir, basename, index) 

def _get_image_fname(data_dir, image_format, index):
    fname = image_format % index
    return '%s/%s' % (data_dir, fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help = "image dir", default =
            "/home/userdata/mxy/xuyi-1017/face/train/")
    parser.add_argument("--start", help = "start frame number", default = 1)
    parser.add_argument("--end", help = "end frame number", default = 1414)
    parser.add_argument(\
            "--image_format", \
            help = "image format can either be pil image file or pickle file",\
            default = '%08d.pkl')
    parser.add_argument("--bshape_dir", help = "bshape obj file dir", default =
            '/home/userdata/mxy/xuyi-1017/face/fit_1009/')
    parser.add_argument("--bshape_format", help = "bshape file pattern, must be either obj or pkl", default = 'fit_%d.obj.partial_face.pkl')
    parser.add_argument("--augmentation_size", help = "rotate and translate", default = 10)
    parser.add_argument("--use_augmentation", help = "do data augmentation or not", default = False)

    parser.add_argument("--tfrecord_dir", help = "tfreccord dir", default = "/home/mengxy/lzhBMP/xuyi/tfrecord/")
    parser.add_argument("--tfrecord_fname", help = "tfrecord basename", default = "tf_record")
    parser.add_argument("--tfrecord_num", help = "store number of images per tfrecord", default = 100)
    parser.add_argument("--ignore_obj", \
            help = "abandon obj filename", \
            default = "/home/mengxy/lzhBMP/abandon.txt")
    #parser.add_argument("--pca_num", help = "pca basis number", default = 160)


    args = parser.parse_args()
    
    idx = 0
    if not os.path.isdir(args.tfrecord_dir):
		os.makedirs(args.tfrecord_dir)
		print("making dir {}".format(args.tfrecord_dir))
    record_fname = _get_output_tfrecord_name(args.tfrecord_dir, args.tfrecord_fname, idx)
    tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)
    print("Saving to {}".format(record_fname))

    abandon = []
    if os.path.isfile(args.ignore_obj):
        abandon_f = open(args.ignore_obj, 'r')
        for line in abandon_f:
            if line == '\n' or line == '\t' or line == '\r':
                continue
            l = line.replace('\n', '')
            l = l.replace('\t', '')
            l = l.replace('\r', '')
            abandon.append(l)
   
    tmp = 0
    cnt = 0
    for ii in xrange(int(args.start), int(args.end) + 1):
        image_fname = _get_image_fname(args.data_dir, args.image_format, ii)
        bshape_fname = _get_image_fname(args.bshape_dir, args.bshape_format, ii - int(args.start) + 1)
        if any([a in bshape_fname for a in abandon]):
            print("\t[INFO] : ignore {}\t".format(bshape_fname))
            continue

        if not os.path.isfile(bshape_fname):
            print("bshape file {} does not exist".format(bshape_fname))
            continue
        if not os.path.isfile(image_fname) :
            print("image file {} does not exist".format(image_fname))
            continue

        if bshape_fname[-3:] == 'obj' :
            bshape, _ = load_obj(bshape_fname)
        elif bshape_fname[-3:] == 'pkl' :
            pkl_f = open(bshape_fname)
            bshape = pickle.load(pkl_f)
            pkl_f.close()
        else:
            continue
        if image_fname[-3:] == 'pkl':
            pkl_f = open(image_fname)
            img = pickle.load(pkl_f)
            img = np.reshape(img, [256, 256])
            img = Image.fromarray(np.uint8(img * 255))
            pkl_f.close()
        else:
            try:
                img = Image.open(image_fname)
            except IOError:
                print("Failed to open image file {}".format(image_fname))
                continue
            img = img.convert(mode = "L")



        bshape = np.reshape(bshape, [-1])
        bshape = np.array(bshape)
        bshape = bshape.astype(np.float32)
        if not args.use_augmentation :
            example = tf.train.Example(features = tf.train.Features(feature = {\
                    'image' : _bytes_feature(img.tobytes()), \
                    'v' : _floats_feature(bshape) \
                    }))
            tfrecord_writer.write(example.SerializeToString())
            tmp += 1
            if tmp % args.tfrecord_num == 0:
                tfrecord_writer.close()
                idx += 1
                record_fname = _get_output_tfrecord_name(args.tfrecord_dir, args.tfrecord_fname, idx)
                tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)
                print("Saving to {}".format(record_fname))
                print("finish {}".format(tmp / (float(int(args.end) - int(args.start) + 1))))


        else:

            rotation = np.random.normal(0, 10, int(args.augmentation_size))
            x_shift = np.random.normal(0, 30, int(args.augmentation_size))
            y_shift = np.random.normal(0, 30, int(args.augmentation_size))
            for jj in xrange(0, int(args.augmentation_size)):
                rotated = img.rotate(rotation[jj])
                translated = rotated.transform( \
                        rotated.size,\
                        Image.AFFINE,\
                        (1, 0, x_shift[jj], 0, 1, y_shift[jj]))
                translated = np.array(translated)
                #rotated = np.array(rotated)
       
             # print("img shape is {}".format(translated.shape))
                example = tf.train.Example(features = tf.train.Features(feature = { \
                        'image' : _bytes_feature(rotated.tobytes()), \
                        'v' : _floats_feature(bshape) \
                     }))
#               print("{}".format(example))
                tfrecord_writer.write(example.SerializeToString())
                tmp += 1
                if tmp % args.tfrecord_num == 0 :
                    tfrecord_writer.close()
                    idx += 1
                    record_fname = _get_output_tfrecord_name(args.tfrecord_dir, args.tfrecord_fname, idx)
                    print("Saving to {}".format(record_fname))
                    tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)
                cnt += 1
                print("finish {}".format(cnt / (int(args.augmentation_size) * float(int(args.end) - int(args.start) + 1))))

 
if __name__  == "__main__":
    main()
