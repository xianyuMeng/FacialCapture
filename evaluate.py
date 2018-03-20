from __future__ import print_function
from __future__ import absolute_import 
from __future__ import division

from ModelConfig import ModelConfig
from VisualModule import Module

import os
import pickle
import pprint
import tensorflow as tf
import numpy as np
from PIL import Image
import glob

def load_obj(obj_fname):
    if not os.path.isfile(obj_fname):
        print(' file {} does not exist'.format(obj_fname))
        return  None, None
    else:
        with open(obj_fname, 'r') as f:
            vertices = []
            faces = []
            for line in f:
                if 'v' in line and 'vn' not in line:
                    v = [float(x) for x in list(filter(lambda x : is_float(x), line.split()))]
                    vertices.append(v)
                if 'f' in line:
                    line = line.replace('//', ' ')
                    ff = [int(x) for x in list(filter(lambda x : is_int(x), line.split()))]
                    if len(ff) == 6:
                        ff_extracted = [ff[0], ff[2], ff[4]]
                    elif len(ff) == 3:
                        ff_extracted = ff
                    faces.append(ff_extracted)
        vertices = np.array(vertices)
        faces = np.array(faces)
        return vertices, faces

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def plot_obj(v, face, filename):
    with open(filename, 'w') as f:
        for i in v:
            f.write('v {} {} {} \n'.format(i[0], i[1], i[2]))
        for j in face:
            f.write('f {} {} {} \n'.format(j[0], j[1], j[2]))
    f.close()
    return

def plot_ply(v, face, filename):
    if v.shape[0] < v.shape[1]:
        v = np.transpose(v)

    if face.any():
        if face.shape[0] < face.shape[1]:
            face = np.transpose(face)
        nf = face.shape[0]
    nv = v.shape[0]
    #nf = face.shape[0]
    with open(filename, 'wt') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % nv)
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %d\n' % nf)
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for ii in xrange(0, nv):
            f.write('%f %f %f\n' % (v[ii][0], v[ii][1], v[ii][2]))
        if face.any():
            for ii in xrange(0, nf):
                f.write('3 %d %d %d\n' % (face[ii][0] - 1, face[ii][1] - 1,
                    face[ii][2] - 1))
        f.close()
    return





FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string(\
        "checkpoint_dir", \
        "", \
        "path to checkpoint file")
tf.flags.DEFINE_string( \
        "index_file", \
        "/home/mengxy/lzhBMP/xuyi_index.pkl", \
        "path to face index file")
tf.flags.DEFINE_string( \
        "bshape_base_file", \
        "/home/userdata/mxy/xuyi-1017/face/fit_1009/fit_1.obj", \
        "path to base bshape file ( usually use neutral face) ")
tf.flags.DEFINE_string( \
        "image_dir", \
        "/home/userdata/mxy/xuyi-1017/face/test/", \
        "path to image dir")
tf.flags.DEFINE_string( \
        "save", \
        "save.txt", \
        "path to save vertices")
tf.flags.DEFINE_string( \
        "output_dir", \
        "/home/userdata/mxy/xuyi-1017/face/inference/", \
        "output mesh dir")

def main(_):
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
   
    if not os.path.isfile(FLAGS.index_file):
        print("index pickle file {} does not exist".format(FLAGS.index_file))
        return
    if not os.path.isfile(FLAGS.bshape_base_file):
        print(" bshape base file {} does not exist".format(FLAGS.bshape_base_file))
        return
    if not os.path.isdir(FLAGS.image_dir):
        print("image dir {} does not exist".format(FLAGS.image_dir))
        return
    if not os.path.isdir(FLAGS.checkpoint_dir):
        print("checkpoint file {} does not exist".format(FLAGS.checkpoint_dir))
        return
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
       tf.logging.info("Creating output dir : %s", FLAGS.output_dir)
       tf.gfile.MakeDirs(FLAGS.output_dir)

    image_files = glob.glob(FLAGS.image_dir + '/*.bmp')
    print("# images : {}".format(len(image_files)))

    try:
        index_file = open(FLAGS.index_file)
    except IOError:
        return
    index = pickle.load(index_file)
    index = index.astype(int)
    base_v, base_f = load_obj(FLAGS.bshape_base_file)
    
    
    config = ModelConfig()
    model = Module(config, mode = 'inference')
    model.build()
    saver = tf.train.Saver()
    coef = open(FLAGS.save, 'w')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options) ) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        print("ckpt model checkpoint path {}".format(ckpt.model_checkpoint_path))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Restored")
        else:
            return
        #saver.restore(sess, FLAGS.checkpoint_file)
        for ii in image_files:
            if not os.path.isfile(ii):
                print("image file {} does not exist".format(ii))
                continue
            try:
                img = Image.open(ii)
            except IOError:
                continue
            img = img.convert(mode = "L")
            prediction = np.squeeze(sess.run(model.prediction, \
                    feed_dict = { model.image_feed : img.tobytes() } ))
            pca_coef = np.squeeze(sess.run(model.pca_coef, \
                    feed_dict = { model.image_feed : img.tobytes() } ))
           
            for jj in pca_coef:
                coef.write("{} ".format(jj))
            coef.write("\n")
            prediction = prediction.reshape((int(len(prediction) / 3), 3))
            
            deform_v = base_v
            deform_v[index] = prediction
            output_fname = FLAGS.output_dir + "/" + ii[len(FLAGS.image_dir):-4] + '_inference.obj'
            print("output {}".format(output_fname))
            plot_obj(deform_v, base_f, output_fname)
            plot_ply(deform_v, base_f, output_fname[:-4] + '.ply')


if __name__ == "__main__":
    tf.app.run()
