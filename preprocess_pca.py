from __future__ import print_function
import os
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import argparse
import pickle


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
            return np.array(vertices), True

def extract(vertices, index):
    v = []
    for ii in range(0, int(len(vertices) / 3)):
        if ii  in index:
            v.extend([vertices[ii * 3], vertices[ii * 3 + 1], vertices[ii * 3 + 2]])
    return v

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _floats_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))


def _get_image_fname(data_dir, image_format, index):
    fname = image_format % index
    return '%s/%s' % (data_dir, fname)

def _get_output_tfrecord_fname(data_dir, data_format, index):
    fname = data_format % index
    return '%s/%s' % (data_dir, fname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bshape_dir", help = "bshape obj file fir", default =
            "/home/userdata/mxy/xuyi-1017/face/fit_1009/")
    parser.add_argument("--bshape_format", help = "bshape file pattern", default = "fit_%d.obj")
    parser.add_argument("--pca_num", help = "pca basis number", default = 160)
    parser.add_argument("--start", help = "bshape start number", default = 1)
    parser.add_argument("--end", help = "bshape end number", default = 1500)
    parser.add_argument("--pkl_fname", help = "pickle filename", default = "pca.pkl")
    parser.add_argument("--index", help = "index for bshape file", default =
            "/home/mengxy/lzhBMP/xuyi_index.pkl")
    parser.add_argument("--mean_pkl", help = "store mean vector of inputs", default = "mesh_mean.pkl")
    parser.add_argument("--save_partial_face", help = "save extracted face corresponding to index file", default = True)
    args = parser.parse_args()
	
    mesh = []
    Mesh_not_exist = []
    for ii in range(int(args.start), int(args.end) + 1):
        m, good = load_obj(_get_image_fname( \
                args.bshape_dir, \
                args.bshape_format,\
                ii))
        if not good:
          Mesh_not_exist.append(ii)
          continue
        else:
          m = np.reshape(m, -1)
          mesh.append(m)
    mesh = np.array(mesh)
    print("Loading Mesh of shape {}\n".format(mesh.shape))

    if os.path.isfile(args.index):
        print("Loading index file {}".format(args.index))
        index_f = open(args.index, 'rb')
        index = pickle.load(index_f)
        print("index shape is {}".format(index.shape))
        mesh_sub = [extract(m, index) for m in list(mesh)]
        mesh_sub = np.array(mesh_sub)
        print("extracted mesh shape is {}".format(mesh_sub.shape))
        if args.save_partial_face :
            tmp = 0
            for ii in range(int(args.start), int(args.end) + 1):
                if ii not in Mesh_not_exist:
                  bshape_fname = _get_image_fname( \
                        args.bshape_dir, \
                        args.bshape_format, \
                        ii)
                  extracted_fname = open(bshape_fname + '.partial_face.pkl', 'wb')
                  print("Saving to {}".format(extracted_fname))
                  pickle.dump(mesh_sub[tmp].astype(np.float32), extracted_fname, -1)
                  tmp += 1
                  extracted_fname.close()
                else:
                  continue
        mesh = mesh_sub
        index_f.close()
    else:
        print("index file does not exist, calculate pca basis for all vertices")

#    mean_mesh = np.mean(mesh, axis = 0)
    print("Loading training mesh of shape {}".format(mesh.shape))
#    for ii in xrange(0, mesh.shape[0]):
#        mesh[ii] = mesh[ii] - mean_mesh
#    print("Finish processing mesh")
    mean_mesh = np.mean(mesh, axis = 0)
    mean_fname = os.path.join(args.bshape_dir, args.mean_pkl)
    print("Saving to {}".format(mean_fname))
    mean_fname = open(mean_fname, 'wb')
    pickle.dump(mean_mesh.astype(np.float32), mean_fname, -1)
    mean_fname.close()


    pca = PCA(args.pca_num).fit(mesh)
    print("Finish PCA; n_com = {}".format(args.pca_num))
    
    pkl_fname = os.path.join(args.bshape_dir, args.pkl_fname)
    pkl = open(pkl_fname, 'wb')
    pickle.dump(pca.components_.astype(np.float32), pkl, -1)
    pkl.close()

if __name__ == "__main__":
    main()
