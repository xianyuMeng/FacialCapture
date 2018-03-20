from __future__ import print_function
import os
import numpy as np
import pickle
import pprint
import argparse
from PIL import Image
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.externals import joblib

from preprocess_obj import _get_image_fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( \
            "--data_dir", \
            help = "image dir", \
            default = '/home/userdata/mxy/xuyi-1017/face/')
    parser.add_argument( \
            "--start", \
            help = 'image start number', \
            default = 1)
    parser.add_argument( \
            "--end", \
            help = 'image end number', \
            default = 1500)
    parser.add_argument( \
            '--image_format', \
            help = "image format", \
            default = '0032_%08d.png')
    parser.add_argument(\
            '--pkl_format', \
            help = 'output pickle file format', \
            default = '%08d.pkl')
    
    args = parser.parse_args()

    image_data = []
    for ii in xrange(int(args.start), int(args.end) + 1):
        image_fname = _get_image_fname(args.data_dir, args.image_format, ii)
        img = Image.open(image_fname)
        img = img.convert(mode = "L")
        bbox = (850, 725, 1450, 1200)
        img = img.crop(bbox).resize((256, 256), Image.BILINEAR)
        #img = img.rotate(90)
        img.save(image_fname[:-4] + '_crop.bmp', 'bmp') 
        img = np.array(img)
        img = np.reshape(img, [-1])
        image_data.append(img)

    print("Loading {} images".format(len(image_data)))
    image_data = np.array(image_data)
    print("Finish loading images")
    image_data = image_data.astype(np.float64)
    scaler = preprocessing.MinMaxScaler()
    image_scaled = scaler.fit_transform(image_data)

    scaler_model = open(args.data_dir + '/scaler_model.pkl', 'wb')
    print("Saving scaler model to {}".format(scaler_model))
    joblib.dump(scaler, scaler_model)
    scaler_model.close()

#    image_data -= np.mean(image_data, axis = 0)
#    var = np.std(image_data, axis = 0)
#    image_data /= var
#    pca = PCA(n_components = len(image_data), whiten = True)
#    pca.fit(image_data)
#    print("Finish PCA")

#    image_data -= np.mean(image_data, axis = 0)
#    cov = np.dot(image_data.T, image_data) / image_data.shape[0]
#    u, s, v = np.linalg.svd(cov)
#    image_data_rot = np.dot(image_data, u)
#    image_data_white = image_data_rot / np.sqrt(s + 1e-5)

    for ii in xrange(int(args.start), int(args.end) + 1):
#        image_fname = _get_image_fname(args.data_dir, args.image_format, ii)
#        image_data = Image.open(image_fname)
#        image_data = image_data.convert(mode = "L")
#        image_data = np.array(image_data)
#        image_data = np.reshape(image_data, [1, -1])
#        image_data = pca.transform(image_data)
        pkl_fname = _get_image_fname(args.data_dir, args.pkl_format, ii)
        pkl = open(pkl_fname, 'wb')
        print("Savin to {}".format(pkl_fname))
        pickle.dump(image_scaled[ii - int(args.start)].astype(np.float32), pkl, -1)
        pkl.close()

if __name__ == "__main__":
    main()
