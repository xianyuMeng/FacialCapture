## dirty config file
class ModelConfig(object):
    def __init__(self):
        
        self.image_feature_name = 'image'

        self.predict_feature_name = 'v'

        #self.num_weight = 51
        self.predict_num = 5043 * 3
        self.pca_num = 51

        self.image_height = 256
        self.image_width = 256

        self.file_pattern = "/home/mengxy/lzhBMP/xuyi/tfrecord/tf_record*"
        self.pkl_fname = "/home/mengxy/lzhBMP/pca_51.pkl"
        self.mean_pkl_fname = "/home/mengxy/lzhBMP/mean_mesh.pkl"

        self.train_dir = ""

        ##training config
        self.initialize_scale = 0.01

        self.max_checkpoint_keep = 10

        self.optimizer = 'Adam'

        self.batch_size = 32

        self.clip_gradients = 5.
        
