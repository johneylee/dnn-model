import os
import time
import config as cfg
import senet_train_library as senet
import tensorflow as tf
from tools_for_network import *

# Load data.
print('Load data...')
[train_input, train_refer, devel_input, devel_refer] \
    = open_data(cfg.train_input_data_path, cfg.train_refer_data_path,
                cfg.devel_input_data_path, cfg.devel_refer_data_path)

# Define hyper parameters for learning.
hps = senet.hparams(mode=cfg.mode, job_dir=cfg.job_dir,
					num_context_window=cfg.num_context_window,
					dnn_struct=get_structure_params(cfg.input_dim, 
													cfg.output_dim,
													cfg.layer_size,
													cfg.num_layer,
													cfg.num_context_window),
					learning_rate=cfg.learning_rate, max_epochs=cfg.max_epochs)

# Define tensorflow session.
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % cfg.gpu_id
options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_frac)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=options))

# Initialize network for speech enhancement.
net = senet.SpeechEnhancementNetwork()
net.set_params(hps)
net.initialize_model(sess)

# Train network for speech enhancement
net.train(sess, train_input, train_refer, devel_input, devel_refer)
