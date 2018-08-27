
# coding: utf-8

# ## MEFN with real-nvp as generative model and gramian of vgg as penalty
# 
# * Note: You will need to download a pre-trained vgg network from, for example https://github.com/ry/tensorflow-vgg16/raw/master/vgg16-20160129.tfmodel.torrent, then set the vgg_path
# 
# * Seems that VGG network is tailored to 224 x 224 x 3 images so for now we will just stick to that size
# 
# * code for building penalty with vgg network is based on paper https://arxiv.org/abs/1603.03417, code from https://github.com/ProofByConstruction/texture-networks
# 
# * code for model real-nvp is based on paper https://arxiv.org/abs/1605.08803, code from https://github.com/taesung89/real-nvp
# 
# * Warning: This code is extremely slow on CPU (really slow), mostly a proof-of-concept.

import os
import sys
import time
import json
import argparse
import pickle

import numpy as np
import tensorflow as tf

sys.path.append('vgg')
sys.path.append('real_nvp')

import real_nvp.nn as real_nvp_nn # for adam optimizer
from real_nvp.model import model_spec as real_nvp_model_spec # transforming image to latent
from real_nvp.model import inv_model_jac_spec as real_nvp_inv_model_spec # transforming latent to image

from vgg.vgg_network import VGGNetwork # read vgg network and compute style and content loss
from vgg.network_helpers import load_image

#import matplotlib
#import matplotlib.pylab as plt
#get_ipython().magic(u'matplotlib inline')

#-----------------------------
# command line argument, not all of them are useful
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--style_img_path', type=str, default='img/style.jpg', help='path for the style image')
parser.add_argument('--vgg_path', type=str, default='vgg/vgg16.tfmodel', help='path for vgg network')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
#parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=100, help='Every how many epochs to write checkpoint/samples?')
#parser.add_argument('-r', '--load_params', type=int, default=0, help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
# optimization
parser.add_argument('--entropy', type=int, default=1, help='0 = No entropy, 1 = Decreasing penalty, -1 = Constant penalty')
parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='Base learning rate')
#parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
#parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_iter', type=int, default=500, help='How many epochs to run in total?')
parser.add_argument('--lam_use', type=float, default=0.99, help='lambda value when --entropy=-1') 
#parser.add_argument('-g', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
# evaluation
#parser.add_argument('--sample_batch_size', type=int, default=16, help='How many images to process in paralell during sampling?')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')


# * Put the right image path to style_img_path
# * Put the downloaded vgg model to vgg_path
args = parser.parse_args()
#args = parser.parse_args(["--style_img_path=img/style.jpg", 
#                          "--vgg_path=vgg/vgg16.tfmodel",
#                          "--save_dir=checkpoints", "--max_iter=200", 
#                          "--save_interval=20", "--batch_size=1"])
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

pickle.dump({'args':args}, open("%s/args.save"%args.save_dir, "wb"), 0)

#--------------------------------------------
# ## Set up the real-nvp generative model
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

obs_shape = (224,224,3) # size of the image to use
model_spec = real_nvp_model_spec
inv_model_spec = real_nvp_inv_model_spec
nn = real_nvp_nn

# create the model
model = tf.make_template('model', model_spec)
inv_model = tf.make_template('model', inv_model_spec, unique_name_='model')

# run once for data dependent initialization of parameters
z_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
gen_par = model(z_init)

#----------------------------------------------
# ## Use vgg network to set up the style loss
style_img_path = args.style_img_path
texture_image = tf.to_float(tf.constant(load_image(style_img_path).reshape((1, 224, 224, 3))))

z_sample = tf.placeholder(tf.float32, shape = (args.batch_size,) + obs_shape)
x, inv_jac = inv_model(z_sample)

image_vgg = VGGNetwork("image_vgg", tf.concat(0, [texture_image, x, x]), 
                       1, args.batch_size, args.batch_size, args.vgg_path)

# constraint loss
con_loss_vec = image_vgg.style_loss([(i, 1) for i in range(1, 6)])

con_loss = tf.reduce_mean(con_loss_vec)
con_loss1 = tf.reduce_mean(con_loss_vec[:int(args.batch_size / 2)])
con_loss2 = tf.reduce_mean(con_loss_vec[int(args.batch_size / 2):])


#--------------------------------------
# ## Final loss is a combination of entropy and cost

# compute entropy
entropy = tf.reduce_mean(inv_jac) + tf.reduce_mean(tf.reduce_sum(z_sample ** 2 * 0.5 + np.log(2 * np.pi) * 0.5, [1,2,3]))

# loss is a combination of -entropy and constraint violation
c_augL = tf.placeholder(tf.float32, shape = [])
lam = tf.placeholder(tf.float32, shape = [])
cost = -entropy + lam * con_loss + c_augL / 2.0 * (con_loss ** 2)

# build the SGD optimizer
all_params = tf.trainable_variables()
cost_grad1 = tf.gradients(-entropy + lam * con_loss, all_params)
cost_grad2 = tf.gradients(con_loss1, all_params)
cost_grad = [i + c_augL * con_loss2 * j for i,j in zip(cost_grad1, cost_grad2)]

tf_lr = tf.placeholder(tf.float32, shape=[])
optimizer = nn.adam_updates(all_params, cost_grad, lr=tf_lr, mom1=0.95, mom2=0.9995)

#--------------------------------------
# ## Training

n_iter=0
ent_ls = []
con_ls = []
lam_use_ls = []
c_use_ls = []
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

print("Start training")
with tf.Session() as sess: # = tf.InteractiveSession()
    sess.run(initializer) #, {z_init: np.random.normal(0.0, 1.0, (args.init_batch_size,) + obs_shape)})
    feed_dict = { z_sample: np.random.normal(0.0, 1.0, (args.batch_size,) + obs_shape)}
    hk = sess.run(con_loss, feed_dict)
    c_use = 1e-9
    gamma = 0.25
    lam_use = 0.0
    for i_augL in range(6):
        print("augL iter %d, lam = %f, c = %f"%(i_augL, lam_use, c_use))
        for i in range(args.max_iter):
            feed_dict = { z_sample: np.random.normal(0.0, 1.0, (args.batch_size,) + obs_shape), 
                          tf_lr:args.learning_rate, lam: lam_use, c_augL: c_use}
            entropy_tmp, cost_con_tmp, _ = sess.run([entropy, con_loss, optimizer], feed_dict)
            n_iter += 1
            print("iter%d, entropy=%f, constraint=%f"%(n_iter, entropy_tmp, cost_con_tmp))
            sys.stdout.flush()
            con_ls.append(cost_con_tmp)
            ent_ls.append(entropy_tmp)
            lam_use_ls.append(lam_use)
            c_use_ls.append(c_use)
            if n_iter % args.save_interval == 0:
                # ## save samples
                x_sample_ls = []
                for i_samp in range(10):
                    x_sample = sess.run(x, {z_sample: np.random.normal(0.0, 1.0, (args.batch_size,) + obs_shape)})
                    x_sample_ls.append(x_sample)
                pickle.dump({'x_sample_ls':x_sample_ls}, open("%s/sample_%d.save"%(args.save_dir, n_iter), "wb"), 0)
                saver.save(sess, "%s/params_%d.ckpt"%(args.save_dir, n_iter))
                pickle.dump({'ent_ls': ent_ls, 'con_ls': con_ls, 'lam_use_ls': lam_use_ls}, open("%s/track.save"%args.save_dir , "wb"), 0)
        #plt.imshow(x_sample[0,:,:,:], interpolation='none')

        # updating
        feed_dict = { z_sample: np.random.normal(0.0, 1.0, (args.batch_size,) + obs_shape)}
        hk_new = sess.run(con_loss, feed_dict)
        lam_use += c_use*hk_new
        if np.linalg.norm(hk_new) > gamma*np.linalg.norm(hk):
            c_use *= 4
        hk = hk_new    










