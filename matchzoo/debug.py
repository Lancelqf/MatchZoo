import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute, Reshape, Dot, Concatenate

sess = tf.InteractiveSession()
K.set_session(sess)

q_embed = tf.constant([[[-0.1],[0.2],[0.3],[0.4]], [[0.1],[0.2],[0.3],[0.0]], [[0.1],[0.2],[0.3],[0.9]]])
d_embed = tf.constant([[[0.1],[0.2],[0.3],[0.4],[0.5]], [[0.2],[0.3],[0.4],[0.5],[0.6]], [[0.3],[0.4],[0.5],[0.6],[0.7]]])

mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])

mm_k_pos = Lambda(lambda x: K.tf.nn.top_k(x, k=2, sorted=True)[0])(mm1)

mm2 = Activation(lambda x: -1*x)(mm)
mm_k_neg = Lambda(lambda x: K.tf.nn.top_k(x, k=2, sorted=True)[0])(mm2)
mm_k_neg = Activation(lambda x: -1*x)(mm_k_neg)


'''
#mm_k, _ = tf.nn.top_k(mm, k=2, sorted=True)
mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=2, sorted=True)[0])(mm)
print 'mm_k value:'
print mm_k.eval()
print K.shape(mm_k).eval()

act1 = Activation(advanced_activations.ThresholdedReLU(theta=0.1))(mm_k)
print 'act1 value:'
print act1.eval()
print K.shape(act1).eval()

sum1 = Lambda(lambda x:K.tf.reduce_sum(x,2))(act1)
print 'sum1 value:'
print sum1.eval()
print K.shape(sum1).eval()

act2 = Activation('tanh')(sum1)
print 'act2 value:'
print act2.eval()
print K.shape(act2).eval()

sum2 = Lambda(lambda x:K.tf.reduce_sum(x,1))(act2)
print 'sum2 value:'
print sum2.eval()
print K.shape(sum2).eval()

mean = Lambda(lambda x: x/4.0)(sum2)
print 'mean value:'
print mean.eval()
print K.shape(mean).eval()

out2 = Reshape((1,))(mean)
print 'out2 value:'
print out2.eval()
print K.shape(out2).eval()

comb = Lambda(lambda x: K.tf.stack(x, 1))([out2, out2])
print 'comb value:'
print comb.eval()
print K.shape(comb).eval()

comb_reshape = Reshape((2,))(comb)
print 'comb_reshape value:'
print comb_reshape.eval()
print K.shape(comb_reshape).eval()
'''
sess.close()