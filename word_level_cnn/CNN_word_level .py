
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
import math
import random
import json
import time
from sklearn.utils import shuffle
import re


# In[ ]:

path_ = '/home/qingyi/data/yelp/yelp_'
dataset =1


# In[ ]:

print('dataset', dataset)
train_y = []
train_x = []
with open(path_ + 'train' + str(dataset) + '.json') as f:
    for line in f:
        t = json.loads(line)
        train_y.append(t['label'])
        train_x.append(t['text'])
train_x,train_y = shuffle(train_x,train_y)

test_y = []
test_x = []

with open(path_ + 'test' + str(dataset) + '.json') as f:
    for line in f:
        t = json.loads(line)
        test_y.append(t['label'])
        test_x.append(t['text'])
test_x,test_y= shuffle(test_x,test_y)
print 'loaded properly'


# In[ ]:

def clean_split_length(data):
    return len(clean_str(data).split())
def clean_split(data):
    return clean_str(data).split()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# In[ ]:

def count_each(data):
    data = clean_split(data)
    for word in data:
        if word in count:
            count[word]+=1
        else:
            count[word]=1

tick = time.time()
count = {}
instance =0
for x in train_x:
    count_each(x)
    instance +=1 
    if instance%100000==0:
        print 'instance processed '+ str(instance)
tock= time.time()
print str((tock-tick)/60) + 'minutes to process'


# In[ ]:

#computed in full file
num_class= 2 # number of differnt classes
batch_size = 128
embedding_size=300
vocab_size =len(dictionary)
n_train= len(train_x)
num_epoch=2
max_sentence_length =500

print ('vocab size', vocab_size)


# In[ ]:

def vectorizer(vec,size):
    vector =[]
    for each in vec:
        v= np.zeros(size)
        v[each]=1
        vector.append(v)
    return vector
train_y= vectorizer(train_y,num_class)
test_y=vectorizer(test_y,num_class)


# In[ ]:

# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf
x = tf.placeholder(tf.int32, shape =[None, max_sentence_length])
y_ =tf.placeholder(tf.float32, shape = [None,num_class])


# In[ ]:

sess= tf.InteractiveSession()


# In[ ]:

#building a multilayer convolutional network
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)


def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)


#Convolution and Pooling
def conv2d(x, W,Strides = [1, 1, 1, 1]):
   return tf.nn.conv2d(x, W, strides=Strides, padding='VALID')


def max_pool_2x2(x, k_size=[1, 1, 1, 1], Strides=[1, 1, 1, 1]):
   return tf.nn.max_pool(x, ksize=k_size,
                         strides=Strides, padding='VALID')


# In[ ]:

with tf.name_scope('embeddings'):
    embeddings= tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="embeddings")
    x_embed_tensor= tf.nn.embedding_lookup(embeddings,x)
    x_embed= tf.expand_dims(x_embed_tensor,-1)


# In[ ]:

#First convolutional layer 
#parameter
pooled_outputs=[]
filter_sizes =[2,3,4]
num_filters = 50
s= 3 # hyperparemeter for norm scalings
with tf.name_scope("conv-layer"):
    W_shape = [filter_sizes[0],embedding_size,1, num_filters]
    W_2 = weight_variable(W_shape) 
    b = bias_variable([num_filters])
    conv =  conv2d(x_embed, W_2) + b
    h_conv_2= tf.nn.relu(conv)
    pool = max_pool_2x2(h_conv_2, [1, max_sentence_length-filter_sizes[0]+1, 1, 1 ])
    pooled_outputs.append(pool)
    
    W_shape = [filter_sizes[1],embedding_size,1, num_filters]
    W_3 = weight_variable(W_shape) 
    b = bias_variable([num_filters])
    conv =  conv2d(x_embed, W_3) + b
    h_conv_3 = tf.nn.relu(conv)
    pool = max_pool_2x2(h_conv_3, [1, max_sentence_length-filter_sizes[1]+1, 1, 1 ])
    pooled_outputs.append(pool)
        
    W_shape = [filter_sizes[2],embedding_size,1, num_filters]
    W_4 = weight_variable(W_shape) 
    b = bias_variable([num_filters])
    conv =  conv2d(x_embed, W_4) + b
    h_conv_4 = tf.nn.relu(conv)
    pool = max_pool_2x2(h_conv_4, [1, max_sentence_length-filter_sizes[2]+1, 1, 1 ])
    pooled_outputs.append(pool)


# In[ ]:

num_filter_total= num_filters*len(filter_sizes)
h_pool=tf.concat(3,pooled_outputs)
h_flat= tf.reshape(h_pool,[-1, num_filter_total])


# In[ ]:

# Dropout
with tf.name_scope("dropout"):
    keep_prob = tf.placeholder(tf.float32)
    h_num_class1_drop = tf.nn.dropout(h_flat, keep_prob)


# In[ ]:

l2_loss = tf.constant(0.0)
with tf.name_scope("output"):
    W= tf.get_variable(
        "W",
        shape = [num_filter_total, num_class],
        initializer = tf.contrib.layers.xavier_initializer())
    b= tf.Variable(tf.constant(0.1, shape=[num_class], name ="b"))
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores =tf.nn.xw_plus_b(h_num_class1_drop,W,b, name ="scores")
    predictions= tf.argmax(scores, 1, name="predictions")


# In[ ]:

#Calculate the mean cross entropy 
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(scores,y_)
    loss = tf.reduce_mean(losses) + s * l2_loss
    ce_sum = tf.scalar_summary("cross entropy (loss)", loss)


# In[ ]:

#Accuracy
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    training_summary = tf.scalar_summary("training_accuracy", accuracy)
    validation_summary = tf.scalar_summary("validation_accuracy", accuracy)


# In[ ]:

#training step
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(loss)
train_step= optimizer.apply_gradients(grads_and_vars, global_step= global_step)


# In[ ]:

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/home/qingyi/tensorflow/tensorboard", sess.graph)


# In[ ]:

sess.run(tf.initialize_all_variables())


# In[ ]:

saver = tf.train.Saver()


# In[ ]:

def get_word_id(w):
    if w in dictionary:
        return dictionary[w]
    return dictionary['UNK']

def get_id(x):
    x = clean_split(x)
    ids = np.full((max_sentence_length), dictionary['NOWORD'],dtype=np.int32)
    if len(x)> max_sentence_length:
        x= x[-max_sentence_length:]
    for i in range(len(x)):
        ids[i]=get_word_id(x[i])
    return ids       

def get_batch_id(batch_x):
    ids=[]
    for x in batch_x:
        ids.append(get_id(x))
    return ids


# In[ ]:

def test_acc():
    t_len=batch_size*15
    test_acc_all=[]
    for j in range(len(test_x)/t_len):
            batch_test = j*t_len
            batch_test_end = batch_test+t_len
            batch_test_x= get_batch_id(test_x[batch_test:batch_test_end])
            batch_test_y= test_y[batch_test:batch_test_end]
            te_ac, test_sum= sess.run([accuracy, validation_summary],feed_dict={x: batch_test_x, y_: batch_test_y, keep_prob: 1.0})
            test_acc_all.append(te_ac)
    test_accuracy=np.mean(test_acc_all)
    print("test accuracy %g" % test_accuracy)   


# In[ ]:

s=0
train_steps =n_train/batch_size*num_epoch
tick = time.time()


# In[ ]:

while s < train_steps:
    batch_ind=s*batch_size%n_train
    batch_ind_end=batch_ind +batch_size 
    batch_x = get_batch_id(train_x[batch_ind:batch_ind_end])
    batch_y= train_y[batch_ind:batch_ind_end]
    
    train_step.run(session=sess, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    if s% (n_train/batch_size) ==0 and s >0:
        print 'after' +str(s/n_train)+ 'epoch'
        test_acc()
        train_x, train_y=shuffle(train_x,train_y)
    if s% 200 ==0:
        feed_dict= {x: batch_x, y_: batch_y, keep_prob: 1.0}
        summary_str = sess.run(merged, feed_dict= feed_dict)
        writer.add_summary(summary_str, s)
        train_accuracy, train_sum = sess.run([accuracy, training_summary], feed_dict=feed_dict)
        train_loss = loss.eval(session=sess, feed_dict= feed_dict)
        writer.add_summary(train_sum, s) 
        tock= time.time()
        test_acc_all=[]
        
        #evalutate a portion of the test set randomly choosen
        t_len=batch_size*10
        rg = random.randrange(len(test_y)/(10*t_len))
        
        for j in range(rg,rg+10):
            batch_test = j*t_len
            batch_test_end = batch_test+t_len
            batch_test_x= get_batch_id(test_x[batch_test:batch_test_end])
            batch_test_y= test_y[batch_test:batch_test_end]
            te_ac, test_sum= sess.run([accuracy, validation_summary],feed_dict={x: batch_test_x, y_: batch_test_y, keep_prob: 1.0})
            test_acc_all.append(te_ac)
            
        test_accuracy=np.mean(test_acc_all)
        print("step %d, training accuracy %g, loss %g " % (s, train_accuracy, train_loss))
        print("test accuracy %g" % test_accuracy)
        print('time elapsed',(tock-tick)/3600)
    
    if s % 1000 == 0:
        # Append the step number to the checkpoint name:
            saver.save(sess, 'yelp_run/my-model', global_step=s)

    s+=1
          

