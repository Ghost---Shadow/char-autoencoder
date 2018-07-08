import tensorflow as tf
import numpy as np

class MyLSTM:
    def __init__(self,input_size,state_size):
        self.input_size = input_size
        self.state_size = state_size

        # LSTM Trainable variables
        self.W = {}
        self.U = {}
        self.B = {}
        with tf.name_scope('LSTM_Weights'):
            for index in ['i','f','o','g']:
                self.W[index] = tf.Variable(tf.random_uniform([input_size,state_size],minval=-.1,maxval=.1),name='W_'+index)
                self.U[index] = tf.Variable(tf.random_uniform([state_size,state_size],minval=-.1,maxval=.1),name='U_'+index)
                self.B[index] = tf.Variable(tf.random_uniform([state_size],minval=-.1,maxval=.1),name='B_'+index)

    def _unroll(self,x,s,c):
        time_steps = len(x)

        # Static RNN
        outputs = []
        for t in range(time_steps):
            with tf.name_scope('unroll_'+str(t)):
                i = tf.nn.sigmoid(tf.matmul(x[t],self.W['i']) + tf.matmul(s[t],self.U['i']) + self.B['i'],name='i')
                f = tf.nn.sigmoid(tf.matmul(x[t],self.W['f']) + tf.matmul(s[t],self.U['f']) + self.B['f'],name='f')
                o = tf.nn.sigmoid(tf.matmul(x[t],self.W['o']) + tf.matmul(s[t],self.U['o']) + self.B['o'],name='o')

                g = tf.nn.tanh(tf.matmul(x[t],self.W['g']) + tf.matmul(s[t],self.U['g']) + self.B['g'],name='g')
                
                c.append(tf.add(tf.multiply(c[t],f),tf.multiply(g,i),name='c'))
                s.append(tf.multiply(tf.nn.tanh(c[-1]),o,name='s'))
                
                outputs.append(o)

        return outputs,s,c
        
    def encode(self,x):
        with tf.name_scope('encoder'):
            # Set initial state to zero
            initial_state = tf.zeros([1,self.state_size])
            s = [initial_state]

            initial_candidate = tf.zeros([1,self.state_size])
            c = [initial_candidate]

            # Unroll to get states
            _,s,c = self._unroll(x,s,c)

            # Take the last state
            s = s[-1]
            c = c[-1]
        return s,c

    def decode(self,s,c,time_steps):
        s = [s]
        c = [c]
        with tf.name_scope('decoder'):
            with tf.name_scope('get_batch_size'):
                batch_size = tf.shape(s[0])[0]
            
            # Pad with 0s for input
            x = tf.unstack(tf.zeros([batch_size,time_steps,self.input_size]),axis=1,name='zero_input')

            # Unroll to get outputs
            outputs,_,_ = self._unroll(x,s,c)
        return outputs

    def autoencode(self,x):
        time_steps = len(x)
        s,c = self.encode(x)
        o = self.decode(s,c,time_steps)
        return o
'''
time_steps = 3
input_size = 5
state_size = 4

x_ = tf.placeholder(tf.float32,[None,time_steps,input_size])
s_ = tf.placeholder(tf.float32,[None,state_size])
c_ = tf.placeholder(tf.float32,[None,state_size])
x = tf.unstack(x_,axis=1)
s = s_
c = c_

x_data = np.zeros((2,time_steps,input_size),np.float32)
s_data = np.zeros((2,state_size),np.float32)
c_data = np.zeros((2,state_size),np.float32)

myLstm = MyLSTM(input_size, state_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(myLstm.autoencode(x),{x_:x_data})

    encode_op = myLstm.encode(x)
    sa,ca = sess.run(encode_op,{x_:x_data})
    print(sa.shape,ca.shape)

    decode_op = myLstm.decode(s,c,time_steps)
    oa = sess.run(decode_op,{s_:s_data,c_:c_data})
    print(np.array(oa).shape)
    
    summaryWriter = tf.summary.FileWriter('./logs',sess.graph)
'''
