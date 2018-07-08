import tensorflow as tf
import numpy as np

from MyLSTM import MyLSTM
from preprocess import stringsToArray

class Seq2SeqAutoencoder:
    def __init__(self,sess,state_size=128,time_steps=10,input_size=27,unk_token=26,logdir='./logs'):
        # Global constants
        self.UNK_TOKEN = unk_token
        self.state_size = state_size
        self.time_steps = time_steps
        self.input_size = input_size
        self.interp_steps = 4

        # Placeholder
        self.x_ = tf.placeholder(tf.int32,[None,time_steps,1],name='x_input')
        self.s_ = tf.placeholder(tf.float32,[None,state_size],name='s_input')
        self.c_ = tf.placeholder(tf.float32,[None,state_size],name='c_input')        

        with tf.name_scope('process_input'):
            self.x = tf.squeeze(tf.one_hot(self.x_,self.input_size,axis=2),
                                axis=3,name='x_one_hot')
            self.y = self._getY(self.x)
            self.x = tf.unstack(self.x,axis=1)
            self.s = self.s_
            self.c = self.c_

        with tf.name_scope('prediction_vars'):
            self.W1 = tf.Variable(tf.random_normal([self.state_size,self.state_size//2]))
            self.W1 = tf.expand_dims(self.W1,0)
            self.B1 = tf.Variable(tf.random_normal([self.state_size//2]))
            
            self.W2 = tf.Variable(tf.random_normal([self.state_size//2,self.input_size]))
            self.W2 = tf.expand_dims(self.W2,0)
            self.B2 = tf.Variable(tf.random_normal([self.input_size]))

        # Model
        myLstm = MyLSTM(self.input_size,self.state_size)
        with tf.name_scope('auto_encoder'):
            self.outputs = myLstm.autoencode(self.x)
            
        with tf.name_scope('get_predictions'):
            self.predictions = self._getPredictions(self.outputs)

        # Evaluation
        self.encode_op = myLstm.encode(self.x)
        self.decode_op = self._getPredictions(myLstm.decode(self.s,self.c,time_steps))
        self.decode_op = tf.argmax(self.decode_op,2)

        # Text summary
        text_init = np.squeeze(np.array([[['']*(self.interp_steps+1)]*(self.interp_steps+1)]))
        #text_init = ['','','']
        self.text_variable = tf.Variable(text_init,name='text_variable')
        tf.summary.text('text_summary',tf.convert_to_tensor(self.text_variable,dtype=tf.string))

        # Loss and optimization
        self.loss = self._getLoss(self.x,self.predictions)
        self.optimization_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Evaluation
        self.accuracy = self._getAccuracy(self.x,self.predictions)
        self.summaryWriter = tf.summary.FileWriter(logdir,sess.graph)        
        self.summary_op = tf.summary.merge_all()

        # Init all vars
        init=tf.global_variables_initializer()
        sess.run(init)

    def _getPredictions(self,outputs):        
        with tf.name_scope('reshape_outputs'):
            outputs = tf.stack(outputs,name='restack')
            outputs = tf.transpose(outputs,[1,0,2],name='transpose')
            batch_size = tf.shape(outputs,name='batch_size')[0]

        with tf.name_scope('prediction_layer_1'):          
            W1 = tf.tile(self.W1,[batch_size,1,1])
            layer1 = tf.nn.sigmoid(tf.matmul(outputs,W1)+self.B1)

        with tf.name_scope('prediction_layer_2'):
            W2 = tf.tile(self.W2,[batch_size,1,1])
            predictions = tf.matmul(layer1,W2)+self.B2
            
        return predictions
    
    def _getY(self,x):
        with tf.name_scope('reverse_input'):
            y = tf.reverse(x,axis=[1])
        return y
    
    def _getLoss(self,x,predictions):
        y = self.y

        with tf.name_scope('loss_calculation'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss',loss)
        return loss

    def _getAccuracy(self,x,predictions):
        y = self.y

        with tf.name_scope('calculate_accuracy'):
            arg_maxes1 = tf.argmax(predictions,2)
            arg_maxes2 = tf.argmax(y,2)
            correct_predictions=tf.equal(arg_maxes1,arg_maxes2)
            accuracy=tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
            tf.summary.scalar('accuracy',accuracy)
        return accuracy

    def argmaxToString(self,v):
        s = ""
        for i in v:
            if i == self.UNK_TOKEN:
                s += ' '
            else:
                s += chr(i + ord('a'))
        
        s = s[::-1]
        return s

    def _getGrid(self,a,b,c):
        d = np.zeros([self.interp_steps+1,self.interp_steps+1,a.shape[0]])

        ab = b - a
        ac = c - a
        
        for i in range(self.interp_steps+1):
            t1 = i / self.interp_steps
            for j in range(self.interp_steps+1):
                t2 = j / self.interp_steps                
                d[i,j] = a + (t1 * ab + t2 * ac)
                
        return d
        
    
    def _getInterpolatedVectors(self,s,c):
        batch_size = (self.interp_steps + 1) ** 2
        s_m = np.reshape(self._getGrid(s[0],s[1],s[2]),[batch_size,self.state_size])
        c_m = np.reshape(self._getGrid(c[0],c[1],c[2]),[batch_size,self.state_size])
        return s_m,c_m
    
    def generate_text_summary(self,sess,strings):
        assert len(strings) == 3
        with tf.name_scope('text_summary'):
            # Encode given strings
            v = stringsToArray(strings)
            s,c = sess.run(self.encode_op,feed_dict={self.x_:v})

            # Interpolate in latent space
            s,c = self._getInterpolatedVectors(s,c)

            # Decode
            indices_batch = sess.run(self.decode_op,feed_dict={self.s_:s,self.c_:c})

            # Convert to string
            resultStrings = []
            for indices in indices_batch:
                resultStrings.append(self.argmaxToString(indices))
            resultStrings = np.reshape(np.array(resultStrings),[self.interp_steps+1,
                                                            self.interp_steps+1])

            # Assign to summary variable
            sess.run(self.text_variable.assign(resultStrings))
        print(resultStrings)

    def fit(self,sess,batch):
        sess.run(self.optimization_op, feed_dict={self.x_:batch})

    def eval_and_write_summaries(self,sess,batch,epoch):
        accuracy,loss,summary_output=sess.run([self.accuracy,self.loss,self.summary_op],
                                        feed_dict={self.x_:batch})
        self.summaryWriter.add_summary(summary_output,epoch)
        return accuracy,loss
