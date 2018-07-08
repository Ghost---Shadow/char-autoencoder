import tensorflow as tf

from model import Seq2SeqAutoencoder

sess = tf.InteractiveSession()
ae = Seq2SeqAutoencoder(sess)
saver = tf.train.Saver()
saver.restore(sess,'./checkpoints/model')
strings = [['apple','ball','goat'],
           ['identity','identity','identity'],
           ['baseball','ball','base'],
           ['cabbage','cab','cabin'],
           ['abc','def','ghi'],
           ['toooot','toooooooot','tot'],
           ['ball ball','cat cat','dog dog']]
           
for s in strings:           
    ae.generate_text_summary(sess,s)
    print()
