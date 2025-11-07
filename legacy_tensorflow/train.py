import tensorflow as tf
import numpy as np

from model import Seq2SeqAutoencoder

num_units = 128
time_steps = 10
input_size = 27

EPOCHS = 10000
BATCH_SIZE = 100
CHECKPOINT_EVERY = 250
UNK_TOKEN = 26

data = np.load("./preprocessed.npy")
DATA_SIZE = len(data)


def getBatch(size):
    idx = [np.random.randint(DATA_SIZE) for _ in range(size)]
    return data[idx]


with tf.Session() as sess:
    ae = Seq2SeqAutoencoder(sess, num_units, time_steps, input_size, UNK_TOKEN)
    saver = tf.train.Saver()
    for epoch in range(1, EPOCHS + 1):
        batch = getBatch(BATCH_SIZE)
        ae.fit(sess, batch)

        if epoch % CHECKPOINT_EVERY == 0:
            batch = getBatch(BATCH_SIZE)
            ae.generate_text_summary(sess, ["apple", "ball", "goat"])
            acc, los = ae.eval_and_write_summaries(sess, batch, epoch)
            print("%d\tA: %.4f\tL: %.4f" % (epoch, acc, los))
    saver.save(sess, "./checkpoints/model")

print("\a")
