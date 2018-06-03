import numpy as np
import tensorflow as tf

print("loading FaceNet")
tf.reset_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph('./20180408-102900/model-20180408-102900.meta')
saver.restore(sess, './20180408-102900/model-20180408-102900.ckpt-90')
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

def get_vec(img):
    img = img.astype(np.float32)/255
    return np.array(
        sess.run([embeddings],
                 feed_dict=
                 {images_placeholder: img[None], # add dimensionality
                  phase_train_placeholder: False }
                )[0])
print("loading completed")
