import tensorflow as tf

tf.flags.DEFINE_float('a', 1, 'aa')
tf.flags.DEFINE_float('b', 2, 'bb')
tf.flags.DEFINE_float('c', 3, 'cc')
tf.flags.DEFINE_bool('ok',False,'okokokok')
FLAGS = tf.flags.FLAGS

def main(_):
    print(FLAGS.a)
    print(FLAGS.b)
    print(FLAGS.c)
    print(FLAGS.ok)	
if __name__ == '__main__':
    print(FLAGS.a)
    print(FLAGS.b)
    print(FLAGS.c)
    print(FLAGS.ok)	 
    print(" ")
    tf.app.run()  




#√¸¡Ó python kule555.py --ok False --a 666