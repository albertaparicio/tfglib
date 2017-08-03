# Original code from 'yauheni_selivonchyk' at StackOverflow
# http://stackoverflow.com/a/42815564
# MIT License

import matplotlib.pyplot as plt
import numpy as np


def get_figure():
  fig = plt.figure(num=0, figsize=(6, 4), dpi=300)
  fig.clf()
  return fig


def fig2rgb_array(fig, expand=True):
  fig.canvas.draw()
  buf = fig.canvas.tostring_rgb()
  ncols, nrows = fig.canvas.get_width_height()
  shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
  return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def figure_to_summary(fig, summary_writer, vis_summary, vis_placeholder):
  image = fig2rgb_array(fig)
  summary_writer.add_summary(
      vis_summary.eval(feed_dict={vis_placeholder: image}))


  # if __name__ == '__main__':
  #       # construct graph
  #       x = tf.Variable(initial_value=tf.random_uniform((2, 10)))
  #       inc = x.assign(x + 1)
  #
  #       # construct summary
  #       fig = get_figure()
  #       vis_placeholder = tf.placeholder(tf.uint8, fig2rgb_array(fig).shape)
  #       vis_summary = tf.summary.image('custom', vis_placeholder)
  #
  #       with tf.Session() as sess:
  #         tf.global_variables_initializer().run()
  #         summary_writer = tf.summary.FileWriter('./tmp', sess.graph)
  #
  #         for i in range(100):
  #           # execute step
  #           _, values = sess.run([inc, x])
  #           # draw on the plot
  #           fig = get_figure()
  #           plt.subplot('111').scatter(values[0], values[1])
  #           # save the summary
  #           figure_to_summary(fig)
