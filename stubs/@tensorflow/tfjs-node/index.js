'use strict';

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu');

if (typeof tf.getBackend === 'function' && tf.getBackend() !== 'cpu') {
  try {
    tf.setBackend('cpu');
  } catch (err) {
    // ignore if backend cannot be changed synchronously
  }
}

if (typeof tf.ready === 'function') {
  tf.ready().catch(() => {});
}

module.exports = tf;
