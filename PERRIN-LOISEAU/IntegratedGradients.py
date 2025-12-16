################################################################
# Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
# Modified for TensorFlow 2.x compatibility                    #
#                                                              #
# Keras-compatible implmentation of Integrated Gradients       #
# proposed in "Axiomatic attribution for deep neuron networks" #
# (https://arxiv.org/abs/1703.01365).                          #
#                                                              #
# Keywords: Shapley values, interpretable machine learning     #
################################################################
from __future__ import division, print_function
import numpy as np
from time import sleep
import sys
import tensorflow as tf

from keras.models import Model, Sequential

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements this concept.
'''


class integrated_gradients:
  # model: Keras model that you wish to explain.
  # outchannels: In case the model are multi tasking, you can specify which channels you want.
  def __init__(self, model, outchannels=[], verbose=1):

    # load model supports keras.Model and keras.Sequential
    if isinstance(model, Sequential):
      self.model = model.model
    elif isinstance(model, Model):
      self.model = model
    else:
      print("Invalid input model")
      return -1

    # If outputchanel is specified, use it.
    # Otherwise evalueate all outputs.
    self.outchannels = outchannels
    if len(self.outchannels) == 0:
      if verbose: print("Evaluated output channel (0-based index): All")
      self.outchannels = range(int(self.model.output.shape[1]))
    else:
      if verbose:
        print("Evaluated output channels (0-based index):")
        print(','.join([str(i) for i in self.outchannels]))

    if verbose: print("Building gradient functions")
    if verbose: print("Progress: 100.0%")
    if verbose: print("\nDone.")

  '''
  Compute gradients using GradientTape (TensorFlow 2.x)
  '''
  def _compute_gradients(self, inputs, outc):
    inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(inputs_tensor)
      outputs = self.model(inputs_tensor)
      if len(outputs.shape) > 1:
        target_output = outputs[:, outc]
      else:
        target_output = outputs
    gradients = tape.gradient(target_output, inputs_tensor)
    return gradients.numpy()

  '''
  Input: sample to explain, channel to explain
  Optional inputs:
      - reference: reference values (defaulted to 0s).
      - steps: # steps from reference values to the actual sample.
  Output: list of numpy arrays to integrated over.
  '''

  def explain(self, sample, outc=0, reference=False, num_steps=50, verbose=0):

    # Each element for each input stream.
    samples = []
    numsteps = []
    step_sizes = []

    # If multiple inputs are present, feed them as list of np arrays.
    if isinstance(sample, list):
      # If reference is present, reference and sample size need to be equal.
      if reference != False:
        assert len(sample) == len(reference)
      for i in range(len(sample)):
        if reference == False:
          _output = integrated_gradients.linearly_interpolate(sample[i], False, num_steps)
        else:
          _output = integrated_gradients.linearly_interpolate(sample[i], reference[i], num_steps)
        samples.append(_output[0])
        numsteps.append(_output[1])
        step_sizes.append(_output[2])

    # Or you can feed just a single numpy arrray.
    elif isinstance(sample, np.ndarray):
      _output = integrated_gradients.linearly_interpolate(sample, reference, num_steps)
      samples.append(_output[0])
      numsteps.append(_output[1])
      step_sizes.append(_output[2])

    # Desired channel must be in the list of outputchannels
    assert outc in self.outchannels
    if verbose: print("Explaining the " + str(outc) + "th output.")
    # Compute gradients for all interpolated samples
    explanation = []
    for i in range(len(samples)):
      gradients = self._compute_gradients(samples[i], outc)
      _temp = np.sum(gradients, axis=0)
      explanation.append(np.multiply(_temp, step_sizes[i]))

    if isinstance(sample, list):
      return explanation
    elif isinstance(sample, np.ndarray):
      return explanation[0]
    return -1

  '''
  Input: numpy array of a sample
  Optional inputs:
      - reference: reference values (defaulted to 0s).
      - steps: # steps from reference values to the actual sample.
  Output: list of numpy arrays to integrated over.
  '''

  @staticmethod
  def linearly_interpolate(sample, reference=False, num_steps=50):
    # Use default reference values if reference is not specified
    if reference is False: reference = np.zeros(sample.shape);

    # Reference and sample shape needs to match exactly
    assert sample.shape == reference.shape

    # Calcuated stepwise difference from reference to the actual sample.
    ret = np.zeros(tuple([num_steps] + [i for i in sample.shape]))
    for s in range(num_steps):
      ret[s] = reference + (sample - reference) * (s * 1.0 / num_steps)

    return ret, num_steps, (sample - reference) * (1.0 / num_steps)