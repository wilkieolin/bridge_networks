"""
This file defines the control models used to compare performance of the bridge network
without VSA ops.

Wilkie Olin-Ammentorp, 2021
University of Califonia, San Diego
"""

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as be
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MSE

from data import get_raw_dat
from layers import *
from utils import *
from models import *

"""
Simple dense scaling module.
"""
def DenseScale(n_in, n_hidden, n_out, regularize=True):
    inputs = keras.layers.Input((n_in,))
    encoder = keras.layers.Dense(n_hidden, activation="relu")
    dropout = keras.layers.Dropout(0.5)
    decoder = keras.layers.Dense(n_out, activation="relu")

    if regularize:
        self = keras.Sequential([inputs, encoder, dropout, decoder])
    else:
        self = keras.Sequential([inputs, encoder, decoder])

    return self

"""
Residual projection channel, transforms an input which isn't a VSA symbol.
"""
class ResChannel(keras.Model):
    def __init__(self, n_in, n_hidden, n_out):
        super(ResChannel, self).__init__()
        
        initializer = tf.initializers.glorot_normal()
        
        self.fwd = DenseScale(n_in, n_hidden, n_out)
        # self.proj = tf.Variable(
        #     shape=(n_in, n_out),
        #     initial_value = initializer((n_in, n_out)),
        #     trainable = False)
        
        self.rvs = DenseScale(n_out, n_hidden, n_in)
        # self.rvs_proj = tf.Variable(
        #     shape=(n_out, n_in),
        #     initial_value = initializer((n_out, n_in)),
        #     trainable = False)
        
    def call(self, inputs):
        #forward
        fwd = self.forward(inputs)
        
        #reverse
        rvs = self.reverse(fwd)
        
        return fwd, rvs

    def forward(self, inputs):
        #signal = tf.matmul(inputs, self.proj)
        noise = self.fwd(inputs)
        fwd = inputs + noise
        return fwd

    def reverse(self, inputs):
        denoise = self.rvs(inputs)
        #recovered = tf.matmul(inputs, self.rvs_proj)
        rvs = denoise + inputs
        return rvs

def StandardModel(conv_head, ds_info, **kwargs):
    input_shape = ds_info.features['image'].shape
    n_h = kwargs.get("hidden", 100)

    input0 = keras.layers.Input(shape=input_shape)
    flatten0 = keras.layers.Flatten()
    dense0 = keras.layers.Dense(n_h)


"""
Control version of Bridge model. Uses ReLUs and addition instead of FHRR ops.
"""
class ControlBridgeModel(keras.Model):
    def __init__(self, conv_head, inv_head, ds_info, **kwargs):
        super(ControlBridgeModel, self).__init__()
        
        #set constants for the dataset: image shape and total # of classes
        self.img_shape = ds_info.features['image'].shape
        self.n_classes = ds_info.features['label'].num_classes
        #optionally overried with keyword arg
        self.n_classes = kwargs.get("n_classes", self.n_classes)

        #the dimension of symbols
        self.n_d = kwargs.get("n_d", 1000)
        self.overscan = kwargs.get("overscan", 100.0)
        #the number of hidden neurons in autoencoders
        self.n_hidden = kwargs.get("n_hidden", 100)
        self.n_batch = kwargs.get("n_batch", 128)

        #the image input layer
        self.image_input = keras.Input(shape=self.img_shape)
        #the class input layer
        self.class_input = keras.Input(shape=(self.n_d,))
        #the feature detector (pre-trained)
        self.head = conv_head
        self.head.trainable = False
        #the image reconstructor
        self.inv_head = inv_head
        self.inv_head.trainable = False

        self.image_ch = ResChannel(self.n_d, self.n_hidden, self.n_d)
        self.label_ch = ResChannel(self.n_d, self.n_hidden, self.n_d)
        self.loss_fn = lambda x, y: tf.reduce_mean(MSE(x, y))
        #label to symbol encoder
        self.label_encoder = LabelEncoder(self.n_d, self.n_classes, self.overscan)
        #reassign weights to produce positive normally-distributed values
        self.label_weights = self.label_encoder.encoder.w
        self.label_weights.assign(tf.abs(tf.random.normal(shape=self.label_weights.shape)))

        self.image_encoder = keras.Sequential([StaticLinear(conv_head.output_shape[1], self.n_d, self.overscan),
                                                layers.Activation(tf.math.abs)])


    """
    Accuracy of the network at classification. Runs data from image over to class.
    Confusion: return full confusion matrix.
    Distance: return the average distance between each image symbol and all class symbols.
    """
    def accuracy(self, test_loader, confusion=False,  similarity=False):
        guesses = np.zeros((self.n_classes, self.n_classes), dtype=np.int)

        if similarity:
            samples = np.zeros_like(guesses, dtype=np.float)

        for data in test_loader:
            x, y = data
            ns = x.shape[0]

            #what are the predicted innermost vectors from the image
            x_sym = self.convert_img(x)
            xy_sym = self.image_ch.forward(x_sym)
            #relate these back to the class
            yh_sym = self.label_ch.reverse(xy_sym)
            #get the class symbols
            y_sym = self.label_encoder.encoder.weights[0]
            get_sims = lambda x: MSE(tf.broadcast_to(x, (self.n_classes, self.n_d)), y_sym)
            #find the class the predicted symbol is closest to
            sim = tf.map_fn(get_sims, yh_sym)
            yh = tf.argmin(sim, axis=1)
            
            for i in range(ns):
                guesses[y[i], yh[i]] += 1

                if similarity:
                    samples[y[i],:] += sim[i,:].numpy()

        rvals = []
        total = tf.math.reduce_sum(guesses)

        if confusion:
            rvals.append(guesses)
        else:
            correct = tf.math.reduce_sum(tf.linalg.diag_part(guesses))
            rvals.append(correct / total)

        if similarity:
            rvals.append(samples / tf.math.reduce_sum(guesses, axis=1))

        return tuple(rvals)

    """
    Given an image and class, return the predicted innermost symbols from both inputs.
    """
    def call(self, x, y, **kwargs):
        x_vectors = self.convert_img(x, **kwargs)
        y_vectors = self.convert_label(y)

        x_guess = self.image_ch.forward(x_vectors)
        y_guess = self.label_ch.forward(y_vectors)

        return x_guess, y_guess

    """
    Given an image, convert it to a feature symbol.
    """
    def convert_img(self, images, **kwargs):
        features = self.head(images)
        vectors = self.image_encoder(features)
        return vectors

    """
    Given a label, convert it to a label symbol.
    """
    def convert_label(self, labels, **kwargs):
        vectors = self.label_encoder(labels)
        return vectors

    """
    Method used to create the true image-label vectors when both inputs are present.
    Bind the image with its role, the label with its role, and bundle the two.
    """
    def forward(self, img_vec, lbl_vec):
        h_vectors = img_vec + lbl_vec

        return h_vectors

    """
    Get the self's internal values so they can be stored and inspected or used
    later for distillation.
    """
    def get_internals(self, dataloader):
        x_vecs = []
        y_vecs = []
        hx_vecs = []
        hy_vecs = []
        htrue_vecs = []
        values = {}

        for (x,y) in dataloader:
            xv = self.convert_img(x)
            yv = self.convert_label(y)
            hx = self.image_ch.forward(xv)
            hy = self.label_ch.forward(yv)
            htrue = self.forward(xv, yv)

            x_vecs.append(xv)
            y_vecs.append(yv)
            hx_vecs.append(hx)
            hy_vecs.append(hy)
            htrue_vecs.append(htrue)

        values['x_vecs'] = tf.concat(x_vecs, axis=0)
        values['y_vecs'] = tf.concat(y_vecs, axis=0)
        values['hx_vecs'] = tf.concat(hx_vecs, axis=0)
        values['hy_vecs'] = tf.concat(hy_vecs, axis=0)
        values['htrue_vecs'] = tf.concat(htrue_vecs, axis=0)

        return values

    def get_losses(self, dataloader):

        makearr = lambda: [[] for i in range(self.n_classes)]
        x_fwd = makearr()
        y_fwd = makearr()
        x_rvs = makearr()
        y_rvs = makearr()

        for (x,y) in dataloader:
            xv = self.convert_img(x)
            yv = self.convert_label(y)
            hx = self.image_ch.forward(xv)
            hy = self.label_ch.forward(yv)
            htrue = self.forward(xv, yv)
            xvr = self.image_ch.reverse(htrue)
            yvr = self.label_ch.reverse(htrue)

            for (i,yi) in enumerate(y):
                yi = yi.numpy().item()

                samplef = lambda x, y: MSE(x[i:i+1,:], y[i:i+1,:])
                xf = samplef(hx, htrue)
                xr = samplef(xvr, xv)
                yf = samplef(hy, htrue)
                yr = samplef(yvr, yv)

                x_fwd[yi].append(xf)
                y_fwd[yi].append(yf)
                x_rvs[yi].append(xr)
                y_rvs[yi].append(yr)


        concat = lambda y : list(map(lambda x: np.array(x), y))
        x_fwd = concat(x_fwd)
        y_fwd = concat(y_fwd)
        x_rvs = concat(x_rvs)
        y_rvs = concat(y_rvs)

        results = {}
        results["x_fwd"] = x_fwd
        results["y_fwd"] = y_fwd
        results["x_rvs"] = x_rvs
        results["y_vrs"] = y_rvs

        return results

    """
    Given a set of true external data, generate symbols necessary for training
    """
    def generate_external(self, dataset, from_batched=False):

        if from_batched:
            xs = []
            ys = []
            hs = []

            for data in iter(dataset):
                images, labels = data
                x = self.convert_img(images)
                y = self.convert_label(labels)
                h = self.forward(x, y)

                xs.append(x)
                ys.append(y)
                hs.append(h)

            cfunc = lambda x: tf.concat(x, axis=0)
            x = cfunc(xs)
            y = cfunc(ys)
            h = cfunc(hs)

            return (x, y, h, h, h)

        else:
            images, labels = dataset
            x = self.convert_img(images)
            y = self.convert_label(labels)
            h = self.forward(x, y)

            return (x, y, h, h, h)


    """
    Given a set of class-image symbols, reconstruct the image data from these symbols.
    """
    def reconstruct(self, center_symbols):
        image_symbols = self.image_ch.reverse(center_symbols)
        features = self.img_encoder.reverse(image_symbols)
        images = self.inv_head(features)

        return images

    """
    Given a set of class-image symbols, approximate the inverse of the forward process
    (find input vectors which approximate the likely inputs to create the img-label symbol).
    """
    def reverse(self, center_symbols):
        img_symbol_recon = self.image_ch.reverse(center_symbols)
        lbl_symbol_recon = self.label_ch.reverse(center_symbols)

        return (img_symbol_recon, lbl_symbol_recon)

    """
    Conventional training of the network when external inputs are present at all senses.
    Minimize the conversion and reconstruction losses for all layers & calibrate the BN stage.
    """
    def train_interleaved(self, external_dataset, internal_dataset, batches, split=[127, 1], report_interval=100, generate_internal=False):
        #load the external data
        n_s = external_dataset.cardinality()
        external_dataset = external_dataset.shuffle(n_s)
        external_dataset = external_dataset.batch(split[1])
        external_dataset = external_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        ext_iter = iter(external_dataset)

        if generate_internal:
            int_dataset = self.generate_internal(internal_dataset)
        else:
            int_dataset = internal_dataset

        losses = []
        for b in range(batches):
            try:
                ext_data = next(ext_iter)
            except StopIteration:
                ext_iter = iter(external_dataset)
                ext_data = next(ext_iter)

            ext_data = self.generate_external(ext_data)
            int_data = cut(int_dataset, split[0], rand_sel=True)

            concat = lambda x: tf.concat(x, axis=0)

            interleaved = [concat((int_data[i], ext_data[i])) for i in range(len(int_data))]
            x, y, hx, hy, hp = interleaved

            loss = self.train_step(x, y, hx, hy, hp)

            if b % report_interval == 0:
                print("Training loss", loss)

            losses.append(loss)        

        return np.array(losses), int_dataset

    #unified training method for mixed internal/external data, only takes internal symbols
    #methods accessing train need to manage/update cache and keep track of true/generated data
    def train(self, symbols, epochs, report_interval=100, energy=False):
        losses = []

        symbols = tf.data.Dataset.from_tensor_slices(symbols)
        loader = symbols.batch(self.n_batch)
        loader = loader.prefetch(tf.data.experimental.AUTOTUNE)

        for _ in range(epochs):
            for step, data in enumerate(loader):
                x, y, hx, hy, hp = data
                loss = self.train_step(x, y, hx, hy, hp)
                losses.append(loss)

                if step % report_interval == 0:
                    print("Training loss", loss)

        return np.array(losses)
    
    """
    Carry out the individual steps of training given the symbols at all stages,
    generated from external and/or internal data.
    """
    def train_step(self, x, y, hx, hy, hp):
        makevar = lambda x: tf.Variable(x)
        x = makevar(x)
        y = makevar(y)
        hx = makevar(hx)
        hy = makevar(hy)
        hp = makevar(hp)

        with tf.GradientTape() as tape:
            #make predictions through the channels
            #forward
            hx_hat = self.image_ch.forward(x)
            hy_hat = self.label_ch.forward(y)
            #reverse
            x_hat = self.image_ch.reverse(hp)
            y_hat = self.label_ch.reverse(hp)

            #tally the losses
            #forward
            fwd_x = self.loss_fn(hx, hx_hat)
            rvs_x = self.loss_fn(x, x_hat)
            #reverse
            fwd_y = self.loss_fn(hy, hy_hat)
            rvs_y = self.loss_fn(y, y_hat)
            losses = [fwd_x, rvs_x, fwd_y, rvs_y]
            loss = tf.math.reduce_mean(losses)

        #trainable_vars = self.trainable_variables
        trainable_vars = [*self.image_ch.trainable_variables, *self.label_ch.trainable_variables]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return np.array(losses)