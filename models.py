import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as be
from tensorflow.keras import regularizers

from data import get_raw_dat, cut
from layers import *
from utils import *

"""
Simple Autoencoder self with 1 hidden layer of n_hidden neurons with dropout
"""
def AutoEncoder(n_d, n_hidden, regularize=True):
    inputs = keras.layers.Input((n_d,))
    encoder = CmpxLinear(n_hidden)
    dropout = keras.layers.Dropout(0.25)
    decoder = CmpxLinear(n_d)

    if regularize:
        self = keras.Sequential([inputs, encoder, dropout, decoder])
    else:
        self = keras.Sequential([inputs, encoder, decoder])

    return self


"""
A "channel" which can transport data between 'layers' of an bridge net. Will take an outer input, 
bind it to its role, and add the expected noise. Can also learn to do the reverse operation.
"""
class BindChannel(keras.Model):
    def __init__(self, n_d, n_hidden):
        super(BindChannel, self).__init__()
        
        self.fwd_ae = AutoEncoder(n_d, n_hidden)
        self.rvs_ae = AutoEncoder(n_d, n_hidden)
        
        self.role = make_codebook(1, n_d)
        
    def call(self, inputs):
        #forward
        fwd = self.forward(inputs)
        
        #reverse
        rvs = self.reverse(fwd)
        
        return fwd, rvs

    def forward(self, inputs):
        signal = bind(inputs, self.role)
        noise = self.fwd_ae(inputs)
        fwd = signal + noise
        fwd = remap_phase(fwd)
        return fwd

    def reverse(self, inputs):
        denoise = self.rvs_ae(inputs)
        recovered = unbind(inputs, self.role)
        rvs = denoise + recovered
        rvs = remap_phase(rvs)
        return rvs

"""
A "channel" which can transport data between 'layers' of an bridge net. This one does not bind data
to a static role.
"""
class Channel(keras.Model):
    def __init__(self, n_d, n_hidden):
        super(Channel, self).__init__()
        
        self.fwd_ae = AutoEncoder(n_d, n_hidden)
        self.rvs_ae = AutoEncoder(n_d, n_hidden)
        
    def call(self, inputs):
        #forward
        fwd = self.forward(inputs)
        
        #reverse
        rvs = self.reverse(fwd)
        
        return fwd, rvs

    def forward(self, inputs):
        signal = inputs
        noise = self.fwd_ae(inputs)
        fwd = signal + noise
        fwd = remap_phase(fwd)
        return fwd

    def reverse(self, inputs):
        denoise = self.rvs_ae(inputs)
        recovered = inputs
        rvs = denoise + recovered
        rvs = remap_phase(rvs)
        return rvs

"""
Convolutional autoencoder which is pretrained before the bridge network.
Its encoder and decoder are then used to convert images into features for
the bridge network to learn on new datasets. 
"""
class Conv_AE(keras.Model):
    def __init__(self, img_shape, n_classes, **kwargs):
        super(Conv_AE, self).__init__()
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        
        #The encoder (image -> feature)
        self.conv_head = keras.Sequential(
            [
                keras.Input(shape=self.img_shape),
                layers.BatchNormalization(),
                layers.Conv2D(32, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(self.weight_decay), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.5),

                layers.Conv2D(64, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(self.weight_decay), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.3),
                layers.Flatten()
            ]
        )
        self.conv_head.compile()

        self.n_dim = self.conv_head.layers[-1].output_shape[1]
        
        #The classification layer at the middle of the AE
        self.classifier = layers.Dense(self.n_classes, activation="softmax")
        
        #The decoder (feature -> image)
        self.inv_head = keras.Sequential(
            [
                keras.Input(shape=self.conv_head.output_shape[1:]),
                layers.Reshape((5,5,64)),
                layers.Dropout(0.3),
                layers.UpSampling2D((2,2)),
                layers.Conv2DTranspose(32, kernel_size=(4, 4), kernel_regularizer=regularizers.l2(self.weight_decay), activation="relu"),
                layers.Dropout(0.5),
                layers.UpSampling2D((2, 2)),
                layers.Conv2DTranspose(1, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(self.weight_decay), activation="sigmoid")
            ]
        )
        self.inv_head.compile()

        self.compile(optimizer="rmsprop")
    """
    Return the AE's classification accuracy.
    """
    def accuracy_class(self, test_loader):
        correct = 0
        total = 0
        
        for data in test_loader:
            x, y = data
            n_s = y.shape[0]
            total += n_s
            
            yh = self.classify(x)
            yh = keras.backend.argmax(yh)
            agree = yh == y
            agree = tf.cast(agree, tf.int32)
            correct += int(tf.math.reduce_sum(agree))
            
        return correct/total
    
    """
    Return the predicted class for an input tensor of images.
    """
    def classify(self, data):
        features = self.conv_head(data)
        yh = self.classifier(features)
        #yh = keras.backend.argmax(yh)
        return yh
    
    """
    Return the model configuration to allow it to be saved.
    """
    def get_config(self):
        return {"conv_head": self.conv_head,
               "classifier": self.classifier,
               "inv_head": self.inv_head}
    
    """
    Return images after they have been passed (compressed) through the AE.
    """
    def reconstruct(self, data):
        features = self.conv_head(data)
        images = self.inv_head(features)
        return images
    
    """
    Training step for the optimization process. Reduce classification error at the middle
    and reconstruction error between input / reconstruction.
    """
    def train_step(self, inputs):
        x, y = inputs
        
        fwd_lossfn = keras.losses.SparseCategoricalCrossentropy()
        rvs_lossfn = keras.losses.BinaryCrossentropy()
        
        with tf.GradientTape() as tape:
            feat = self.conv_head(x)
            yh = self.classifier(feat)
            fwd_loss = fwd_lossfn(y, yh)
            
            xh = self.inv_head(feat)
            rvs_loss = rvs_lossfn(x, xh)
            loss = tf.reduce_mean([fwd_loss, rvs_loss])
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss
    
    """
    Train the AE model given a dataset. 
    """
    def train(self, train_loader, epochs, report_interval=100):
        losses = []

        for epoch in range(epochs):
            for step, data in enumerate(train_loader):
                loss = self.train_step(data).numpy()
                losses.append(loss)

                if step % report_interval == 0:
                    print("Training loss", loss)

        return losses


"""
Given a standard int input, convert it to a VSA symbol representing that class.
"""
class LabelEncoder(keras.Model):
    def __init__(self, n_d, n_classes, overscan=1.0, noise=0.0):
        super(LabelEncoder, self).__init__()
        self.onehot = lambda l: be.one_hot(l, n_classes)
        self.encoder = StaticLinear(n_classes, n_d, overscan)
        self.noise = noise
        if noise > 0.0:
            self.noisy = True
        else:
            self.noisy = False

    def decode(self, inputs):
        x = self.encoder.reverse(inputs)
        return x

    def call(self, inputs):
        x = self.onehot(inputs)
        x = self.encoder(x)
        if self.noisy:
            shp = x.shape
            x = x + tf.random.normal(shape=shp, stddev=self.noise)
            x = remap_phase(x)
        
        return x

"""
Converts a vector of features from a detector (e.g. convolutional net, ResNet) and projects/
normalizes them into a VSA symbol.
"""
class VSA_Encoder(keras.Model):
    def __init__(self, n_in, n_out, overscan=1.0, sigma=3.0, name='VSA Encoder'):
        super(VSA_Encoder, self).__init__()
        self.transform = StaticLinear(n_in, n_out, overscan)
        self.norm_symbols = Normalize(sigma)
      
    def call(self, inputs, **kwargs):
        x = self.transform(inputs)
        x = self.norm_symbols(x, **kwargs)
        x = tf.clip_by_value(x, -1.0, 1.0)
        return x

    def reverse(self, inputs):
        x = self.norm_symbols.reverse(inputs)
        x = self.transform.reverse(x)

        return x                
            
"""
"Iris" model of image classification. Takes intputs from two senses (vision, hearing) and learns how
the invariants between these data relate to one another. 
"""
class IrisModel(keras.Model):
    def __init__(self, conv_head, inv_head, ds_info, **kwargs):
        super(IrisModel, self).__init__()
        
        #set constants for the dataset: image shape and total # of classes
        self.img_shape = ds_info.features['image'].shape
        self.n_classes = ds_info.features['label'].num_classes
        #optionally overried with keyword arg
        self.n_classes = kwargs.get("n_classes", self.n_classes)
        #the normalization constant used in the image encoder. 
        self.sigma = kwargs.get("sigma", 3.0)
        #the dimension of VSA symbols
        self.n_d = kwargs.get("n_d", 1000)
        self.overscan = kwargs.get("overscan", 100.0)
        #the number of hidden neurons in autoencoders
        self.n_hidden = kwargs.get("n_hidden", 100)
        self.n_batch = kwargs.get("n_batch", 128)
        self.bind = kwargs.get("bind", False)
        self.noise = kwargs.get("noise", 0.0)

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
        #feature to symbol encoder
        self.img_encoder = VSA_Encoder(self.head.output_shape[1], self.n_d, self.overscan, self.sigma)
        #symbol to innermost symbol channels
        if self.bind:
            self.image_ch = BindChannel(self.n_d, self.n_hidden)
            self.label_ch = BindChannel(self.n_d, self.n_hidden)
        else:
            self.image_ch = Channel(self.n_d, self.n_hidden)
            self.label_ch = Channel(self.n_d, self.n_hidden)
        #label to symbol encoder
        self.label_encoder = LabelEncoder(self.n_d, self.n_classes, self.overscan, noise=self.noise)
        #symbol to innermost symbol encoder
        #replay memory
        self.cache = tf.zeros((0, self.n_d), dtype=tf.float32)
        self.cache_limit = kwargs.get("cache_limit", 10000)

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
            #find the class the predicted symbol is closest to
            sim = similarity_outer(yh_sym, y_sym)
            yh = tf.argmax(sim, axis=1)
            
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
        vectors = self.img_encoder(features, **kwargs)
        return vectors

    """
    Given a label, convert it to a label symbol.
    """
    def convert_label(self, labels, **kwargs):
        vectors = self.label_encoder(labels)
        return vectors

    def distill(self, symbols, epochs, report_interval=100):
        losses = []

        symbols = tf.data.Dataset.from_tensor_slices(symbols)
        loader = symbols.batch(self.n_batch)
        loader = loader.prefetch(tf.data.experimental.AUTOTUNE)

        for _ in range(epochs):
            for step, data in enumerate(loader):
                x, y, _, _, hp = data
                loss = self.distill_step(x, y, hp)
                losses.append(loss)

                if step % report_interval == 0:
                    print("Training loss", loss)

        return np.array(losses)
    
    """
    Carry out the individual steps of distillation given hprime and x/y symbols.
    """
    def distill_step(self, x, y, hp):
        makevar = lambda x: tf.Variable(x)
        x = makevar(x)
        y = makevar(y)
        hp = makevar(hp)

        with tf.GradientTape() as tape:
            #make predictions through the channels
            #reverse
            x_hat = self.image_ch.reverse(hp)
            y_hat = self.label_ch.reverse(hp)

            #tally the losses
            #reverse
            rvs_x = vsa_loss(x, x_hat)
            rvs_y = vsa_loss(y, y_hat)
            losses = [rvs_x, rvs_y]
            loss = tf.math.reduce_mean(losses)

        #trainable_vars = self.trainable_variables
        trainable_vars = [*self.image_ch.rvs_ae.trainable_variables, *self.label_ch.rvs_ae.trainable_variables]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return np.array(losses)


    """
    Method used to create the true image-label vectors when both inputs are present.
    Bind the image with its role, the label with its role, and bundle the two.
    """
    def forward(self, img_vec, lbl_vec):
        if self.bind:
            img_vec = bind(img_vec, self.image_ch.role)
            lbl_vec = bind(lbl_vec, self.label_ch.role)

        h_vectors = bundle(img_vec, lbl_vec)
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

                samplef = lambda x, y: vsa_loss(x[i:i+1,:], y[i:i+1,:])
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
                x = self.convert_img(images, training=True)
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
            x = self.convert_img(images, training=True)
            y = self.convert_label(labels)
            h = self.forward(x, y)

            return (x, y, h, h, h)

    """
    Given any internal symbols, generate corresponding values
    """
    def generate_internal(self, inner_symbols, truefwd=False):
        x = self.image_ch.reverse(inner_symbols)
        y = self.label_ch.reverse(inner_symbols)

        if truefwd:
            hx = self.forward(x, y)
            hy = hx
            # hx = inner_symbols
            # hy = inner_symbols
        else:
            hx = self.image_ch.forward(x)
            hy = self.label_ch.forward(y)

        return (x, y, hx, hy, inner_symbols)

    def random_samples(self, n_s):
        transform = lambda x: x * 2.0 - 1.0
        return transform(tf.random.uniform((n_s, self.n_d)))

    """
    Given a desired number of samples, distill a number of center symbols which 
    can be used to retrain/preserve the current model.
    """
    def reconstruct_info(self, n_s, epochs, report_interval=100, **kwargs):
        symbols = tf.Variable(self.random_samples(n_s))

        losses = []
        for i in range(epochs):
            loss = self.reconstruct_info_step(symbols, **kwargs)
            losses.append(np.array(loss))

            if i % report_interval == 0:
                print(losses[-1])

        return symbols.value(), np.array(losses)
        

    def reconstruct_rbm(self, n_s, steps):
        symbols = self.random_samples(n_s)

        losses = []
        for i in range(steps):
            x = self.image_ch.reverse(symbols)
            y = self.label_ch.reverse(symbols)
            hx = self.image_ch.forward(x)
            hy = self.label_ch.forward(y)
            hp = bundle(hx, hy)

            loss = vsa_loss(symbols, hp)
            losses.append(np.array(loss))

            symbols = hp


        return symbols, np.array(losses)
        
    def reconstruct_info_step(self, h0, optimizer=None):

        #option to use external optimizer other than model default
        if optimizer is None:
            optimizer = self.optimizer

        with tf.GradientTape() as tape:
            x = self.image_ch.reverse(h0)
            y = self.label_ch.reverse(h0)

            hx = self.image_ch.forward(x)
            hy = self.label_ch.forward(y)

            losses = []

            #distillation examples should provide the locations where the
            # loss between forward/backward passes must be low
            x_loop = vsa_loss(h0, hx)
            y_loop = vsa_loss(h0, hy)

            losses.append(x_loop)
            losses.append(y_loop)

            # loss_fn = lambda x, y: 1 - tf.math.abs(vsa_loss(x,y))

            # #create a lower loss the more different the example is after passing back
            # #through the channels
            # if rvs_info:
            #     x_back = loss_fn(h0, x)
            #     y_back = loss_fn(h0, y)
            #     losses.append(x_back)
            #     losses.append(y_back)

            # #create a lower loss the more different the example is after passing forward
            # #through the channels
            # if fwd_info:
            #     x_fwd = loss_fn(hx, x)
            #     y_fwd = loss_fn(hy, y)
            #     losses.append(x_fwd)
            #     losses.append(y_fwd)
 
            loss = tf.reduce_mean(losses)

        trainable_vars = [h0]
        gradients = tape.gradient(loss, trainable_vars)

        optimizer.apply_gradients(zip(gradients, trainable_vars))

        return losses

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
        int_dataset = tf.data.Dataset.from_tensor_slices(int_dataset)
        int_dataset = int_dataset.batch(split[0])
        int_dataset = int_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        int_iter = iter(int_dataset)

        losses = []
        for b in range(batches):
            #get the external data
            try:
                ext_data = next(ext_iter)
            except StopIteration:
                ext_iter = iter(external_dataset)
                ext_data = next(ext_iter)

            ext_data = self.generate_external(ext_data)

            #get the internal data
            try:
                int_data = next(int_iter)
            except StopIteration:
                int_iter = iter(int_dataset)
                int_data = next(int_iter)

            #interleave the sets of symbols
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
                if energy:
                    loss = self.train_step_energy(x, y, hx, hy, hp)
                else:
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
            fwd_x = vsa_loss(hx, hx_hat)
            rvs_x = vsa_loss(x, x_hat)
            #reverse
            fwd_y = vsa_loss(hy, hy_hat)
            rvs_y = vsa_loss(y, y_hat)
            losses = [fwd_x, rvs_x, fwd_y, rvs_y]
            loss = tf.math.reduce_mean(losses)

        #trainable_vars = self.trainable_variables
        trainable_vars = [*self.image_ch.trainable_variables, *self.label_ch.trainable_variables]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return np.array(losses)

    """
    Carry out the individual steps of training given the symbols at all stages,
    generated from external and/or internal data.
    """
    def train_step_energy(self, x, y, hx, hy, hp):
        makevar = lambda x: tf.Variable(x)
        n_s = x.shape[0]

        #positive examples
        x = makevar(x)
        y = makevar(y)
        hx = makevar(hx)
        hy = makevar(hy)
        hp = makevar(hp)

        #negative examples
        hpn = self.random_samples(n_s)
        hpn = makevar(hpn)

        with tf.GradientTape() as tape:
            #positive examples
            #make predictions through the channels
            #forward
            hx_hat = self.image_ch.forward(x)
            hy_hat = self.label_ch.forward(y)
            #reverse
            x_hat = self.image_ch.reverse(hp)
            y_hat = self.label_ch.reverse(hp)

            #tally the losses
            #forward
            fwd_x = vsa_loss(hx, hx_hat)
            rvs_x = vsa_loss(x, x_hat)
            #reverse
            fwd_y = vsa_loss(hy, hy_hat)
            rvs_y = vsa_loss(y, y_hat)

            #negative examples
            xn_hat = self.image_ch.reverse(hpn)
            yn_hat = self.label_ch.reverse(hpn)
            hxn_hat = self.image_ch.forward(xn_hat)
            hyn_hat = self.label_ch.forward(yn_hat)

            recon_xn = tf.math.abs(1 - vsa_loss(hpn, hxn_hat))
            recon_yn = tf.math.abs(1 - vsa_loss(hpn, hyn_hat))

            losses = [fwd_x, rvs_x, fwd_y, rvs_y, recon_xn, recon_yn]
            loss = tf.math.reduce_mean(losses)

        #trainable_vars = self.trainable_variables
        trainable_vars = [*self.image_ch.trainable_variables, *self.label_ch.trainable_variables]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return np.array(losses)

    """
    Given a set of image-label symbols which are to be stored in the internal cache, take a random
    subset and add it to the internal cache, randomly removing entries which are over the cache limit.
    """
    def update_cache(self, loader, cache_proportion=1.0):
        imlbl_symbols = []
        for data in iter(loader):
            (x, y) = data
            x = self.convert_img(x)
            y = self.convert_label(y)
            h = self.forward(x,y)
            imlbl_symbols.append(h)
        imlbl_symbols = tf.concat(imlbl_symbols, axis=0)

        n_true = imlbl_symbols.shape[0]
        n_sample = int(tf.floor(n_true * cache_proportion))
        sample_inds = tf.random.shuffle(tf.range(0, n_true, 1))[0:n_sample]
        sample_symbols = tf.gather(imlbl_symbols, sample_inds, axis=0)
        
        #add the new examples to the cache, shuffle it, and cut it down if it's over the limit
        self.cache = tf.concat((self.cache, sample_symbols), axis=0)
        self.cache = tf.random.shuffle(self.cache)

        current_size = self.cache.shape[0]
        if current_size > self.cache_limit:
            self.cache = self.cache[0:self.cache_limit, ...]
