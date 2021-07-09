import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from utils import similarity

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

"""
Load a standard TF image dataset and apply the normal transforms
(cache, shuffle, batch, prefetch)
"""
def load_dataset(dataset, n_batch=-1):
    (ds_train, ds_test), ds_info = tfds.load(dataset, 
                    split=['train', 'test'], 
                    data_dir="~/data",
                    shuffle_files=True,
                    as_supervised=True,
                    with_info=True)

    ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

    if n_batch > 0:
        ds_train = ds_train.batch(n_batch)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if n_batch > 0:
        ds_test = ds_test.batch(n_batch)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, ds_info

def get_h(model, data):
    hs = [model(d[0], d[1]) for d in iter(data)]
    hs = tf.concat(hs[0:-1],axis=1)
    return hs

"""
Recreate the original dataset from the loader (only use on small datasets that fit in RAM)
"""
def get_raw_dat(data):
    data = [(d[0], d[1]) for d in iter(data)]
    xs = tf.concat([d[0] for d in data],axis=0)
    ys = tf.concat([d[1] for d in data],axis=0)
    
    return xs, ys
    
"""
Given a list of symbols x, select a subset (randomly, by default, otherwise the first n samples)
"""
def cut(x, n, rand_sel=True):
    d = []
    n_s = x[0].shape[0]

    if rand_sel:
        inds = tf.random.shuffle(tf.range(0,n_s))[0:n]
        
    for i in range(len(x)):
        if rand_sel:
            data = tf.gather(x[i], inds, axis=0)
            d.append(data)
        else:
            d.append(x[i][0:n,...])
        
    return tuple(d)
"""
Given a tensor of the raw data, separate inputs by class and return dataloaders
for each class.
"""
def class_data(xs, ys, batch_size=128):
    cs = tf.sort(tf.unique(ys)[0])
    c_inds = [ys == i for i in cs]
    
    dls = []
    for inds in c_inds:
        x_c = xs[inds]
        y_c = ys[inds]
        dl = tf.data.Dataset.from_tensor_slices((x_c, y_c))
        dl = dl.batch(batch_size)
        dls.append(dl)
        
    return dls

def class_data_pairs(xs, ys, pairs=[(0,1), (2,3), (4,5), (6,7), (8,9)]):
    cs = tf.sort(tf.unique(ys)[0])
    c_inds = [ys == i for i in cs]
    dls = []
    
    for pair in pairs:
        x_data = []
        y_data = []
        
        for c in pair:
            
            inds = c_inds[c]
            x_data.append(xs[inds])
            y_data.append(ys[inds])
            
        
        x_c = tf.concat(x_data, axis=0)
        y_c = tf.concat(y_data, axis=0)
        
        n_ex = x_c.shape[0]
        print(n_ex)
        perm = tf.random.shuffle(tf.range(0,n_ex))
        shuffle = lambda x: tf.gather(x, perm, axis=0)
        x_c = shuffle(x_c)
        y_c = shuffle(y_c)
        
        dl = tf.data.Dataset.from_tensor_slices((x_c, y_c))
        dls.append(dl)

    return dls

def make_interleaved(model, new_data, n_int, n_ext):
    r_samples = model.random_samples(n_int)
    int_symbols = model.generate_internal(r_samples)
    ext_symbols = model.generate_external(new_data)
    ext_symbols = cut(ext_symbols, n_ext)
    symbols = interleave_symbols(int_symbols, ext_symbols)
    
    return symbols

"""
Given an external dataset, separate out the desired classes and mix those examples in with
data regenerated from the model's cache according to the mixture.
"""
def cache_data_mix(model, xs, ys, external=[(2,3)], mixture=[0.1, 0.9], return_new=False):
    n_external = len(external)
    assert n_external == len(mixture)-1, "Need proportion for each pair to be included"
    
    #set up arguments
    cs = tf.sort(tf.unique(ys)[0])
    c_inds = [ys == i for i in cs]
    
    all_x = []
    all_y = []
    dls = []
    
    #regenerate data from the cache
    (xs_c, ys_c) = model.reverse(model.cache)
    
    shuffle = gen_shuffle(xs_c.shape[0])
    xs_c = shuffle(xs_c)
    ys_c = shuffle(ys_c)
    
    all_x.append(xs_c)
    all_y.append(ys_c)
    
    #add external samples
    for pair in external:
        x_data = []
        y_data = []
        
        #get the data for each of the classes
        for c in pair:
            inds = c_inds[c]
            x_data.append(xs[inds])
            y_data.append(ys[inds])
            
        #concat it down
        x_c = tf.concat(x_data, axis=0)
        y_c = tf.concat(y_data, axis=0)
        
        #convert it to symbols
        x_c = model.convert_img(x_c, training=True)
        y_c = model.convert_label(y_c)
        
        #shuffle them
        shuffle = gen_shuffle(x_c.shape[0])
        x_c = shuffle(x_c)
        y_c = shuffle(y_c)
        
        #add it to the data
        all_x.append(x_c)
        all_y.append(y_c)
        
    #cut down examples based on the proportions
    lengths = [x.shape[0] for x in all_x]
    proportions = np.array(mixture) / np.sum(mixture)
    total = np.sum(lengths)
    include = np.array([proportions[i]*lengths[i] for i in range(n_external+1)])
    include = include.astype("int")
    print(include)
    
    for i in range(n_external+1):
        all_x[i] = all_x[i][0:include[i],...]
        all_y[i] = all_y[i][0:include[i],...]

    concat = lambda x: tf.concat(x, axis=0)

    #return a copy of the new data we're learning so it can be added to the cache
    if return_new:
        new_data = (concat(all_x[1:]), concat(all_y[1:]))

    all_x = concat(all_x)
    all_y = concat(all_y)
    
    n_x = all_x.shape[0]
    shuffle = gen_shuffle(n_x)
    all_x = shuffle(all_x)
    all_y = shuffle(all_y)
    
    dl = tf.data.Dataset.from_tensor_slices((all_x, all_y))

    if not return_new:
        return dl
    else:
        return dl, new_data

def class_data_mix(xs, ys, pairs=[(0,1), (2,3)], mixture=[0.1, 0.9]):
    n_pairs = len(pairs)
    assert n_pairs == len(mixture), "Need proportion for each pair to be included"
    
    cs = tf.sort(tf.unique(ys)[0])
    c_inds = [ys == i for i in cs]
    
    all_x = []
    all_y = []
    dls = []
    
    for pair in pairs:
        x_data = []
        y_data = []
        
        for c in pair:
            inds = c_inds[c]
            x_data.append(xs[inds])
            y_data.append(ys[inds])
            
        
        x_c = tf.concat(x_data, axis=0)
        y_c = tf.concat(y_data, axis=0)
        
        n_ex = x_c.shape[0]
        shuffle = gen_shuffle(n_ex)
        x_c = shuffle(x_c)
        y_c = shuffle(y_c)
        
        all_x.append(x_c)
        all_y.append(y_c)
        
    
    lengths = [x.shape[0] for x in all_x]
    proportions = np.array(mixture) / np.sum(mixture)
    total = np.sum(lengths)
    include = np.array([proportions[i]*lengths[i] for i in range(n_pairs)])
    include = include.astype("int")
    print(include)
    
    for i in range(n_pairs):
        all_x[i] = all_x[i][0:include[i],...]
        all_y[i] = all_y[i][0:include[i],...]
        
    concat = lambda x: tf.concat(x, axis=0)
    all_x = concat(all_x)
    all_y = concat(all_y)
    
    dl = tf.data.Dataset.from_tensor_slices((all_x, all_y))

    return dl

def interleave_symbols(external, internal):
    ns_e = external[0].shape[0]
    ns_i = internal[0].shape[0]
    ns = ns_e + ns_i
    l = len(external)
    
    concat = lambda x: tf.concat(x, axis=0)
    data = [concat((external[i], internal[i])) for i in range(l)]
    
    perm = tf.random.shuffle(tf.range(0,ns))
    shuffle = lambda x: tf.gather(x, perm, axis=0)
    for i in range(l):
        data[i] = shuffle(data[i])
        
    return tuple(data)

def regen_add_data(model, xs, ys, new=(0,1)):

    cs = tf.sort(tf.unique(ys)[0])
    c_inds = [ys == i for i in cs]
    
    all_x = []
    all_y = []
    dls = []

    concat = lambda x: tf.concat(x, axis=0)
    
    #new external data
    x_data = []
    y_data = []
    
    for c in new:
        inds = c_inds[c]
        x_data.append(xs[inds])
        y_data.append(ys[inds])
        
    
    x_c = concat(x_data)
    y_c = concat(y_data)
    
    n_ex = x_c.shape[0]
    shuffle = gen_shuffle(n_ex)
    x_c = shuffle(x_c)
    y_c = shuffle(y_c)

    #convert the images/labels to symbols
    x_c = model.convert_img(x_c)
    y_c = model.convert_label(y_c)
    
    all_x.append(x_c)
    all_y.append(y_c)

    #old internal data
    cache = model.cache
    old_x = model.image_ch.reverse(cache)
    old_y = model.label_ch.reverse(cache)

    all_x.append(old_x)
    all_y.append(old_y)
    
    #concat and shuffle it
    all_x = concat(all_x)
    all_y = concat(all_y)

    n_ex = all_x.shape[0]
    shuffle = gen_shuffle(n_ex)
    all_x = shuffle(all_x)
    all_y = shuffle(all_y)
    
    dl = tf.data.Dataset.from_tensor_slices((all_x, all_y))

    return dl


def gen_shuffle(n_ex):
    perm = tf.random.shuffle(tf.range(0,n_ex))
    shuffle = lambda x: tf.gather(x, perm, axis=0)
    return shuffle

def confusion_to_accuracy(confusion):
    total = tf.math.reduce_sum(confusion)
    correct = tf.math.reduce_sum(tf.linalg.diag_part(confusion))
    return correct / total

def plotinternals(model, data, nbins=25):
    ints1 = model.get_internals(data)
    x_sym = ints1['x_vecs']
    y_sym = ints1['y_vecs']
    xy_sym = ints1['htrue_vecs']
    
    xy_xh_sym = model.image_ch.forward(x_sym)
    x_sym_recon = model.image_ch.reverse(xy_sym)
    y_sym_recon = model.label_ch.reverse(xy_sym)
    
    plt.figure()
    plt.hist(similarity(x_sym, x_sym_recon).numpy().reshape(-1), bins=nbins)
    plt.xlim(0,1)
    plt.title("similarity of recovered X vectors from XY true")
    
    plt.figure()
    plt.hist(similarity(y_sym, y_sym_recon).numpy().reshape(-1), bins=nbins)
    plt.xlim(0,1)
    plt.title("similarity of recovered Y vectors from XY true")
    
    plt.figure()
    plt.hist(similarity(xy_xh_sym, xy_sym).numpy().reshape(-1), bins=nbins)
    plt.xlim(0,1)
    plt.title("similarity of XY(X) and true XY")
    