"""
This file contains utility functions for manipulating VSAs and other tasks.

Wilkie Olin-Ammentorp, 2021
University of Califonia, San Diego
"""

import tensorflow as tf
import numpy as np

def avgphase(trains, n_n):
    n_b = len(trains)

    outputs = []
    for i in range(n_b):
        nz_i, nz_t = trains[i]

        avgphase = np.zeros(n_n, "float")
        for n in range(n_n):
            inds = nz_i == n
            avgphase[n] = np.mean(nz_t[inds])

        outputs.append(avgphase)
        
    return np.array(outputs)

"""
FHNN binding of two vectors together (x (x) y)
"""
def bind(x, y):
    vec = x + y
    vec = remap_phase(vec)
    return vec

"""
FHNN bundling of two vectors together (x (+) y)
"""
def bundle(x, y):
    #find the phase difference between the vectors
    angles = y - x
    #is the second vector proceeding or preceding the first
    signs = tf.math.sign(angles)
    #which angles are greater than 1.0 (pi)
    obtuse = tf.cast(tf.math.abs(angles) > 1.0, tf.float32)
    #for these large angles, we 
    offsets = obtuse * 2.0 * signs
    angles += offsets
    z = x + angles / 2.0
    return remap_phase(z)


"""
FHNN bundling op for arbitarily many vectors (x1 + x2 + ... + xn)
NOTE: autograd seems to have issues with this function and gradients explode
when propagating back through this version of the function
"""
def bundle_many(*args):
    pi = tf.constant(np.pi)
    
    vec = tf.concat([phase_to_complex(x) for x in args],axis=0)
    bundled = tf.math.reduce_sum(vec, axis=0)
    bundled = tf.math.angle(bundled) / pi
    
    return bundled

"""
Given a confusion matrix, calculate the corresponding accuracy score.
"""
def confusion_to_acc(confusion):
    return np.sum(np.diag(confusion)) / np.sum(confusion)
    
def construct_sparse(n1, n2, overscan=1.0):
    shp = (n1,n2)
    m = np.zeros(shp).ravel()
    
    #the number of times we'll loop through
    while overscan >= 1.0:
        #cutoff by remaining overscan
        cutoff = int(overscan * n2)
        overscan -= 1.0
        
        #the permutation from dest -> source we're assigning this time
        sources = np.floor(n1 * np.random.rand(n2)).astype(np.int)[0:cutoff]
        dests = np.arange(0, n2, 1).astype(np.int)[0:cutoff]
        #shuffle mutates original array
        np.random.shuffle(dests)
        indices = [sources, dests]
        indices = np.ravel_multi_index(indices, shp)
        
        #return indices
        m[indices] = np.random.uniform(low=-1, high=1, size=(len(indices)))
        
    m = np.reshape(m, shp)
            
    return tf.constant(m, dtype="float32")

"""
Given a codebook (cb) and a symbol (bundle), find the n nearest neighbors to that symbol in the codebook.
"""
def knn(cb, bundle, n):
    sim = tf.reshape(similarity_outer(bundle, cb), -1)
    knn = tf.argsort(sim, direction='DESCENDING')[0:n]
    return knn

"""
Convert an series of integral labels to FHNN vectors via a predefined codebook
"""
def label_to_vsa(labels, codebook):
    return tf.stack([codebook[i,:] for i in labels])


"""
Limit Tensorflow not reserve all GPU memory ahead of time.
"""
def limit_gpus():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


"""
Make a unique vector for each class in the dataset
"""
def make_codebook(n_classes, n_d, matrix=True):
    if not matrix:
        cb = {}
        for i in range(n_classes):
            cb[i] = tf.random.uniform((1, n_d), minval=-1.0, maxval=1.0)
    else:
        cb = tf.random.uniform((n_classes, n_d), minval=-1.0, maxval=1.0)

    return cb

"""
Match normalization parameters between two bridge models.
"""
def match_normalize(source, dest):
    norm = source.img_encoder.norm_symbols
    mm = norm.moving_mean
    ms = norm.moving_std
    
    norm2 = dest.img_encoder.norm_symbols
    norm2.moving_mean = mm
    norm2.moving_std = ms

"""
Bind multiple vectors together at once (x1 (x) x2 (x) ... (x) xn)
"""
def multi_bind(*x):
    vec = tf.stack(x, axis=0)
    vec = tf.math.reduce_sum(vec, axis=0)
    vec = remap_phase(x)
    return vec

"""
Bind two vectors together, with the operation being done partially (x (x) y^fraction)
e.g. partially binding forward for continuous variable such as time
"""
def partial_bind(x, y, exp=1.0):
    vec = bind(power(y, exp), x)
    vec = remap_phase(vec)
    return vec

"""
Use Euler's identity to quickly convert a vector of phases to complex numbers
for addition / other ops.
"""
def phase_to_complex(x):
    x = tf.complex(x, tf.zeros_like(x))
    pi = tf.complex(np.pi, 0.0)
    im = tf.complex(0.0, 1.0)
    return tf.exp(im * x * pi)

"""
Raise a VSA symbol to a power.
"""
def power(x, exp):
    vec = x * tf.constant(exp, dtype=tf.float32)
    vec = remap_phase(vec)
    return vec

"""
Move phases from (-inf, inf) into (-pi, pi)
"""
def remap_phase(x):
    #move from (-inf, inf) to (0,2)
    n1 = tf.constant(-1.0)
    tau = tf.constant(2.0)
    pi = tf.constant(1.0)
    
    x = tf.math.floormod(x, tau)
    #move (1,2) to (-1, 0) to be consistent with tanh activations
    return n1 * tau * tf.cast(tf.math.greater(x, pi), tf.float32) + x

"""
Return a matrix with sparsity% empty cells and the other values normally distributed.
"""
def sparse_random(shape, sparsity):
    mat = tf.random.normal(shape)
    mask = tf.cast((tf.random.uniform(shape) > sparsity), "float32")
    return mask * mat

"""
Restrict visible GPUs
"""
def set_gpu(idx):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[idx], 'GPU')

"""
Return the similarity of two phase vectors defined by the FHNN framework
"""
def similarity(x, y):
    assert x.shape == y.shape, "Function is for comparing similarity of tensors with identical shapes and 1:1 mapping: " + str(x.shape) + " " + str(y.shape)
    pi = tf.constant(np.pi)
    return tf.math.reduce_mean(tf.math.cos(pi*(x - y)), axis=1)

"""
Given two tensors with non-matching first dimensions, peform the similarity calculation 
across the outer product of the vectors (pairwise similarity between each row in 2 2D tensors).
"""
def similarity_outer(x, y):
    x_shp = x.shape
    y_shp = y.shape
    pi = tf.constant(np.pi)

    assert x_shp[1] == y_shp[1], "VSA Vector Dimension (2) must match between tensors"
    z_shp = [x_shp[0], y_shp[0], y_shp[1]]
    zx_shp = [y_shp[0], x_shp[0], x_shp[1]] #make a permutation of the shape so broadcast can work
    permu = [1,0,2]
    #broadcast and match the shapes
    xb = tf.transpose(tf.broadcast_to(x, zx_shp), permu)
    yb = tf.broadcast_to(y, z_shp)
    #perform the similarity calculation on the reshaped data
    angles = xb - yb
    similarities = tf.math.reduce_mean(tf.cos(pi*angles), axis=2)
    return similarities

"""
A loss function which maximises similarity between all VSA vectors.
"""
def vsa_loss(y, yh):
    loss = tf.math.reduce_mean(1 - similarity(y, yh))
    return loss

"""
Unbind a vector from a product (if z = x (x) y, do z (/) y = x)
"""
def unbind(x, y):
    vec = x - y
    vec = remap_phase(vec)
    return vec

"""
Given two vectors, the first a bundle and the second a component,
return the bundle without the second element.
"""
def unbundle(z, y):
    angles = y - z
    
    signs = tf.math.sign(angles)
    obtuse = tf.cast(tf.math.abs(angles) > 1.0, tf.float32)
    offsets = obtuse * 2.0 * signs
    angles += offsets
    
    x = (z - angles)
    x = remap_phase(x)
    return x