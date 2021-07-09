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
#@tf.function
def bind(x, y):
    vec = x + y
    vec = remap_phase(vec)
    return vec

def bind_cmpx(x, y):
    x_angles = tf.math.angle(x)
    y_angles = tf.math.angle(y)
    x_mag = tf.math.abs(x)

    z_angles = x_angles + y_angles
    z_real = x_mag * tf.math.cos(z_angles)
    z_im = x_mag * tf.math.sin(z_angles)

    z = tf.complex(z_real, z_im)
    return z

# def bundle(x, y):
    
    
#     x = phase_to_complex(x)
#     y = phase_to_complex(y)
    
#     z = tf.math.angle(tf.add(x, y)) / pi
    
#     return z

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

def bundle_cmpx(x, y):
    return x + y

"""
FHNN bundling op for arbitarily many vectors (x1 + x2 + ... + xn)
NOTE: autograd seems to have issues with this function and gradients explode
when propagating back through this version of the function
TODO: custom gradient function to fix / test with newer version?
"""
#@tf.function
def bundle_many(*args):
    pi = tf.constant(np.pi)
    
    vec = tf.concat([phase_to_complex(x) for x in args],axis=0)
    bundled = tf.math.reduce_sum(vec, axis=0)
    bundled = tf.math.angle(bundled) / pi
    
    # tf.stack(x, axis=0)
    # vec = phase_to_complex(vec)
    
    # bundled = tf.math.reduce_sum(vec, axis=0)
    # bundled = tf.math.angle(bundled) / pi
    return bundled

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


def findspks(sol, threshold=2e-3, refractory=0.25, period=1.0):
    refrac_t = period*refractory
    ts = sol.t
    tmax = ts[-1]
    zs = sol.y
    n_t = sol.t.shape[0]
    n_n = sol.y.shape[0]
    
    #find where voltage reaches its max
    voltage = np.imag(zs)
    dvs = np.gradient(voltage, axis=1)
    dsign = np.sign(dvs)
    spks = np.diff(dsign, axis=1, prepend=np.zeros_like((zs.shape[1]))) < 0
    
    #filter by threshold
    above_t = voltage > threshold
    spks = spks * above_t
    
        
    #apply the refractory period
    if refractory > 0.0:
        for t in range(n_t):
            #current time + refractory window
            stop = ts[t] + refrac_t
            if stop > tmax:
                stop_i = -1
            else:
                #find the first index where t > stop
                stop_i = np.nonzero(ts > stop)[0][0]
            
            for n in range(n_n):
                #if there's a spike
                if spks[n,t] == True:
                    spks[n,t+1:stop_i] = False

    return spks

def findspks_max(sol, threshold=0.05, period=1.0):
    all_spks = []
    
    #slice the solution into its periods
    zslices = split_by_period(sol, dtype="v", period=period)
    n_periods = len(zslices)
    n_t = sol.t.shape[0]
    n_neurons = sol.y.shape[0]
    
    for i in range(n_periods):
        zslice = zslices[i]
        spk_slice = np.zeros_like(zslice, dtype="float")
        
        vs = np.imag(zslice)
        #find the ind of the maximum voltage value
        make2d = lambda x: x.reshape((n_neurons,1))
        i_maxes = np.argmax(vs, axis=1).reshape((n_neurons,1))
        #return spk_slice, i_maxes
        np.put_along_axis(spk_slice, indices=i_maxes, values=1.0, axis=1)
        
        #take the corresponding value
        max_values = np.max(vs, axis=1)
        positive = max_values > threshold
        positive = make2d(positive)
        #only mark spikes where voltage goes positive
        np.multiply(positive, spk_slice)
        all_spks.append(spk_slice)
        
    all_spks = np.concatenate(all_spks, axis=1)
    missing_t = n_t - all_spks.shape[1]
    all_spks = np.concatenate((all_spks, np.zeros((n_neurons, missing_t))), axis=1)
    return all_spks
    
def knn(cb, bundle, n):
    sim = tf.reshape(similarity_outer(bundle, cb), -1)
    knn = tf.argsort(sim, direction='DESCENDING')[0:n]
    return knn

"""
Convert an series of integral labels to FHNN vectors via a predefined codebook
"""
def label_to_vsa(labels, codebook):
    return tf.stack([codebook[i,:] for i in labels])

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def limit_gpus():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


"""
Make a unique vector for each class in the dataset
"""
# def make_codebook(n_classes, n_d):
#     return tf.random.uniform((n_classes, n_d), minval=-1.0, maxval=1.0)

def make_codebook(n_classes, n_d, matrix=True):
    if not matrix:
        cb = {}
        for i in range(n_classes):
            cb[i] = tf.random.uniform((1, n_d), minval=-1.0, maxval=1.0)
    else:
        cb = tf.random.uniform((n_classes, n_d), minval=-1.0, maxval=1.0)

    return cb

def make_sequence(n_v, time_op, time0):
    #TODO vectorize
    times = [bind(time0, time_op)]
    for i in range(n_v-1):
        times.append(bind(times[-1], time_op))
    times = tf.concat(times, axis=1)
    
    return times

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
#@tf.function
def partial_bind(x, y, exp=1.0):
    vec = bind(power(y, exp), x)
    vec = remap_phase(vec)
    return vec

"""
Use Euler's identity to quickly convert a vector of phases to complex numbers
for addition / other ops.
"""
#@tf.function
def phase_to_complex(x):
    x = tf.complex(x, tf.zeros_like(x))
    pi = tf.complex(np.pi, 0.0)
    im = tf.complex(0.0, 1.0)
    return tf.exp(im * x * pi)

def power(x, exp):
    vec = x * tf.constant(exp, dtype=tf.float32)
    vec = remap_phase(vec)
    return vec

"""
Move phases from (-inf, inf) into (-pi, pi)
"""
#@tf.function
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

def split_by_period(sol, dtype="v", period=1.0):
    if dtype=="v":
        #looking at the voltage
        offset = period*0.25
    else:
        #looking at the current
        offset = 0.0
        
    ts = sol.t
    zs = sol.y
    tmax = ts[-1]
    periods = int(tmax // period)
    
    slices = []
    for i in range(periods):
        #find the start and stop of the period for this cycle
        tcenter = i*period + offset
        halfcycle = period/2.0
        tstart = tcenter - halfcycle
        tstop =  tcenter + halfcycle
        #print(str(tstart) + " " + str(tstop))
        
        #grab the corresponding indices and values from the solution
        inds = (ts > tstart) * (ts < tstop)
        zslice = zs[:,inds]
        slices.append(zslice)
        
    return slices
    
"""
Restrict visible GPUs since TF is a little greedy
"""
def set_gpu(idx):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[idx], 'GPU')

"""
Return the similarity of two phase vectors defined by the FHNN framework
"""
#@tf.function
def similarity(x, y):
    assert x.shape == y.shape, "Function is for comparing similarity of tensors with identical shapes and 1:1 mapping: " + str(x.shape) + " " + str(y.shape)
    pi = tf.constant(np.pi)
    return tf.math.reduce_mean(tf.math.cos(pi*(x - y)), axis=1)

def similarity_cmpx(x, y):
    assert x.shape == y.shape, "Function is for comparing similarity of tensors with identical shapes and 1:1 mapping"
    pi = tf.constant(np.pi)
    return tf.math.reduce_mean(tf.math.cos(tf.math.angle(x) - tf.math.angle(y)), axis=1)

"""
Given two tensors with non-matching first dimensions, peform the similarity calculation 
across the outer product of the vectors (pairwise similarity between each row in 2 2D tensors).
"""
#@tf.function
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
#@tf.function
def vsa_loss(y, yh):
    loss = tf.math.reduce_mean(1 - similarity(y, yh))
    return loss

"""
Unbind a vector from a product (if z = x (x) y, do z (/) y = x)
"""
#@tf.function
def unbind(x, y):
    vec = x - y
    vec = remap_phase(vec)
    return vec

def unbind_cmpx(z, y):
    z_angles = tf.math.angle(z)
    y_angles = tf.math.angle(y)
    z_mag = tf.math.abs(z)

    x_angles = z_angles - y_angles
    x_real = z_mag * tf.math.cos(x_angles)
    x_im = z_mag * tf.math.sin(x_angles)

    x = tf.complex(x_real, x_im)
    return x

def unbundle(z, y):
    angles = y - z
    
    signs = tf.math.sign(angles)
    obtuse = tf.cast(tf.math.abs(angles) > 1.0, tf.float32)
    offsets = obtuse * 2.0 * signs
    angles += offsets
    
    x = (z - angles)
    x = remap_phase(x)
    return x

def unbundle_cmpx(z, y):
    return z - y