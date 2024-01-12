from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

from numpy import genfromtxt
from skimage.metrics import structural_similarity as ssim
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import logging
from numpy import matlib
from scipy.signal import find_peaks
from scipy.signal import hilbert, chirp

from scipy.stats import zscore

import os
import scipy.io


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def SubDiagonal(arr):
    D = arr.shape[0]
    dim = int(D*(D-1)*0.5)
    subdiag=np.zeros((dim))
    for i in range(1, D):
        subdiag[i-1]=arr[i][i - 1]
        
    return subdiag

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def upper_tri_masking_mat(A):
    mat_mit = np.zeros((62,62))
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    mat_mit[mask] = A[mask]
    return mat_mit


    
def data_in_latent(models,
                 data,
                 batch_size=128):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data

    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)

    return z_mean

def recons_error_in_FC(models,
                 data, color,nombre,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    #plt.figure(figsize=(10, 10))
#    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    x_decoded = decoder.predict(z_mean)

    return x_decoded


def compute_freq(signal,fs,inf,sup):
    z1= hilbert(signal-np.mean(signal))
    inst_phase1 = np.unwrap(np.angle(z1[inf:sup],deg=False))#inst phase  
    inst_freq1 = np.diff(inst_phase1)/(2*np.pi)*fs #inst frequency
    freq = np.mean(inst_freq1)
    peaks, _ = find_peaks(signal, height=0)
    return freq
# MNIST dataset

# load data


mat = scipy.io.loadmat('patterns2mani2_EM.mat')
my_data=mat['Phasestot']

TIME_POINTS = mat['time']  # Number of time points that we wanna end up with!
TIME_POINTS = TIME_POINTS[0,0]
NSUB = mat['NSUB']
NSUB = NSUB[0,0]


    
label = genfromtxt('label_HCP.csv',delimiter=',')
Isubdiag = genfromtxt('isubdiag.csv',delimiter=',')
Isubdiag = Isubdiag.astype(int)

my_data=np.transpose(my_data)
my_data = scipy.stats.zscore(my_data)

# slip data in train and test (the data is randomized before, randomized )
x_train = my_data[0:int(len(my_data)*0.7)]
x_test = my_data[int(len(my_data)*0.7)+1:len(my_data)]


label_1 = label
y_train = label_1[0:int(len(my_data)*0.7)]
y_test = label_1[int(len(my_data)*0.7)+1:len(my_data)]


original_dim = 62


# network parameters
input_shape = (original_dim, )
intermediate_dim = 264
batch_size = 128
latent_dim = 9
epochs = 20

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='tanh')(x) # linear o sigmoid

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')



models = (encoder, decoder)
data = (x_test, y_test)
reconstruction_loss = binary_crossentropy(inputs,outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

                                                
vae.load_weights('vae_FCsHCP_FULL_dim9.h5')


# err de corr y reconstr
FCerror =[]
FCemp2 = np.zeros((NSUB, original_dim, original_dim))
TP=TIME_POINTS
for i in range(NSUB):
    print(i)
    s1 = my_data[i*TP:(i+1)*TP,:]+1
    s1 = scipy.stats.zscore(s1) # ZSCORE 
    label_s1= np.ones(TP)*1
    s1_data = (s1,label_s1)
    zz=recons_error_in_FC(models,
                          s1_data,'orange','N3',
                          batch_size=batch_size,
                          model_name="vae_mlp")
    FC_zz = np.corrcoef(np.transpose(zz))
    FC_s1 = np.corrcoef(np.transpose(s1))
    FCemp2[i,:,:] = FC_s1
    FC_zz = FC_zz.flatten(order='C')
    FC_s1 = FC_s1.flatten(order='C')
    FC_zz = FC_zz[Isubdiag]    
    FC_s1 = FC_s1[Isubdiag]
    cc = np.corrcoef(FC_zz,FC_s1)
    FCerror.append(cc[0,1])
    
FCerror =np.array(FCerror)
FCemp = np.mean(FCemp2,0)    
    
    
    
## Hopf model in latent space


NUM_PARCELS = latent_dim
TR = 0.72   # Repetition time: this is the sampling time of our instrument. Points will be simulated in between
dt = .05 * TR   # Time interval of simulations and integration
sig = 6e-4; dsig = np.sqrt(dt) * sig    # Scaling factors for the noise!
G=0.1


NUM_RUNS = 9   # time of repeat model in each time
NITER = 300 # iterations of gradiente desc EC
NSUB=100  # subject from empirical data

fittFC_cor = np.zeros((NITER,NSUB))
fittFC_dis = np.zeros((NITER,NSUB))
fittFCr_cor = np.zeros((NITER,NSUB))
fittFCr_dis = np.zeros((NITER,NSUB))
fittFCf_cor = np.zeros((NITER,NSUB))
fittFCf_dis = np.zeros((NITER,NSUB))
CORR_sub = np.zeros((NUM_RUNS,NSUB))
SSIM_sub = np.zeros((NUM_RUNS,NSUB))
cc =np.zeros((NUM_RUNS,NSUB))
cc_mean =np.zeros((NSUB,1))
ssim_mean = np.zeros((NSUB,1))


# for of the subjects


frequency = np.zeros((latent_dim,NSUB))
structural_out = np.zeros((NSUB,latent_dim,latent_dim))
for i in range(NSUB):
    print(i)


    # took 9 subject and average to take 100 NSUB
    FCemplatentr2 = np.zeros((9,latent_dim,latent_dim))
    FCemplatentf2 = np.zeros((9,latent_dim,latent_dim))
    FCemplatent2 = np.zeros((9,latent_dim,latent_dim))
    FCemp_sub2 = np.zeros((9,original_dim,original_dim))
    frequency = np.zeros((latent_dim,9))
    Isubdiag_L =  np.tril_indices_from(np.squeeze(FCemplatentr2[0,:,:]),k=-1)
    for kk in range(9):
        s1 = my_data[9*i*TP+TP*kk:(9*i+1+kk)*TP,:]
        label_s1= np.ones(TP)*1
        s1_data = (s1,label_s1)
        FCemp_sub2[kk,:,:] = np.corrcoef(np.transpose(s1))
        latspace = data_in_latent(models,
                                  s1_data,
                                  batch_size=128)
        latspace=np.transpose(latspace)
        for dim in range(latent_dim):
            frequency[dim,kk] = compute_freq(latspace[dim,:],0.5,10,1000)  
        FCemplatent2[kk,:,:]= np.corrcoef(latspace,rowvar=True)
        aux_tauf = np.corrcoef(latspace[:,:-3],latspace[:,3:])
        FCemplatentf2[kk,:,:] = aux_tauf[latent_dim:,:latent_dim]
        aux_taur = np.corrcoef(latspace[:,-1:3:-1],latspace[:,-3:1:-1])
        FCemplatentr2[kk,:,:]= aux_taur[latent_dim:,:latent_dim]
    FCemplatent = np.mean(FCemplatent2,0)
    FCemplatentr = np.mean(FCemplatentr2,0)
    FCemplatentf = np.mean(FCemplatentf2,0)
    FCemp_sub = np.mean(FCemp_sub2,0)
    
    freq_mean = np.mean(frequency,1)
    
    
    
    hopf_frequencies = np.zeros((latent_dim,1))
    hopf_frequencies[:,0] = freq_mean
    omega = matlib.repmat(2 * np.pi * hopf_frequencies, 1, 2) # Node frequencies
    omega[:, 0] *= -1   # The frequency associated with the x equation is negative

# for the model
    a = -0.02 * np.ones((NUM_PARCELS, 2))  # Bifurcation parameter¡

# for of the subjects

# iteration for compute de effective connec by subject
    structural_conn_new=np.random.rand(latent_dim,latent_dim)
    np.fill_diagonal(structural_conn_new,0)
    for iter in range(NITER):
        #print(iter)
        tss = np.zeros((NUM_RUNS, NUM_PARCELS, TIME_POINTS))
        FCs = np.zeros((NUM_RUNS, NUM_PARCELS, NUM_PARCELS))
        FCtaufs = np.zeros((NUM_RUNS, NUM_PARCELS, NUM_PARCELS))
        FCtaurs = np.zeros((NUM_RUNS, NUM_PARCELS, NUM_PARCELS))
        structural_conn_new /= structural_conn_new.max()
        structural_conn_new *= 0.2  # Renormalized to [0, 0.2] for model reasons
        weighted_conn = G * structural_conn_new # for how the integration is computed
        sum_conn = np.matlib.repmat(weighted_conn.sum(1, keepdims=True), 1, 2)  # for sum(i)(Cij*xj)    
       # Initialize variables
        z = 0.1 * np.ones((NUM_PARCELS, 2))    # x = z[:, 0], y = z[:, 1]
        xs = np.zeros((TIME_POINTS, NUM_PARCELS))    # Array to save data
        nn = 0  # Number of simulated values saved
    
    # a set of simulation for each case for fit the EC
        for subsim in range(NUM_RUNS):
       
       # logging.debug(f"Subsimulation number {subsim + 1}")
        # Initialize variables
            phase = 2*np.pi* np.random.rand(1,NUM_PARCELS)-np.pi
            x=np.cos(phase)
            y =np.sin(phase)
            z[:,0]=x
            z[:,1]=y
            xs = np.zeros((TIME_POINTS, NUM_PARCELS))    # Array to save data
            nn = 0  # Number of simulated values saved
        
        # Discard the first 2k seconds (transient)
            for t in np.arange(0, 2000+dt, dt):
                zz = z[:, ::-1]  # flipped so that zz[:, 0] = y; zz[:, 1] = x
                interaction = weighted_conn @ z - \
                    sum_conn * z  # sum(Cij*xi) - sum(Cij)*xj
                bifur_freq = a * z + zz * omega  # Bifurcation factor and freq terms
                intra_terms = z * (z*z + zz*zz)
            # Gaussian noise
                noise = dsig * np.random.normal(size=(NUM_PARCELS, 2))
            # Integrative step
                z = z + dt * (bifur_freq - intra_terms + interaction) + noise
        
        # Compute and save the non-transient data (x = BOLD signal (interpretation), y = some other osc)
        # The way it has been impleneted here is conservative for the number of points saved
            iter0 = 0
            while nn < TIME_POINTS:
                zz = z[:, ::-1]  # flipped so that zz[:, 0] = y; zz[:, 1] = x
                interaction = weighted_conn @ z - \
                    sum_conn * z  # sum(Cij*xi) - sum(Cij)*xj
                bifur_freq = a * z + zz * omega  # Bifurcation factor and freq terms
                intra_terms = z * (z*z + zz*zz)
            # Gaussian noise
                noise = dsig * np.random.normal(size=(NUM_PARCELS, 2))
            # Integrative step
                z = z + dt * (bifur_freq - intra_terms + interaction) + noise
                iter0 += 1
            # Save simulated data if conditions are met
            # if t % TR < (dt*TR)/5:
                if iter0 >= TR/dt:
                    iter0 = 0
                    xs[nn, :] = z[:, 0].T   # save values from x
                    nn += 1
        
        # Get the timeseries with parcells as rows
        # Save timeseries (ts = xs.T) with the rest of the subsims
            tss2 = scipy.stats.zscore(xs)
            tss[subsim, :, :] = tss2.T
            FCs[subsim, :, :] = np.corrcoef(tss2.T, rowvar=True)
            aux_tauf = np.corrcoef(tss2[:-3,:].T,tss2[3:,:].T)
            FCtaufs[subsim, :, :] = aux_tauf[latent_dim:,:latent_dim]
            aux_taur = np.corrcoef(tss2[-1:3:-1,:].T,tss2[-3:1:-1,:].T)
            FCtaurs[subsim, :, :] = aux_taur[latent_dim:,:latent_dim]

# Get the timeseries with parcells as rows
# Save timeseries (ts = xs.T) with the rest of the subsims
        
        FCs2 = np.mean(FCs,0)
        FCtaurs2 = np.mean(FCtaurs,0)
        FCtaufs2 = np.mean(FCtaufs,0)
        fittFC2= np.corrcoef(FCs2[Isubdiag_L[0],Isubdiag_L[1]],FCemplatent[Isubdiag_L[0],Isubdiag_L[1]])
        fittFC3= np.mean((FCs2[Isubdiag_L[0],Isubdiag_L[1]]-FCemplatent[Isubdiag_L[0],Isubdiag_L[1]])**2)
        fittFCr2= np.corrcoef(FCtaurs2[Isubdiag_L[0],Isubdiag_L[1]],FCemplatentr[Isubdiag_L[0],Isubdiag_L[1]])
        fittFCr3= np.mean((FCtaurs2[Isubdiag_L[0],Isubdiag_L[1]]-FCemplatentr[Isubdiag_L[0],Isubdiag_L[1]])**2)
        fittFCf2= np.corrcoef(FCtaufs2[Isubdiag_L[0],Isubdiag_L[1]],FCemplatentf[Isubdiag_L[0],Isubdiag_L[1]])
        fittFCf3= np.mean((FCtaufs2[Isubdiag_L[0],Isubdiag_L[1]]-FCemplatentf[Isubdiag_L[0],Isubdiag_L[1]])**2)
        
        
        fittFC_cor[iter,i]=fittFC2[0][1]
        fittFC_dis[iter,i]=fittFC3
        fittFCr_cor[iter,i]=fittFCr2[0][1]
        fittFCr_dis[iter,i]=fittFCr3
        fittFCf_cor[iter,i]=fittFCf2[0][1]
        fittFCf_dis[iter,i]=fittFCf3
        
      #  print(fittFC3)
       # print(fittFCr3)
     #   print(fittFCf3)
        for ii in range(latent_dim):
            for j in range(latent_dim):
                if structural_conn_new[ii,j]>0 or j==latent_dim-ii+1:
                    structural_conn_new[ii,j]=structural_conn_new[ii,j]+0.001*(FCemplatent[ii,j]-FCs2[ii,j])-0.001*(FCemplatentf[ii,j]-FCtaufs2[ii,j])+0.001*(FCemplatentr[ii,j]-FCtaurs2[ii,j])
                    if structural_conn_new[ii,j]<0:
                        structural_conn_new[ii,j]=0
                        #  -FLAGREV*0.0001*(FCemplatentr(i,j)-FCtaufsim(i,j)) ...
                        #  +FLAGREV*0.0001*(FCemplatentr(i,j)-FCtaursim(i,j));

# last simulation with the optimal EC
    
    
    FCsim2 = np.zeros((NUM_RUNS, original_dim, original_dim))
    
    
    
    
    weighted_conn = G * structural_conn_new # for how the integration is computed
    sum_conn = np.matlib.repmat(weighted_conn.sum(1, keepdims=True), 1, 2)  # for sum(i)(Cij*xj)    
   # Initialize variables
    z = 0.1 * np.ones((NUM_PARCELS, 2))    # x = z[:, 0], y = z[:, 1]
    xs = np.zeros((TIME_POINTS, NUM_PARCELS))    # Array to save data
    nn = 0  # Number of simulated values saved
    for subsim in range(NUM_RUNS):
            print(subsim)
            # logging.debug(f"Subsimulation n umber {subsim + 1}")
        
        # Initialize variables
            
            phase = 2*np.pi* np.random.rand(1,NUM_PARCELS)-np.pi
            x=np.cos(phase)
            y =np.sin(phase)
            z[:,0]=x
            z[:,1]=y
            xs = np.zeros((TIME_POINTS, NUM_PARCELS))    # Array to save data
            
            nn = 0  # Number of simulated values saved
        
        # Discard the first 2k seconds (transient)
            for t in np.arange(0, 2000+dt, dt):
                zz = z[:, ::-1]  # flipped so that zz[:, 0] = y; zz[:, 1] = x
                interaction = weighted_conn @ z - \
                    sum_conn * z  # sum(Cij*xi) - sum(Cij)*xj
                bifur_freq = a * z + zz * omega  # Bifurcation factor and freq terms
                intra_terms = z * (z*z + zz*zz)
            # Gaussian noise
                noise = dsig * np.random.normal(size=(NUM_PARCELS, 2))
            # Integrative step
                z = z + dt * (bifur_freq - intra_terms + interaction) + noise
        
        # Compute and save the non-transient data (x = BOLD signal (interpretation), y = some other osc)
        # The way it has been impleneted here is conservative for the number of points saved
            iter0 = 0
            while nn < TIME_POINTS:
                zz = z[:, ::-1]  # flipped so that zz[:, 0] = y; zz[:, 1] = x
                interaction = weighted_conn @ z - \
                    sum_conn * z  # sum(Cij*xi) - sum(Cij)*xj
                bifur_freq = a * z + zz * omega  # Bifurcation factor and freq terms
                intra_terms = z * (z*z + zz*zz)
            # Gaussian noise
                noise = dsig * np.random.normal(size=(NUM_PARCELS, 2))
            # Integrative step


                z = z + dt * (bifur_freq - intra_terms + interaction) + noise
                iter0 += 1
            # Save simulated data if conditions are met
            # if t % TR < (dt*TR)/5:
                if iter0 >= TR/dt:
                    iter0 = 0
                    xs[nn, :] = z[:, 0].T   # save values from x
                    nn += 1
        
        # Get the timeseries with parcells as rows
        # Save timeseries (ts = xs.T) with the rest of the subsims
            
            ts2 = scipy.stats.zscore(xs)
            ts2 = ts2.T
            ts_final_deco = decoder.predict(np.transpose(ts2))
            FCsim2[subsim,:,:] = np.corrcoef(np.transpose(ts_final_deco))
            FCsim_sub = np.corrcoef(np.transpose(ts_final_deco))
            FCsim_sub22 = FCsim_sub.flatten(order='C')
            FCemp22 = FCemp_sub.flatten(order='C')
            FCemp22 = FCemp22[Isubdiag]  
            FCsim_sub22 = FCsim_sub22[Isubdiag]
            cc = np.corrcoef(FCsim_sub22,FCemp22)
            CORR_sub[subsim,i]=cc[0,1]
            SSIM_sub[subsim,i] =ssim(FCsim_sub,FCemp)
    
    FCsim = np.mean(FCsim2,0)
    FCsim22 = FCsim.flatten(order='C')
    FCemp22 = FCemp_sub.flatten(order='C')
    FCemp22 = FCemp22[Isubdiag]    
    FCsim22 = FCsim22[Isubdiag]
    cc_mean_aux = np.corrcoef(FCsim22,FCemp22)
    ssim_mean[i] = ssim(FCsim,FCemp_sub)
    cc_mean[i]=cc_mean_aux[0,1]
    structural_out[i,:,:]=structural_conn_new
# Saving the objects:

with open('eff_conn_EM_dim9_FULL.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([structural_out,cc_mean,ssim_mean, fittFC_cor,fittFC_cor, fittFCf_cor,fittFCf_dis], f)

# # Getting back the objects:
# with open('dim8.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     cc, CORR_sub, SSIM_sub, FCemp, FCsim, structural_conn_new,fittFC_cor, fittFC_dis = pickle.load(f)