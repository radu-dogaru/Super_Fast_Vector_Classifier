'''

Created on Wed Apr 18 15:46:56 2018

@author: Radu Dogaru radu_d@ieee.org 

Calls SVM from Sklearn using datasets in Matlab format (LIBSVM style)
For comparison with SFSVC 
 
Last revision: May 18, 2019 

'''


import numpy as np 
import time 
from sklearn.svm import SVC
import scipy.io as sio

#================ALGORITHM PARAMETER  ========================================

#nume='optd64'
nume='usps'

gam=.001; gam=float(gam) # Gamma parameter (only for rbf kernel)
C_=10; C_=float(C_) # C - regularization param. 

tip_nuc='rbf' #  It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used
part=0; # 0 - if all samples are used; n>0 if first samples from training set are used 

#============================================================================


# construct SVM model  
clf = SVC(C=C_,gamma=gam, kernel=tip_nuc)

# load datasets 

timer = time.time()
db=sio.loadmat(nume+'_train.mat')
Samples=db['Samples'].astype('float32')
Samples=Samples.T

print('Training samples: ', np.shape(Samples))
Labels=db['Labels'].astype('int8')
N=np.size(Samples,0)
y=np.reshape(Labels-1,N)

if part>0: 
    if part>N: 
        part=N
    Samples=Samples[0:part,:]
    y=y[0:part]
    print('Only first  ',part,' training samples are used')

    
n=np.size(Samples,0)
M=np.max(Labels)
runtime = time.time() - timer
print( " load train data time: %f s" % runtime)
# Training (may last very much - e.g. with MNIST)
timer = time.time()


clf.fit(Samples,y)


runtime = time.time() - timer
print( " TRAINING time: %f s" % runtime)

# Load test set  

timer = time.time()
db=sio.loadmat(nume+'_test.mat')
Samples=db['Samples'].astype('float32')
Samples=Samples.T
Labels=db['Labels'].astype('int8')
N=np.size(Samples,0)
n=np.size(Samples,0)
M=np.max(Labels)
runtime = time.time() - timer
print( " load test data time: %f s" % runtime)
y=np.reshape(Labels-1,N)


timer = time.time()

Acc=clf.score(Samples, y)

runtime = time.time() - timer
print( " PREDICTION (test) time: %f s" % runtime)
print("Accuracy: %f" %(Acc*100))
nsv=np.shape(clf.support_vectors_)
print( "Support vectors: %d" %nsv[0])
