# Super_Fast_Vector_Classifier
A Python implementation of the algorithm described in paper 
Radu Dogaru, Ioana Dogaru, "Optimized Super Fast Support Vector Classifiers Using Python and Acceleration of RBF Computations",  (2018) ;
https://ieeexplore.ieee.org/document/8484742 

FEATURES: 
There is no learning in the output layer, only a relatively fast selection of support vectors in a RBF-layer optimized for speed. 
Faster than SVM, particulary for large dataset. Unlike SVM the "training" (vector support selection) time is almost linear 
in the number of units. 

Hardware friendly, since arbitrary kernel (RBF) functions can be used 
(in particular the triangular one implemented as typ=1) 

DEPENDENCIES: NUMPY, SCIPY, (SKLEARN  only for SVM) ; Tested on Kaggle and Anaconda 

USAGE:  Run the sfsvc.py file after updating the parameter section (in the beginnimng of code - line 58) 
keep "prag=1", start with large radius (e.g. 200) divide it by 2 until number of support vectors are comparable to 
the number of samples (e.g. 1-10% of samples) then tune it smoothly until best performance is obtained. 

SVM COMPARISON
For comparison with SVM using the same data format, one may use the svm.py file 

R. Dogaru, 
May 2019 
