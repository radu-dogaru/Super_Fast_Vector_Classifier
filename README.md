# Super_Fast_Vector_Classifier
A Python implementation of the algorithm described in paper 
Radu Dogaru, Ioana Dogaru, "Optimized Super Fast Support Vector Classifiers Using Python and Acceleration of RBF Computations",  (2018) ; 

There is no learning in the output layer, only a relatively fast selection of support vectors in a RBF-layer optimized for speed. 
Faster than SVM, particulary for large dataset. Unlike SVM the "training" (vector support selection) time is almost linear 
in the number of units. 

Hardware friendly since arbitrary kernel (RBF) functions can be used (in particular the triangular one implemented as typ=1) 

USAGE: keep "prag=1", start with large radius (e.g. 200) divide it by 2 until number of support vectors are comparable to 
the number of samples (e.g. 1-10% of samples) then tune it smoothly until best performance is obtained. 

FOR comparison with SVM using the same data format one may use the svm.py file 

