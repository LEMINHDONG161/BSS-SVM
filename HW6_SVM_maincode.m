% Example 2.4.1
% "Introduction to Pattern Recognition: A MATLAB Approach"
% S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras

close('all');
clear;

% Generate and plot X1
load('C:\Users\jys\Desktop\Artificial Inteligence\AI\new_data10'); 
read_data_train = data_train;
read_data_test = data_test;
 X1=read_data_train(1:5000,1:2048)';
 y1= read_data_train(1:5000,2049)';
 X2=read_data_test(1:8000,1:2048)';
 y2= read_data_test(1:8000,2049)';

% Generate the required SVM classifier
kernel='poly';
kpar1=1;
kpar2=3;
%C=0.1; 
% C=0.2;
% C= 0.5;
% C=1;
 C=2;
% C=20;
tol=0.001;
steps=100000;
eps=10^(-10);
method=1;
[alpha, w0, w, evals, stp, glob] = SMO2(X1', y1',kernel, kpar1, kpar2, C, tol, steps, eps, method);

% Compute the classification error on the training set
Pe_tr=sum((2*(w*X1-w0>0)-1).*y1<0)/length(y1)

% Compute the classification error on the test set
Pe_te=sum((2*(w*X2-w0>0)-1).*y2<0)/length(y2)

% Plot the classifier hyperplane
global figt4
figt4=2;
svcplot_book(X1',y1',kernel,kpar1,kpar2,alpha,-w0)

% Count the support vectors
sup_vec=sum(alpha>0)

% Compute the margin
marg=2/sqrt(sum(w.^2))

