%% LDA and SVM

close('all');
clear;

load('C:\Users\OSElab\Desktop\MINH DONG\LEMINHDONG_trituenhantao\new_data10'); 
read_data_train = data_train;
read_data_test = data_test;
% Result
%read_data_train 32036 x2049 ; 
%read_data_test  8009x2049;
% 2049 column is the class number of 1 to 10

test=read_data_test(:,1:2048)';
train=read_data_train(:,1:2048)';
c=100;
y=read_data_train(:,2049)'; % class labels

% Scatter matrix computation
[Sw,Sb,Sm]=scatter_mat(train,y);

% Eigendecomposition
[V,D]=eig(inv(Sw)*Sb);

% Sort the eigenvalues in descending order and rearrange the eigenvectors accordingly
s=diag(D);
[s,ind]=sort(s,1,'descend');
V=V(:,ind);
% Select in A the eigenvectors corresponding to non-zero eigenvalues
A=V(:,1:c-1);
% Project the data set on the space spanned by the column vectors of A
X2=A'*test;
X1=A'*train;
 y1= read_data_train(:,2049)';
 y2= read_data_test(:,2049)';

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

