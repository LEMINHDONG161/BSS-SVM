% Example 2.4.2
% "Introduction to Pattern Recognition: A MATLAB Approach"
% S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras

close('all');
clear all;


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

% Generate z1
z1=-ones(10,32036);
for i=1:32036
    z1(y1(i),i)=1;
end

% Generate z2
z2=-ones(10,8009);
for i=1:8009
    z2(y2(i),i)=1;
end



% 2. Compute the SVM classifiers
kernel='linear';
kpar1=0;
kpar2=0;
C=20;
tol=0.001;
steps=100000;
eps=10^(-10);
method=1;
for i=1:10
    [alpha(:,i), w0(i), w(i,:), evals, stp, glob] = SMO2(X1', z1(i,:)', kernel, kpar1, kpar2, C,...
        tol, steps, eps, method);
    marg(i)=2/sqrt(sum(w(i,:).^2));
    sup_vec(i)=sum(alpha(:,i)>0);
end
marg,sup_vec

% Estimate the classification error based on X2
[vali,class_est]=max(w*X2-w0'*ones(1,8009));
err_svm=sum(class_est~=y2)/8009

