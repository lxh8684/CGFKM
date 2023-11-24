clear;clc;

load jaffe_213n_676d_10c_uni.mat;
if exist('y', 'var')
    Y = y;
end
if exist('fea', 'var')
    X = fea;
end
clear fea;
if exist('gnd', 'var')
    Y = gnd;
end
clear gnd;
Y = Y(:);
nCluster = length(unique(Y));
nSmp = length(Y);

batch_size = 1000;
knn_size = 5;
Si = ConstructWHuge(X, 5, batch_size);
nOrder = 5;
Ts = cell(1, nOrder);
Ts{1, 1} = speye(nSmp);
Ts{1, 2} = Si;
for jOrder = 3:nOrder
    tmp1 = multi_blockSize(Si, Ts{1, jOrder-1});
    Ts{1, jOrder} =sparse(2*tmp1 - Ts{1, jOrder-2});
end
clear tmp1;

TXs = cell(1, nOrder);
for iOrder = 1:nOrder
    TXs{1, iOrder} = multi_blockSize(Ts{1, iOrder}, X);
end

A1 = zeros(nOrder, nOrder);
%*********************************************************************
% Merge T and T'
%*********************************************************************
for iOrder = 1:nOrder
    for jOrder = iOrder:nOrder
        e2_ij = sum(sum( TXs{1, iOrder} .* TXs{1, jOrder} ));
        A1(iOrder, jOrder) = e2_ij;
        A1(jOrder, iOrder) = e2_ij;
    end
end
[~, o_2] = eig(A1);
disp(['min eigval is ', num2str(min(diag(o_2)))]);
[label, objHistory, beta] = CGFKM_fast(X, nCluster, nOrder, knn_size, TXs, A1);
result_10 = my_eval_y(label, Y);
