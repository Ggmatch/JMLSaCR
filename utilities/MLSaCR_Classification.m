function class = MLSaCR_Classification(DataTrains, CTrain, DataTests, lambda, gamma, c, M, k)

penalty_factor = 1e+7;

DataTrain = DataTrains(:,3:end);
NUM = size(DataTrain,1);
DataTest  = DataTests(:,3:end);

DDT = DataTrain*DataTrain';

numClass = length(CTrain);
[m Nt]= size(DataTest);
for j = 1: m
    if mod(j,round(m/20))==0
        fprintf('*...');        
    end
    %% spectral
    Y = DataTest(j, :);
    minus_xy = DataTrain' - repmat(Y', [1 size(DataTrain,1)]);
    norms_k1 = diag(minus_xy'*M*minus_xy);
    [B,I_k1] = sort(norms_k1);
    nums_k1 = max(1, round(NUM*k));
    temp_k1 = I_k1(1:nums_k1);  
    
    %% spatial
    xy = DataTests(j, 1:2);
    XY = DataTrains(:, 1:2);    
    norms_k2 = sum((abs(XY' - repmat(xy', [1 size(XY,1)]))).^c);
    norms_k2 = norms_k2./max(norms_k2);
    [B,I_k2] = sort(norms_k2);
    nums_k2 = max(1,round(NUM*k));
    temp_k2 = I_k2(1:nums_k2);
    
    temp_all = union(temp_k1,temp_k2);
    temp_k1 = ismember(I_k1,temp_all);
    norms_k1(temp_k1 == 0) = penalty_factor.*norms_k1(temp_k1 == 0);
    G = diag(lambda.*norms_k1);
    
    D = diag(gamma.*norms_k2);
    
    %% get representation coefficients
    weights = (DDT +  G + D)\(DataTrain*Y');
    
    a = 0;
    for i = 1: numClass 
        % Obtain Multihypothesis from training data
        HX = DataTrain((a+1): (CTrain(i)+a), :); % sam x dim
        HW = weights((a+1): (CTrain(i)+a));
        a = CTrain(i) + a;
        Y_hat = HW'*HX;
        
        Dist_Y(j, i) = norm(Y - Y_hat); 
    end
   Dist_Y(j, :) = Dist_Y(j, :)./sum(Dist_Y(j, :));
end
[~, class] = min(Dist_Y'); 
