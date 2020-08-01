addpath('utilities');
addpath('data');
addpath('ml_toolbox');
addpath('itml');

dataset = 'Indian';

%% load the HSI dataset
if strcmp(dataset,'Indian')
    load Indian_pines_corrected;load Indian_pines_gt;load Indian_pines_randp
    data = indian_pines_corrected;        gth = indian_pines_gt;
elseif strcmp(dataset,'Salinas')
    load Salinas_corrected;load Salinas_gt;load Salinas_randp
    data = salinas_corrected;
    gth = salinas_gt;
elseif strcmp(dataset,'PaviaU')
    load PaviaU;load PaviaU_gt;load PaviaU_randp;
    data = paviaU;        gth = paviaU_gt;
end

%% smoothing
for i=1:size(data,3)
    data(:,:,i) = imfilter(data(:,:,i),fspecial('average',7));
end

%% hyper-parameters setting
lambda = 1e-4;
gamma = 1;
c = 4;
k = 0.9;

%% experimental parameters setting
train_num = 20;
train_ratio = 0.1;
iter = 1;
randpp=randp{iter};
[DataTest, DataTrain, CTest, CTrain] = samplesdivide(data,gth,train_num,train_ratio,randpp,dataset);
% get labels of train and test datasets
true_label = [];
y_train = [];
start = 0;
a=0;
for i = 1:length(CTest)
    true_label(start+1:start+CTest(i)) = i;
    start = start+CTest(i);
    y_train(a+1:a+CTrain(i),1) = i;
    a = a + CTrain(i);
end

%% run
% If prepare the M in advance
% load('M.mat','M');
M = MetricLearningAutotuneKnn(@ItmlAlg, y_train, DataTrain(:,3:end)); % generate only once for same dataset
% save('M.mat','M');
estim_label = MLSaCR_Classification(DataTrain, CTrain, DataTest, lambda, gamma, c, M, k);
[accur_NRS, aa, K, ua, confu] = Confusion(true_label,estim_label);
fprintf('\n dataset is %s, training number is %d, accuracy is %f.\n', dataset, train_num, accur_NRS);