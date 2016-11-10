function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
lambda = ei.lambda;
%% forward prop
%%% YOUR CODE HERE %%%
%initial a cell to store data and the activation value of Hidden layers
data_a_Hidden = cell(numHidden + 1,1);
data_a_Hidden{1,1} = data;
clear data;
for i = 1:numHidden
    %input--hidden
    z = bsxfun(@plus, stack{1,1}.W * data_a_Hidden{i,1}, stack{1,1}.b);
    data_a_Hidden{i + 1,1} = sigmoid(z);
end
%hidden--output
output_z=bsxfun(@plus, stack{2,1}.W * data_a_Hidden{numHidden + 1,1},stack{2,1}.b);
exp_z=exp(output_z);
exp_sum=sum(exp_z);
%probality of prediction
pred_prob=bsxfun(@rdivide,exp_z,exp_sum);

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

%cross-entropy
y=full(sparse(labels, 1:length(labels), 1));
ceCost = - sum(sum(y.*log(pred_prob)));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
%the residual error of ouput layer
delta= (pred_prob - y);
gradStack{2,1}.W = delta * data_a_Hidden{numHidden+1,1}';
gradStack{2,1}.b = sum(delta,2);

%the residual error of hidden layer
for i = numHidden:-1:1
    delta = stack{i+1,1}.W'*delta.*(data_a_Hidden{i+1,1}.*(1-data_a_Hidden{i+1,1}));
    gradStack{1,1}.W = delta * data_a_Hidden{i,1}';
    gradStack{1,1}.b = sum(delta,2);
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%weight penalty cost
wCost = sum(theta.*theta)/2;
cost = ceCost + lambda*wCost;

%gradient of weight decay
for i = numHidden + 1:-1:1
    gradStack{i,1}.W = gradStack{i,1}.W + lambda .* stack{i,1}.W;
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



