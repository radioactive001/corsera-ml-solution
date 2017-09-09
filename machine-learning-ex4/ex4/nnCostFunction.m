function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

eye_mat=eye(num_labels);
y_mat=eye_mat(y,:);

A1=[ones(m,1) X];
Z2=A1*Theta1';
A2_temp=sigmoid(Z2);
A2=[ones(m,1) A2_temp];
Z3=A2*Theta2';
A3=sigmoid(Z3);
H=A3;


S1=-(y_mat'*log(H));
S2=-((1-y_mat)'*log(1-H));

S=(S1+S2);

J1=trace(S)/m;
S3=sum(sum(Theta1(:,2:end).^2));

S4=sum(sum(Theta2(:,2:end).^2));
S5=(lambda*(S3+S4))/(2*m);

J=J1+S5;


for i=1:num_labels
 %step 1   
 A1=[ones(m,1) X];
 Z2=A1*Theta1';
 A2_temp=sigmoid(Z2);
 A2=[ones(m,1) A2_temp];
 Z3=A2*Theta2';
 A3=sigmoid(Z3);
 H=A3;
 
 
 %step 2
 D3=A3-y_mat;
 
 %step 3
 sig=sigmoidGradient(Z2);
 Theta2_temp=(Theta2(:,2:end));
 temp_t=D3 *(Theta2_temp) ;
 D2=temp_t .*sig;
 
 %step 5
 delta1=(A1'*D2)';
 delta2=(A2'*D3)';


 
end

%creating a vector of zeros
 z1_temp=zeros(size(Theta1,1),1);
 z2_temp=zeros(size(Theta2,1),1);
%
Theta1_temp=[z1_temp Theta1(:,2:end)];
Theta2_temp=[z2_temp Theta2(:,2:end)];

Theta1_grad=(delta1/m)+(lambda*Theta1_temp)/m;
Theta2_grad=delta2/m+(lambda*Theta2_temp)/m;




% df=ad;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
