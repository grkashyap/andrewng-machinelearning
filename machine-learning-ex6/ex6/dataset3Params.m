function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

CParms = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaParms = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i=1:length(CParms)
  C_i = CParms(i);
  for j=1:length(sigmaParms)
    sigma_j = sigmaParms(j);
    model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
    predictions = svmPredict(model,Xval);
    predError(j) = mean(double(predictions ~= yval));
    
    if (j==1)
      predError_j = predError(j);
    else
      predError_j = [predError_j; predError(j)];
    end   
  end
  
  if (i==1)
      predError_i=predError_j;
    else
      predError_i=[predError_i predError_j];
    end
end 

[r,c]=find(predError_i==min(min(predError_i)));

C = CParms(c);
sigma = sigmaParms(r);

end
