function [R, S, SigmaKr] = FAANV(X, r, thres, init, epsilon)
% FAANV Factor Analysis for Anisotropic Noise in Vector estimation
%
% Inputs:
%   X       - Data matrix
%   r       - Desired rank
%   thres   - Convergence threshold, usually 10^-3
%   init    - Initial estimate of the noise covariance matrix
%   epsilon - Small positive constant (not described in original comments)
%
% Outputs:
%   R       - Estimated covariance matrix
%   S       - Estimated Low rank matrix
%   SigmaKr - Kronecker product of estimated noise covariance matrix with identity matrix
%
% Description:
%   This function implements the Factor Analysis for Anisotropic Noise in Vector (FAANV)
%   estimation algorithm. It estimates the covariance matrix, low rank matrix,
%   and noise covariance matrix from the input data.
%
% Author: Prabhu Babu (prabhubabu@care.iitd.ac.in)
% Date: 24/01/2023
% Modified by: M-.G. Wang
% Date: 05/05/2023

M       = size(X,1) / 3;       % Dimension
R_cap   = X * X' / size(X,2);  % Sample covariance matrix
Sigma   = init;                % Initialization for the diagonal matrix
SigmaKr = kron(Sigma, eye(3));
IT      = 1;                   % Loop indicator
count   = 0;                   % Loop counter
LL      = 10^5;                % Initial value of the log-likelihood, taken to be a large value.

% Main iteration loop
while(IT==1)
    R_tilda  = diag(sqrt(1 ./ diag(SigmaKr))) * R_cap * diag(sqrt(1 ./ diag(SigmaKr)));
    [U, mu]  = eig(R_tilda); % Calculation of U
    [~, ind] = sort(diag(real(mu)), 'descend');
    S        = U(:, ind(1:r)) * diag(sqrt(max(diag(real(mu(ind(1:r), ind(1:r)))) - epsilon, 0)));
    Gamma    = inv(S * S' + epsilon * eye(3 * M));
    tmp      = R_cap .* Gamma;
    s        = sqrt(diag(Sigma));
    % Inner loop to calculate the diagonal elements of noise
    % Covariance matrix
    for titer = 1:10 % the loop over the sigma's are run for fixed number of times (10)
        for k = 1:M
            b    = -real(sum(sum(tmp(k*3-2 : k*3, :))' ./ kron(s, [1; 1; 1]))) - sum(tmp(k*3-2 : k*3, k*3-2 : k*3), "all") / s(k);
            c    = -real(sum(tmp(k*3-2 : k*3, k*3-2 : k*3), "all"));
            s(k) = (-b + sqrt(b^2 - c*12)) / 6;
        end
    end
    count  = count + 1;
    LL_new = real(log(det(sqrtm(SigmaKr) * S * (sqrtm(SigmaKr)*S)' + SigmaKr)) + trace(R_cap / (sqrtm(SigmaKr) * S * (sqrtm(SigmaKr)*S)' + SigmaKr))); % Calculation of the log-likelihood objective
    if((abs(LL - LL_new) / abs(LL) <= thres) || (count > 50)) % Convergence check
        IT = 0;
    end
    Sigma     = diag(s.^2); % Estimate of the noise covarinace matrix
    SigmaKr   = kron(Sigma, eye(3));
    LL(count) = LL_new;
end
R = sqrtm(SigmaKr) * S * (sqrtm(SigmaKr) * S)' + SigmaKr; % Estimated covariance matrix
S = sqrtm(SigmaKr) * S;                                   % Estimated Low rank matrix
