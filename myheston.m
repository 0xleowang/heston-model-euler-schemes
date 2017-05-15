%
% My implementation of the Heston model to price European Call option 
%  by using Monte Carlo Simulation with Euler discretization.
%
% Author: Shiyi Wang (Leo)
% Date: 13th May 2017
% 
% Input:   S_0    - Spot price of the underlying asset
%          K      - Strike price
%          T      - Time interval (years)
%          r      - Risk-free rate
%          V_0    - Initial variance
%          theta  - Long-term average variance
%          kappa  - Speed of mean-reversion of the variance
%          omega  - Volatility of variance
%          rho    - Instantaneous correlation coefficient
%          steps_per_year - Number of steps per year
%          num_paths - Number of simulation paths
%          schemes - 1 - Absorption
%                    2 - Reflection
%                    3 - Partial truncation
%                    4 - Full truncation
%
% Output:  C_T   - The price of the European call option.
%          Delta - Rate of change of the theoretical option value
%          Gamma - Rate of change in the Delta respect to chagnes in the
%                   underlying price.
%
function [C_T_c, Delta, Gamma] = myheston(S_0,K,T,r,V_0,...
                                        theta,kappa,omega,rho,...
                                        steps_per_year,num_paths,scheme)
switch scheme
    case 1    % Absorption
        f_1 = @(x)max(x, 0);  f_2 = @(x)max(x, 0);  f_3 = @(x)max(x, 0);
    case 2    % Reflection
        f_1 = @(x)abs(x);     f_2 = @(x)abs(x);     f_3 = @(x)abs(x);
    case 3    % Partial truncation
        f_1 = @(x)x;          f_2 = @(x)x;          f_3 = @(x)max(0,x);
    case 4    % Full truncation
        f_1 = @(x)x;          f_2 = @(x)max(0,x);   f_3 = @(x)max(0,x);
end

dt = 1/steps_per_year; % Step size
dS = 0.01*S_0; % Change of price for calculating Greeks

% Initialize matrices to record values during the simulation
V_tilde = zeros(num_paths, T/dt+1);
V       = zeros(num_paths, T/dt+1);
ln_S_c    = zeros(num_paths, T/dt+1); % for S0
ln_S_b    = zeros(num_paths, T/dt+1); % for S0-dS
ln_S_f    = zeros(num_paths, T/dt+1); % for S0+dS

% Set values at t=0
V_tilde(:,1) = V_0;
V(:,1)       = V_0;
ln_S_c(:,1)    = log(S_0);
ln_S_b(:,1)    = log(S_0-dS);
ln_S_f(:,1)    = log(S_0+dS);

% Generate values of correlated Bronian motions
dW_V = normrnd(0,sqrt(dt),num_paths,T/dt);
dZ   = normrnd(0,1,num_paths,T/dt);
dW_S = rho.*dW_V + sqrt(1-rho^2).*dZ.*sqrt(dt);

% Monte Carlo Simulation
for j = 1:T/dt
    V_tilde(:,j+1) = f_1(V_tilde(:,j)) ...
                      - kappa.*dt.*(f_2(V_tilde(:,j))-theta) ...
                      + omega.*sqrt(f_3(V_tilde(:,j))).*dW_V(:,j);
    V(:,j+1)       = f_3(V_tilde(:,j+1));
    ln_S_c(:,j+1)  = ln_S_c(:,j) + (r-0.5.*V(:,j)).*dt ...
                      + sqrt(V(:,j)).*dW_S(:,j);
    ln_S_b(:,j+1)  = ln_S_b(:,j) + (r-0.5.*V(:,j)).*dt ...
                      + sqrt(V(:,j)).*dW_S(:,j);
    ln_S_f(:,j+1)  = ln_S_f(:,j) + (r-0.5.*V(:,j)).*dt ...
                      + sqrt(V(:,j)).*dW_S(:,j);
end

% Revert log(S) to S.
S_T_c = exp(ln_S_c(:,end));
S_T_b = exp(ln_S_b(:,end));
S_T_f = exp(ln_S_f(:,end));

% Calculate the option's price
C_T_c = exp(-r*T)*mean(max(S_T_c-K,0));
C_T_b = exp(-r*T)*mean(max(S_T_b-K,0));
C_T_f = exp(-r*T)*mean(max(S_T_f-K,0));

% Calculate Greeks: Delta and Gamma
Delta = (C_T_f - C_T_b) / (2*dS);
Gamma = (C_T_f - 2*C_T_c + C_T_b) / (dS^2);

end