%% MTH5530 - Introduction to Computational Finance and Monte Carlo Methods
% Assignment
%
% Author: Shiyi Wang (Leo)
% Date: 13th May 2017
%

% Initialize variables 
S_0   = 100;   % Spot price of the underlying asset
K     = 100;   % Strike price
T     = 5;     % Time interval (years)
r     = 0.05;  % Risk-free rate
V_0   = 0.09;  % Initial variance
theta = 0.09;  % Long-term average variance
kappa = 2;     % Speed of mean-reversion of the variance
omega = 1;     % Volatility of volatility
rho   = -0.3;  % Instantaneous correlation coefficient

% Error estimating functions
x_true = 34.9998; % true value of the call option

% Functions for error analysis
calc_bias = @(x) abs(mean(x)-x_true);
calc_std  = @(x) sqrt(mean((x-mean(x)).^2));
calc_RMSE = @(x) sqrt(calc_bias(x)^2+calc_std(x)^2);

paths_list = [10000, 40000, 160000];
steps_list = [20,    40,    80];

% Calculate Delta and Gamma by Black-Scholes model
sigma = 0.3;
fprintf('Delta_BS: %.4f\n', blsdelta(S_0,K,r,T,sigma));
fprintf('Gamma_BS: %.4f\n', blsgamma(S_0,K,r,T,sigma));

% Simulation
num_iters = 100;
for scheme = 1:4 % iteration for 4 schemes
    switch scheme
        case 1; scheme_name = 'Absorption';
        case 2; scheme_name = 'Reflection';
        case 3; scheme_name = 'Partial Truncation';
        case 4; scheme_name = 'Full Truncation';
    end
    
    for ii = 1:3 % iteration for various number of paths and step sizes
        fprintf('------------------------------------------------\n');
        % Choose the number of path and corresponding step size
        num_paths      = paths_list(ii);
        steps_per_year = steps_list(ii);
        fprintf('Scheme: %s | Paths: %d | Steps/year: %d\n',...
                 scheme_name, num_paths, steps_per_year);
        
        % Initialize vectors to record C_T, Delta and Gamma     
        C_T_list   = zeros(num_iters,1);
        Delta_list = zeros(num_iters,1);
        Gamma_list = zeros(num_iters,1);

        % Repeat 100 times to calculate the price
        tic;
        parfor iter = 1:num_iters % parallel for-loop
            [C_T_list(iter),Delta_list(iter),Gamma_list(iter)] = ...
                myheston(S_0, K, T, r, V_0, theta, kappa, omega, rho,...
                         steps_per_year, num_paths, scheme);
        end
        toc;
        
        % Output result
        fprintf('C_T: %.4f | Delta: %.4f | Gamma: %.4f\n', ...
            mean(C_T_list), mean(Delta_list), mean(Gamma_list));
        fprintf('Bias: %.4f | Std: %.4f | RMSE: %.4f\n', ...
            calc_bias(C_T_list), calc_std(C_T_list), calc_RMSE(C_T_list));
    end
end
    
    
  