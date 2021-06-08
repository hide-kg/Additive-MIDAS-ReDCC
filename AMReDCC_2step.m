function [estimpara, forecast_fit, logL] = AMReDCC_2step(RC, L, test_start)
%
% 2021/3/26
%   Bauwens et al. (2016)のReDCCモデルのMIDAS拡張型モデルの実装
%   推定方法はBauwens et al. (2012)の2ステップ推定を用いる
%

%% 初期設定
[K, ~, T] = size(RC);
para0 = amredcc_initpara(K);
[gamma0, delta0, m0, theta0, omega_s0, alpha0, beta0, omega_r0, nu0] = AMReDCC_transpara(para0,K);
options = optimoptions('fminunc', 'Display', 'off');
RV = zeros(T, K);
RCorr = zeros(K, K, T);
for t = 1:T
    [RCorr(:,:,t), RV(t,:)] = cov_to_corr(RC(:,:,t));
end

%% ステップ1
para0_garch = [gamma0, delta0, m0, theta0, omega_s0];
gamma = zeros(K,1);
delta = zeros(K,1);
m = zeros(K,1);
theta = zeros(K,1);
omega_s = zeros(K,1);

D = zeros(K, K, T);
s = zeros(T, K);

for i = 1:K
    llh = @(x0) -amredcc_step1_llh(x0, RV(1:test_start-1,i), L, 0);
    warning('off') %#ok<*WNOFF>
    para_variance = fminunc(llh, para0_garch(i,:), options);
    warning('on') %#ok<*WNON>
    [~, s(:,i)] = amredcc_step1_llh(para_variance, RV(:,i), L, 1);
    gamma(i) = para_variance(1);
    delta(i) = para_variance(2);
    m(i) = para_variance(3);
    theta(i) = para_variance(4);
    omega_s(i) = para_variance(5);
    for t = L+1:T
        D(i,i,t) = s(t,i);
    end
end
para_garch = [gamma.^2, delta.^2, m.^2, theta.^2, abs(omega_s)+1];

%% ステップ2
para0_dcc = [alpha0, beta0, omega_r0, nu0];
llh = @(x0) -amredcc_step2(x0, RCorr(:,:,1:test_start-1), RC(:,:,1:test_start-1), D(:,:,1:test_start-1), L, 0);
warning('off') %#ok<*WNOFF>
para_corr = fminunc(llh, para0_dcc, options);
warning('on') %#ok<*WNON>
[~, R] = amredcc_step2(para_corr, RCorr, RC, D, L, 1);
alpha = para_corr(1);
beta = para_corr(2);
omega_r = para_corr(3);

para_dcc = [alpha.^2, beta.^2, abs(omega_r)+1];

%% 共分散の推定
S = zeros(K, K, T);
for t = L+1:T
    S(:,:,t) = sqrt(D(:,:,t)) * R(:,:,t) * sqrt(D(:,:,t));
end

%% 自由度の推定
warning('off') %#ok<*WNOFF>
df = fminunc(@(df) -wishlike(S(:,:,L+1:test_start-1)./df, df, RC(:,:,L+1:test_start-1)), 2*K, options);
warning('on') %#ok<*WNON>

%% 対数尤度の計算
llh = wishlike(S(:,:,L+1:test_start-1)./df, df, RC(:,:,L+1:test_start-1));

%% AICとBICの計算
aic = -2 * llh + 2 * (4 + 5 * K);
bic = -2 * llh + log(K * (T-L+1)) * (4 + 5 * K);

%% パラメータの格納
estimpara = struct();
estimpara.garch = para_garch;
estimpara.correlation = para_dcc;
estimpara.df = df;

forecast_fit = struct();
forecast_fit.covariance = S;

logL = struct();
logL.llh = llh;
logL.AIC = aic;
logL.BIC = bic;
end


%% ステップ1での尤度関数
function [llh, s] = amredcc_step1_llh(para_garch, RV, L, type)
%
% input - 
%   type : 0 (最尤法で使用)
%          1 (RVの推定で使用)
%
% output - 
%   llh : 擬似対数尤度
%   d : RVの予測値

T = size(RV,1);
gamma = para_garch(1)^2;
delta = para_garch(2)^2;
m = para_garch(3).^2;
theta = para_garch(4).^2;
omega_s = abs(para_garch(5)) + 1;

%% 1.1 RVの長期成分
S_var_vector = zeros(1,T);
beta_term_variance = zeros(1,L);

for l = 1:L
    beta_term_variance(1,l) = beta_weight(l, L, omega_s);
end

for t = L+1:T
    RV_lag = RV(t-L:t-1,1)';
    long_term = flip(beta_term_variance')' .* RV_lag;
    S_var_vector(1,t) = m + theta .* sum(long_term, 2);
end

%% 1.2 RVの短期成分
% d : RVの短期成分
% s : スケール行列の対角成分
d = zeros(1,T);
s = zeros(1,T);
for t = 1:L
    d(t) = mean(RV(1:t));
end

llhs = zeros(1,T);
for t = L+1:T
    d(t) = (1 - gamma - delta) + gamma * RV(t-1)/S_var_vector(t) + delta * d(t-1);
    s(t) = S_var_vector(1,t) * d(t);
    if type == 0
        llhs(t) = 1/2 * (-log(s(t)) - RV(t)/s(t));
    end
end

if type == 0
    llh = sum(llhs(L+1:T));
    if gamma + delta >= 1
        llh = -inf;
    end
else
    llh = 0;
end

end

%% ステップ2での尤度関数
function [llh, R] = amredcc_step2(para_dcc, RCorr, RC, D, L, type)
% 
% output - 
%   R : スケール行列の相関行列

[K, ~, T] = size(RC);

alpha = para_dcc(1)^2;
beta = para_dcc(2)^2;
omega_r = abs(para_dcc(3)) + 1;
nu = abs(para_dcc(4)) + K;

%% 2.1 RCorrの長期成分
beta_term_correlation = zeros(1,1,L);
for l = 1:L
    beta_term_correlation(1,1,l) = beta_weight(l, L, omega_r);
end
P_var = zeros(K, K, T);

for t = L+1:T
    RCorr_lag = RCorr(:,:,t-L:t-1);
    P_var(:,:,t) = sum(flip(beta_term_correlation) .* RCorr_lag, 3);
end

%% 2.2 RCorrの短期成分
R = zeros(K, K, T);
Rstar = zeros(K, K, T);
for t = 1:L
    R(:,:,t) = mean(RCorr(:,:,1:t), 3);
    Rstar(:,:,t) = R(:,:,t);
end

llhs = zeros(1,T);
for t = L+1:T
    R(:,:,t) = (1 - alpha - beta) * P_var(:,:,t) ...
        + alpha * RCorr(:,:,t-1) + beta * Rstar(:,:,t-1);
    Rstar(:,:,t) = cov_to_corr((R(:,:,t) + R(:,:,t)')./2);
    if type == 0
        in_trace = (eye(K)/Rstar(:,:,t) - eye(K)) / D(:,:,t) * RC(:,:,t) / D(:,:,t);
        llhs(t) = -nu/2 * log(det(Rstar(:,:,t))) - nu/2 * trace(in_trace);
    end
end

if type == 0
    llh = sum(llhs(L+1:T));
    if alpha + beta >= 1
        llh = -inf;
    end
else
    llh = 0;
end

end
   





%% beta weight
function phi_ell = beta_weight(l, L, omega)
% MIDAS項のbeta weight
% omega > 1である必要がある
j = 1:L;

phi_ell_upp = (1 - l/L).^(omega-1);
phi_ell_low = sum(1 - j./L).^(omega-1);

phi_ell = phi_ell_upp./phi_ell_low;

end

%% 初期パラメータ
function para = amredcc_initpara(K)
% AMReDCCモデルの最尤法のための初期パラメータ
% 銘柄分必要なパラメータは gamma, delta, m, theta, omega_sの5つ
% 残りのパラメータは一つでいい

para_variance = K * 5;
para_correlation = 4;

para = ones(para_variance + para_correlation, 1);

para(1:K) = 0.5 * para(1:K);
para(K+1:2*K) = 0.8 * para(K+1:2*K);
para(2*K+1:3*K) = 0.5 * para(2*K+1:3*K);
para(3*K+1:4*K) = 0.6 * para(3*K+1:4*K);
para(4*K+1:5*K) = 0.5 * para(4*K+1:5*K);

ind = 5*K;
para(ind+1) = 0.5;
para(ind+2) = 0.8;
para(ind+3) = 0.5;

para(ind+4) = 1+K;


end

%% パラメータの形状変換
function [gamma, delta, m, theta, omega_s, alpha, beta, omega_r, nu] = AMReDCC_transpara(para, K)


gamma = para(1:K, 1);
delta = para(K+1:2*K, 1);
m = para(2*K+1:3*K, 1);
theta = para(3*K+1:4*K, 1);
omega_s = para(4*K+1:5*K, 1);

ind = 5*K;

alpha = para(ind + 1);
beta = para(ind + 2);
omega_r = para(ind + 3);

nu = para(ind + 4);
end

%% 共分散行列から相関行列へ
function [correlation, variance] = cov_to_corr(RC)
% 共分散行列から相関行列へ
% 最尤法の途中で挟むので, 行列が半正定値にならない場合がある. 
 
[K, ~, T] = size(RC);
variance = zeros(T, K);
correlation = zeros(K, K, T);
for k = 1:K
    for t = 1:T
        variance(t,k) = RC(k,k,t);
        correlation(:,:,t) = sqrt(diag(diag(RC(:,:,t))))\RC(:,:,t)/sqrt(diag(diag(RC(:,:,t))));
    end
end
end

%% 対数尤度の計算
function [logL] = wishlike(Sigma, df, data)
% Wishart 分布の尤度関数
%   自由度の推定と対数尤度の計算に使う
%
% input : 
%   Sigma - Wishart分布のスケール行列. 推定した共分散行列を自由度で割ったもの
%   df - 自由度
%   data - 実現共分散行列
[K, ~, T] = size(Sigma);
Gamma = 0;
for i = 1:K
    Gamma = Gamma + log(gamma((df + 1 - i)/2));
end

logLs = zeros(T,1);
for t = 1:T
    logLs(t) = -df/2 * log(det(Sigma(:,:,t))) - ...
        1/2 * trace(Sigma(:,:,t)\data(:,:,t)) + ...
        (df - K - 1)/2 * log(det(data(:,:,t)));
end

logLs = logLs - df * K/2 * log(2) - Gamma;

logL = sum(logLs);
end

    
