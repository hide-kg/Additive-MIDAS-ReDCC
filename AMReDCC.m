
%% メイン関数
function [estimpara, forecast_fit, logL] = AMReDCC(RC, L, test_start)
%
% 2021/3/8
%   Bauwens et al. (2016)のReDCCモデルのMIDAS拡張型モデルの実装
%   本プログラムは特にAdditive型モデルの推定をする
%   推定方法は1-ステップ型とする. 
%
% input : 
%   RC - 実現共分散行列
%   test_start - 予測期間の始まり
%
% output : 
%   para_garch - GARCH式のパラメータ
%   para_dcc - 条件付き相関のパラメータ
%
% variable : 
%   para0 - 初期パラメータ
%   gamma, delta - 条件付き分散式のパラメータ
%   m, theta, omega - 条件付き分散式のMIDAS回帰のパラメータ(ただし, omega>1)

[K, ~, T] = size(RC);

para0 = amredcc_initpara(K);
[RCorr, RV] = cov_to_corr(RC);

%% パラメータ推定
warning('off') %#ok<*WNOFF>
options = optimoptions('fminunc','Display', 'off');
ll = @(x0) -AMReDCC_llh(x0, RC, RCorr, RV, L, test_start, 0);
[para] = fminunc(ll, para0, options);
warning('on') %#ok<*WNON>

[gamma, delta, m, theta, omega_s, alpha, beta, omega_r, nu] = AMReDCC_transpara(para, K);
%% パラメータの制約

gamma = gamma.^2;
delta = delta.^2;
alpha = alpha^2;
beta = beta^2;

m = m.^2;
theta = theta.^2;
nu = abs(nu) + K;

omega_s = abs(omega_s) + 1;
omega_r = abs(omega_r) + 1;

%% 共分散行列の予測と対数尤度の取得
[~, S, S_variance, R] = AMReDCC_llh(para, RC, RCorr, RV, L, test_start, 1);

% L+2なのは, MMReDCCモデルに合わせるため
llh = wishlike(S(:,:,L+2:test_start-1)./nu, nu, RC(:,:,L+2:test_start-1));

%% AICとBICの計算
aic = -2 * llh + 2 * (4 + 5 * K);
bic = -2 * llh + log(K * (T-L+1)) * (4 + 5 * K);

estimpara = struct();
estimpara.variance_short = [gamma, delta];
estimpara.variance_long = [m, theta, omega_s];
estimpara.correlation_short = [alpha, beta];
estimpara.correlation_long = omega_r;
estimpara.degree_of_free = nu;

forecast_fit = struct();
forecast_fit.covariance = S;
forecast_fit.variance = S_variance;
forecast_fit.correlation = R;

logL = struct();
logL.llh = llh;
logL.AIC = aic;
logL.BIC = bic;

end

%% 尤度関数
function [llh, S, Scale_diag, Rstar] = AMReDCC_llh(para0, RC, RCorr_input, RV_input, L, test_start, type)
%
% AMReDCCモデルの対数尤度関数のプログラム
% 推定は1-ステップ
% 
% input : 
%   para0 - 初期パラメータ
%   RC - 実現共分散行列
%   L - 長期ラグ
%   test_start - 予測の始まり
%   type - 推定(0)か予測(1)
%
% output : 
%   llh - 対数尤度
%   S - 共分散行列の予測値
%   S_var - 分散の長期成分
%   P_var - 相関の長期成分
%
% variable :
%   L - 240 (1年の日数)

%% 初期設定

if type == 0
    for t = 1:test_start-1
        RCorr(:,:,t) = RCorr_input(:,:,t);
        RV(t,:) = RV_input(t,:);
    end
elseif type == 1
    for t = 1:size(RC, 3)
        RCorr(:,:,t) = RCorr_input(:,:,t);
        RV(t,:) = RV_input(t,:);
    end
    insample_period = test_start-1;
end
        
[K, ~, T] = size(RCorr);


[gamma, delta, m, theta, omega_s, alpha, beta, omega_r, nu] = AMReDCC_transpara(para0, K);

gamma = gamma.^2;
delta = delta.^2;
alpha = alpha^2;
beta = beta^2;

m = m.^2;
theta = theta.^2;
nu = abs(nu) + K;

omega_s = abs(omega_s) + 1;
omega_r = abs(omega_r) + 1;

% 条件付き分散のモデル化
% データのL日目まではモデル生成のために使う

%{
%% 1.1. RVの長期成分
S_var_vector = zeros(K, T);
S_var = zeros(K, K, T);

beta_term_variance = zeros(K, L);

for l = 1:L
    beta_term_variance(:,l) = beta_weight(l, L, omega_s);
end

for k = 1:K
    for t = L+1:T
        RV_lag = RV(t-L:t-1,k)';
        long_term = flip(beta_term_variance(k,:)')' .* RV_lag;
        S_var_vector(k,t) = m(k) + theta(k) .* sum(long_term,2);
        S_var(k,k,t) = S_var_vector(k,t);
    end
end
        

%% 1.2. RVの短期成分
S_diag_vector = zeros(K, T);
for t = 1:L
    S_diag_vector(:,t) = diag(mean(RC(:,:,1:t), 3));
end
    
S_diag = zeros(K, K, T);
for k = 1:K
    for t = L+1:T
        S_diag_vector(k,t) = (1 - gamma(k) - delta(k)) ...
            + gamma(k) .* (RV(t-1,k)./S_var(k,k,t)) + delta(k) .* S_diag_vector(k,t-1);
        S_diag(k,k,t) = S_diag_vector(k,t);
    end
end
%% 1.3. スケール行列の分散の計算

Scale_diag = zeros(K,K,T);
for t = L+1:T
    Scale_diag(:,:,t) = S_var(:,:,t) .* S_diag(:,:,t);
end
%}

%% 1. RVの計算
% 2021/3/29
%   ループを一括りにした
S_var_vector = zeros(K, T);
S_var = zeros(K, K, T);
beta_term_variance = zeros(K, L);
S_diag_vector = zeros(K, T);
S_diag = zeros(K, K, T);
Scale_diag = zeros(K,K,T);
for l = 1:L
    beta_term_variance(:,l) = beta_weight(l, L, omega_s);
    S_diag_vector(:,l) = diag(mean(RC(:,:,1:l), 3));
end
for k = 1:K
    for t = L+1:T
        %% 1.1. RVの長期成分
        RV_lag = RV(t-L:t-1,k)';
        long_term = flip(beta_term_variance(k,:)')' .* RV_lag;
        S_var_vector(k,t) = m(k) + theta(k) .* sum(long_term, 2);
        S_var(k,k,t) = S_var_vector(k,t);
        
        %% 1.2. RVの短期成分
        S_diag_vector(k,t) = (1 - gamma(k) - delta(k)) ...
            + gamma(k) .* (RV(t-1,k)./S_var(k,k,t)) + delta(k) .* S_diag_vector(k,t-1);
        S_diag(k,k,t) = S_diag_vector(k,t);
        
        %% 1.3. スケール行列の分散の計算
        Scale_diag(k,k,t) = S_var(k,k,t) * S_diag(k,k,t);
    end
end

%{
%% 2.1. RCorrの長期成分
beta_term_correlation = zeros(1,1,L);
for l = 1:L
    beta_term_correlation(1,1,l) = beta_weight(l, L, omega_r);
end
P_var = zeros(K, K, T);

for t = L+1:T
    RCorr_lag = RCorr(:,:,t-L:t-1);
    P_var(:,:,t) = sum(flip(beta_term_correlation) .* RCorr_lag, 3);
end


%% 2.2. RCorrの短期成分
R = zeros(K, K, T);
Rstar = zeros(K, K, T);
for t = 1:L
    R(:,:,t) = mean(RCorr(:,:,1:t), 3);
    Rstar(:,:,t) = R(:,:,t);
end
for t = L+1:T
    R(:,:,t) = (1 - alpha - beta) * P_var(:,:,t) ...
        + alpha * RCorr(:,:,t-1) + beta * Rstar(:,:,t-1);
    Rstar(:,:,t) = cov_to_corr((R(:,:,t) + R(:,:,t)')./2);
end
%}
%% 2. RCの計算
% 2021/3/29
%   ループを一括りにした
beta_term_correlation = zeros(1,1,L);
P_var = zeros(K, K, T);
R = zeros(K, K, T);
Rstar = zeros(K, K, T);
for l = 1:L
    beta_term_correlation(1,1,l) = beta_weight(l, L, omega_r);
    R(:,:,l) = mean(RCorr(:,:,1:t), 3);
    Rstar(:,:,t) = R(:,:,t);
end
for t = L+1:T
    %% 2.1. RCorrの長期成分
    RCorr_lag = RCorr(:,:,t-L:t-1);
    P_var(:,:,t) = sum(flip(beta_term_correlation) .* RCorr_lag, 3);
    
    %% 2.2. RCorrの短期成分
    R(:,:,t) = (1 - alpha - beta) * P_var(:,:,t) ...
        + alpha * RCorr(:,:,t-1) + beta * Rstar(:,:,t-1);
    Rstar(:,:,t) = cov_to_corr((R(:,:,t) + R(:,:,t)')./2);
end

%% 3. 共分散行列の計算と対数尤度の計算
S = zeros(K, K, T);
llhs = zeros(T, 1);
if type == 0
    for t = L+1:T
        S(:,:,t) = sqrt(Scale_diag(:,:,t)) * Rstar(:,:,t) * sqrt(Scale_diag(:,:,t));
        llhs(t,1) = -nu/2 * (log(det(S(:,:,t))) + trace(S(:,:,t)\RC(:,:,t)));
    end
    llh = sum(llhs(L+1:T));
elseif type == 1
    for t = L+1:T
        S(:,:,t) = sqrt(Scale_diag(:,:,t)) * Rstar(:,:,t) * sqrt(Scale_diag(:,:,t));
    end
    for t = L+1:insample_period
        llhs(t,1) = -nu/2 * (log(det(S(:,:,t))) + trace(S(:,:,t)\RC(:,:,t)));
    end
    % L+1+1としているのは, MMReDCCモデルの尤度と合わせるため. 
    llh = sum(llhs(L+1+1:insample_period));
end
    


%% パラメータの確認
sum_para = zeros(K, 1);
for k = 1:K
    sum_para(k) = gamma(k) + delta(k);
    if sum_para(k) >= 1
        llh = -inf;
    end
end

if alpha + beta >= 1
        llh = -inf;
end

end

%% beta weight
function phi_ell = beta_weight(l, L, omega)
% MIDAS項のbeta weight
% omega > 1である必要がある
j = 1:L;

phi_ell_upp = (1 - l/L).^(omega-1);
phi_ell_low = sum((1 - j./L).^(omega-1));

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

    
    
    
    
    
    
