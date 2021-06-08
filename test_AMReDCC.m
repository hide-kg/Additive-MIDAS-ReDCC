clc
clear
close all
load('vechRMK_63_20142020.mat')
for t = 1:1624
    RMK(:,:,t) = ivech(MX(:,t));
end

%stock = [5 10 15 20 25 30];
stock = [32 63];
RC = RMK(stock, stock, :);
L = 240;
test_start = 1000;

%% AMReDCC
tic
[estimpara_am_new, forecast_fit_am, logL_am] = AMReDCC(RC, L, test_start);
S_am = forecast_fit_am.covariance;
toc
%% AMREDCC 2ステップ
[estimpara_am2, forecast_fit_am2, logL_am2] = AMReDCC_2step(RC, L, test_start);
S_am2 = forecast_fit_am2.covariance;

%% MMReDCC
[estimpara_mm, forecast_fit_mm, logL_mm] = MMReDCC_onestep(RC, L, test_start);
S_mm = forecast_fit_mm.covariance;

%% RE-DCC
order_var = [1,1];
order_corr = [1,1];
[para_garch_vol3, para_dcc_vol3, S_redcc, llh] = re_dcc_estimate_vol3(RC, test_start, order_var, order_corr);

%% HAR
order = [1,1];
[para_garch, para_dcc, S_har] = har_re_dcc_estimate_vol4(RC, test_start, order);

%% MIDAS-CAW
K = length(stock);
order = [1,1];
para0 = make_initpara_MIDAS(K,order);
type = 1;
[~, ~, ~, ~, ~, ~, S_caw, ~, ~] = ...
    MIDAS_CAW_estimate_vol3(RC, order, test_start, para0, type);

%% EWMA
lambda = 0.94;
S_ewma = RiskMetrics(RC, lambda);

%% Stein
test_start = 1166;
[loss_am_stein, losses_am_stein] = Stein_loss(RC(:,:,test_start:end), S_am(:,:,test_start:end));
[loss_am2_stein, losses_am2_stein] = Stein_loss(RC(:,:,test_start:end), S_am2(:,:,test_start:end));
[loss_mm_stein, losses_mm_stein] = Stein_loss(RC(:,:,test_start:end), S_mm(:,:,test_start:end));
[loss_redcc_stein, losses_redcc_stein] = Stein_loss(RC(:,:,test_start:end), S_redcc(:,:,test_start:end));
[loss_har_stein, losses_har_stein] = Stein_loss(RC(:,:,test_start:end), S_har(:,:,test_start:end));
[loss_caw_stein, losses_caw_stein] = Stein_loss(RC(:,:,test_start:end), S_caw(:,:,test_start:end));
[loss_ewma_stein, losses_ewma_stein] = Stein_loss(RC(:,:,test_start:end), S_ewma(:,:,test_start:end));

%% Fro
loss_am_fro = Frobenius(RC(:,:,test_start:end), S_am(:,:,test_start:end));
loss_am2_fro = Frobenius(RC(:,:,test_start:end), S_am2(:,:,test_start:end));
loss_mm_fro = Frobenius(RC(:,:,test_start:end), S_mm(:,:,test_start:end));
loss_redcc_fro = Frobenius(RC(:,:,test_start:end), S_redcc(:,:,test_start:end));
loss_har_fro = Frobenius(RC(:,:,test_start:end), S_har(:,:,test_start:end));
loss_caw_fro = Frobenius(RC(:,:,test_start:end), S_caw(:,:,test_start:end));
loss_ewma_fro = Frobenius(RC(:,:,test_start:end), S_ewma(:,:,test_start:end));

%% QLIKE
QLIKE_am = QLIKE(RC(:,:,test_start:end), S_am(:,:,test_start:end));
QLIKE_am2 = QLIKE(RC(:,:,test_start:end), S_am2(:,:,test_start:end));
QLIKE_mm = QLIKE(RC(:,:,test_start:end), S_mm(:,:,test_start:end));
QLIKE_redcc = QLIKE(RC(:,:,test_start:end), S_redcc(:,:,test_start:end));
QLIKE_har = QLIKE(RC(:,:,test_start:end), S_har(:,:,test_start:end));
QLIKE_caw = QLIKE(RC(:,:,test_start:end), S_caw(:,:,test_start:end));
QLIKE_ewma = QLIKE(RC(:,:,test_start:end), S_ewma(:,:,test_start:end));


%% 
disp({'AMReDCC', 'MMReDCC', 'Re-DCC', 'HAR', 'EWMA', 'AMReDCC 2ステップ', 'MIDAS-CAW'})
losses_stein = [loss_am_stein, loss_mm_stein, loss_redcc_stein, loss_har_stein, loss_ewma_stein, loss_am2_stein, loss_caw_stein]
losses_fro = [mean(loss_am_fro), mean(loss_mm_fro), mean(loss_redcc_fro), mean(loss_har_fro), mean(loss_ewma_fro), mean(loss_am2_fro), mean(loss_caw_fro)]
losses_QLIKE = [mean(QLIKE_am), mean(QLIKE_mm), mean(QLIKE_redcc), mean(QLIKE_har), mean(QLIKE_ewma), mean(QLIKE_am2), mean(QLIKE_caw)]

alpha = 0.2;
num_block = 25;
inte = 1000;
%loss_stein = [losses_am_stein, losses_mm_stein, losses_redcc_stein, losses_har_stein, losses_ewma_stein];
%loss_fro = [loss_am_fro, loss_mm_fro, loss_redcc_fro, loss_har_fro, loss_ewma_fro];
%loss_qlike = [QLIKE_am, QLIKE_mm, QLIKE_redcc, QLIKE_har, QLIKE_ewma];

loss_stein = [losses_am_stein, losses_mm_stein, losses_redcc_stein, losses_ewma_stein];
loss_fro = [loss_am_fro, loss_mm_fro, loss_redcc_fro, loss_ewma_fro];
loss_qlike = [QLIKE_am, QLIKE_mm, QLIKE_redcc, QLIKE_ewma];

[includer_stein, p_val_stein, excluder_stein] = mcs(loss_stein, alpha, inte, num_block, 'BLOCK');
[includer_fro, p_val_fro, excluder_fro] = mcs(loss_fro, alpha, inte, num_block, 'BLOCK');
[includer_qlike, p_val_qlike, excluder_qlike] = mcs(loss_qlike, alpha, inte, num_block, 'BLOCK');

model_stein = [excluder_stein;includer_stein];
result_stein = [model_stein, p_val_stein]

model_fro = [excluder_fro; includer_fro];
result_fro = [model_fro, p_val_fro]

model_qlike = [excluder_qlike; includer_qlike];
result_qlike = [model_qlike, p_val_qlike]

%% plot
i1 = 2;
i2 = 2;
t_start = 242;

s_am = cov_to_var(S_am, 241, 1624, i1, i2);
s_mm = cov_to_var(S_mm, 241, 1624, i1, i2);
S_Redcc = cov_to_var(S_redcc, 241, 1624, i1, i2);
s_har = cov_to_var(S_har, 241, 1624, i1, i2);
ewma = cov_to_var(S_ewma, 241, 1624, i1, i2);
RV   = cov_to_var(RC, 241, 1624, i1, i2);

figure
plot(RV(t_start:end),'b')
hold on
plot(s_mm(t_start:end), 'r', 'LineWidth',1.5)
xline(759)
title('MMRe-DCC')

figure
plot(RV(t_start:end),'b')
hold on
plot(s_am(t_start:end), 'r', 'LineWidth',1.5)
xline(759)
title('AMRe-DCC')

figure
plot(RV(t_start:end),'b')
hold on
plot(s_har(t_start:end), 'r', 'LineWidth',1.5)
xline(759)
title('HAR Re-DCC')

figure
plot(RV(t_start:end), 'b')
hold on
plot(S_Redcc(t_start:end), 'r', 'LineWidth',1.5)
xline(759)
title('Re-DCC')

figure
plot(RV(t_start:end), 'b')
hold on
plot(ewma(t_start:end), 'r', 'LineWidth', 1.5)
xline(759)
title('EWMA')

