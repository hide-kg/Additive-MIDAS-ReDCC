
clear
close all
load('vechRMK_63_20142020.mat')
for t = 1:1624
    RMK(:,:,t) = ivech(MX(:,t));
end

stock = [5 10 15 20 25 30 40 50 60];
%stock = [5 10];
RC = RMK(stock, stock, :);
L = 240;
test_start = 1000;
tic
[estimpara_am2, forecast_fit_am2, logL_am2] = AMReDCC_2step(RC, L, test_start);
toc
S_am = forecast_fit_am2.covariance;

%%
i1 = 1;
i2 = 1;
t_start = 242;
for t = 241:1624
    s_am(t) = S_am(i1,i2,t);
    RV(t) = RC(i1,i2,t);
end
figure
plot(RV(t_start:end),'b')
hold on
plot(s_am(t_start:end), 'r', 'LineWidth',1.5)
xline(759)
title('AMRe-DCC')
