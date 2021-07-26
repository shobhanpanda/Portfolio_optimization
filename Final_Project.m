%%
clc;
clear;
close all;
%%
%Importing and cleaning data
[aapl_17, ~, ~] = xlsread('./Data_17-18.xlsx','AAPL');
[pfe_17, ~, ~] = xlsread('./Data_17-18.xlsx','PFE');
aapl_17 = flip(aapl_17(:,1));
pfe_17 = flip(pfe_17(:,1));

[aapl_18, ~, ~] = xlsread('./Data_18-19.xlsx','AAPL');
[pfe_18, ~, ~] = xlsread('./Data_18-19.xlsx','PFE');

aapl_18 = flip(aapl_18(:,1));
pfe_18 = flip(pfe_18(:,1));

%% Optimizing weights for historical data
vol_aapl = std(aapl_17);
vol_pfe = std(pfe_17);
cov_aapl_pfe = corr(aapl_17, pfe_17);
return_aapl = (aapl_17(end)-aapl_17(1))/aapl_17(1);
return_pfe = (pfe_17(end)-pfe_17(1))/pfe_17(1);

w1 = optimvar('w1');
w2 = optimvar('w2');
%Objective function
obj = -Calculate_sharpe(w1,w2,aapl_17,pfe_17);
%Constraint
balance = w1+w2==1;
w1greater = w1>=0;
w2greater= w2>=0;

prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;

%Solve
% show(prob)
x0.w1 = 0.5;
x0.w2 = 0.5;
[sol,fval] = solve(prob,x0);

fprintf("Sharpe ratio for historical price : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f\n" ,sol.w1,sol.w2);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2], [aapl_17, pfe_17]));

%% Multidimensional GBM due to high correlation of both stocks
correlation = [1 cov_aapl_pfe; cov_aapl_pfe 1];
Return = diag([return_aapl return_pfe]);
Sigma = diag([vol_aapl vol_pfe]);
Startstate = [aapl_17(end) ;pfe_17(end)];

gbm = gbm(Return, Sigma,'StartState' ,Startstate,'correlation', correlation);
nobs = 250;
DeltaTime = 1/nobs;
% nTrials = 20000;
nTrials = 1000;
ss = simulate(gbm,nobs,  'DeltaTime', DeltaTime, 'nTrials', nTrials);

appl_future_gbm_unsqueze = squeeze(ss(:,1,:));
pfe_future_gbm_unsqueze = squeeze(ss(:,2,:));

% Taking mean of all paths at each time step
appl_future_gbm = mean(appl_future_gbm_unsqueze,2);
pfe_future_gbm = mean(pfe_future_gbm_unsqueze,2);

% Optimizing for GBM
w1 = optimvar('w1');
w2 = optimvar('w2');
%Objective function
obj = -Calculate_sharpe(w1,w2,appl_future_gbm,pfe_future_gbm);
%Constraint
balance = w1+w2==1;
w1greater = w1>=0;
w2greater= w2>=0;

prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;

%Solve
% show(prob)
x0.w1 = 0.5;
x0.w2 = 0.5;
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio for GBM price : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f\n" ,sol.w1,sol.w2);
actual_sharpe = Calculate_sharpe(sol.w1,sol.w2,aapl_18,pfe_18);
fprintf("Actual Sharpe : %f \n" ,actual_sharpe);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2], [aapl_18, pfe_18]));

%% Multidimensional CEV due to high correlation of both stocks

alpha = 0.5*ones(2,1);
cev = cev(Return, alpha,Sigma,'StartState' ,Startstate,'correlation', correlation);
DeltaTime = 1/nobs;
% nTrials = 20000;
ss = simulate(cev,nobs,  'DeltaTime', DeltaTime, 'nTrials', nTrials);

appl_future_cev_unsqueze = squeeze(ss(:,1,:));
pfe_future_cev_unsqueze = squeeze(ss(:,2,:));

% Taking mean of all paths at each time step
appl_future_cev = mean(appl_future_cev_unsqueze,2);
pfe_future_cev = mean(pfe_future_cev_unsqueze,2);

% Optimizing for CEV
w1 = optimvar('w1');
w2 = optimvar('w2');
%Objective function
obj = -Calculate_sharpe(w1,w2,appl_future_cev,pfe_future_cev);
%Constraint
balance = w1+w2==1;
w1greater = w1>=0;
w2greater= w2>=0;

prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;

%Solve
% show(prob)
x0.w1 = 0.5;
x0.w2 = 0.5;
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio for CEV price : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f\n" ,sol.w1,sol.w2);
actual_sharpe = Calculate_sharpe(sol.w1,sol.w2,aapl_18,pfe_18);
fprintf("Actual Sharpe : %f \n" ,actual_sharpe);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2], [aapl_18, pfe_18]));

%% Merton Jump Diffusion
% Estimating jumps by taking more than 2.5% change in closing price as jump
aapl_17_ret = tick2ret(aapl_17);
pfe_17_ret = tick2ret(pfe_17);

jump_barrier_aapl = prctile(aapl_17_ret,95);
jump_barrier_pfe = prctile(pfe_17_ret,95);

%For AAPL
aapl_17_jumps = (aapl_17(2:end) - aapl_17(1:end-1))./aapl_17(1:end-1);
aapl_17_jumpfreq = sum(abs(aapl_17_jumps)>jump_barrier_aapl);
aapl_17_jumpmean = mean(aapl_17_jumps>jump_barrier_aapl);
aapl_17_jumpvol =  std(aapl_17_jumps>jump_barrier_aapl);

%For PFE
pfe_17_jumps = (pfe_17(2:end) - pfe_17(1:end-1))./pfe_17(1:end-1);
pfe_17_jumpfreq = sum(abs(pfe_17_jumps)>jump_barrier_pfe);
pfe_17_jumpmean = mean(pfe_17_jumps>jump_barrier_pfe);
pfe_17_jumpvol =  std(pfe_17_jumps>jump_barrier_pfe);

JumpFreq = (aapl_17_jumpfreq+pfe_17_jumpfreq)/2;
JumpMean = [aapl_17_jumpmean ; pfe_17_jumpmean];
JumpVol = [aapl_17_jumpvol ; pfe_17_jumpvol];


merton_dynamics = merton(Return,Sigma,JumpFreq,JumpMean,JumpVol,'StartState',Startstate,'correlation', correlation);
ss = simulate(merton_dynamics,nobs,  'DeltaTime', DeltaTime, 'nTrials', nTrials);
appl_future_merton_unsqueze = squeeze(ss(:,1,:));
pfe_future_merton_unsqueze = squeeze(ss(:,2,:));

% Taking mean of all paths at each time step
appl_future_merton = mean(appl_future_merton_unsqueze,2);
pfe_future_merton = mean(pfe_future_merton_unsqueze,2);

% Optimizing for Merton
w1 = optimvar('w1');
w2 = optimvar('w2');
%Objective function
obj = -Calculate_sharpe(w1,w2,appl_future_merton,pfe_future_merton);

%Add constraint
balance = w1+w2==1;
w1greater = w1>=0;
w2greater= w2>=0;
prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;


%Solve
% show(prob)
x0.w1 = 0.5;
x0.w2 = 0.5;
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio for Merton price : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f\n" ,sol.w1,sol.w2);
actual_sharpe = Calculate_sharpe(sol.w1,sol.w2,aapl_18,pfe_18);
fprintf("Actual Sharpe : %f \n" ,actual_sharpe);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2], [aapl_18, pfe_18]));

%% Heston Model
[aapl_vol_17, ~, ~] = xlsread('./IMPLIED_VOL_1_YEAR.xlsx','AAPL');
[pfe_vol_17, ~, ~] = xlsread('./IMPLIED_VOL_1_YEAR.xlsx','PFE');
aapl_vol_17 = flip(aapl_vol_17(:,1));
pfe_vol_17 = flip(pfe_vol_17(:,1));

%Heston parameters
aapl_level = mean(aapl_vol_17);
pfe_level = mean(pfe_vol_17);
aapl_vol_vol = std(aapl_vol_17);
pfe_vol_vol = std(pfe_vol_17);
speed = 1.5;

%Simulate independently
heston_model = heston(return_aapl,speed,aapl_level,aapl_vol_vol,...
    'correlation',0,'StartState', [aapl_17(end) ;vol_aapl ]);
ss = simulate(heston_model, nobs ,'DeltaTime', DeltaTime,'nTrials', nTrials);
appl_future_heston_unsqueze = squeeze(ss(:,1,:));
appl_future_heston = mean(appl_future_heston_unsqueze,2);

heston_model = heston(return_pfe,speed,pfe_level,pfe_vol_vol,...
    'correlation',0,'StartState', [pfe_17(end);vol_pfe]);
ss = simulate(heston_model, nobs ,'DeltaTime', DeltaTime,'nTrials', nTrials);
pfe_future_heston_unsqueze = squeeze(ss(:,1,:));
pfe_future_heston = mean(pfe_future_heston_unsqueze,2);

% Optimizing for Heston
w1 = optimvar('w1');
w2 = optimvar('w2');
%Objective function
obj = -Calculate_sharpe(w1,w2,appl_future_heston,pfe_future_heston) ;

%Add constraint
balance = w1+w2==1;
w1greater = w1>=0;
w2greater= w2>=0;
prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;


%Solve
% show(prob)
x0.w1 = 0.5;
x0.w2 = 0.5;
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio for Heston price : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f\n" ,sol.w1,sol.w2);
actual_sharpe = Calculate_sharpe(sol.w1,sol.w2,aapl_18,pfe_18);
fprintf("Actual Sharpe : %f \n" ,actual_sharpe);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2], [aapl_18, pfe_18]));

%% Adding Options to the portfolio
aapl_strike = median(aapl_17);
pfe_strike =  median(pfe_17);
[risk_free_rate,~,~] = xlsread('./Option_data.xlsx','Rates');
[aapl_17_option, ~, ~] = xlsread('./Option_data.xlsx','AAPL_Strike@40@2017');
[pfe_17_option, ~, ~] = xlsread('./Option_data.xlsx','PFE_Strike@32@2017');
aapl_17_option_ts = aapl_17_option(:,6);
aapl_17_option_ts = [ones(52,1)*aapl_17_option_ts(1);aapl_17_option_ts]; %Imputing for missing data
pfe_17_option_ts = pfe_17_option(:,6);

aapl_17_option = aapl_17_option(1,6);
pfe_17_option = pfe_17_option(1,6);
risk_free_rate = mean(risk_free_rate(:,5));

%%
% GBM
aapl_option_payoff_gbm = appl_future_gbm_unsqueze(end,:) - aapl_strike;
pfe_option_payoff_gbm = pfe_future_gbm_unsqueze(end,:) - pfe_strike;


w1 = optimvar('w1');
w2 = optimvar('w2');
w3 = optimvar('w3');
w4 = optimvar('w4');
%Objective function
obj = -Calculate_sharpe2(w1,w2,w3,w4,aapl_17,pfe_17,appl_future_gbm_unsqueze,...
    pfe_future_gbm_unsqueze,aapl_option_payoff_gbm, pfe_option_payoff_gbm,...
    aapl_17_option,pfe_17_option);

%Add constraint
balance = w1+w2+w3+w4==1;
w1greater = w1>=0;
w2greater= w2>=0;
w3greater= w3>=-1;
w4greater= w4>=-1;
prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;
prob.Constraints.constr4 = w3greater;
prob.Constraints.constr5 = w4greater;

x0.w1 = 0.25;
x0.w2 = 0.25;
x0.w3 = 0.25;
x0.w4 = 0.25;
% show(prob)
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio after adding options GBM : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f  W3 : %f  W4 : %f\n" ,sol.w1,sol.w2,sol.w3,sol.w4);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2;sol.w3;sol.w4],...
    [aapl_18, pfe_18, aapl_17_option_ts, pfe_17_option_ts]));

%%
% CEV
aapl_option_payoff_cev = appl_future_cev(end) - aapl_strike;
pfe_option_payoff_cev = pfe_future_cev(end) - pfe_strike;

w1 = optimvar('w1');
w2 = optimvar('w2');
w3 = optimvar('w3');
w4 = optimvar('w4');
%Objective function
obj = -Calculate_sharpe2(w1,w2,w3,w4,aapl_17,pfe_17,appl_future_cev_unsqueze,pfe_future_cev_unsqueze,...
    aapl_option_payoff_cev, pfe_option_payoff_cev,aapl_17_option,pfe_17_option);

%Add constraint
balance = w1+w2+w3+w4==1;
w1greater = w1>=0;
w2greater= w2>=0;
w3greater= w3>=-1;
w4greater= w4>=-1;
prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;
prob.Constraints.constr4 = w3greater;
prob.Constraints.constr5 = w4greater;

x0.w1 = 0.25;
x0.w2 = 0.25;
x0.w3 = 0.25;
x0.w4 = 0.25;
% show(prob)
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio after adding options CEV : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f  W3 : %f  W4 : %f\n" ,sol.w1,sol.w2,sol.w3,sol.w4);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2;sol.w3;sol.w4],...
    [aapl_18, pfe_18, aapl_17_option_ts, pfe_17_option_ts]));

%% Merton
aapl_option_payoff_merton = appl_future_merton(end) - aapl_strike;
pfe_option_payoff_merton = pfe_future_merton(end) - pfe_strike;

w1 = optimvar('w1');
w2 = optimvar('w2');
w3 = optimvar('w3');
w4 = optimvar('w4');
%Objective function
obj = -Calculate_sharpe2(w1,w2,w3,w4,aapl_17,pfe_17,appl_future_merton_unsqueze,...
    pfe_future_merton_unsqueze,...
    aapl_option_payoff_merton, pfe_option_payoff_merton,aapl_17_option,pfe_17_option);

%Add constraint
balance = w1+w2+w3+w4==1;
w1greater = w1>=0;
w2greater= w2>=0;
w3greater= w3>=-1;
w4greater= w4>=-1;
prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;
prob.Constraints.constr4 = w3greater;
prob.Constraints.constr5 = w4greater;

x0.w1 = 0.25;
x0.w2 = 0.25;
x0.w3 = 0.25;
x0.w4 = 0.25;
% show(prob)
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio after adding options Merton : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f  W3 : %f  W4 : %f\n" ,sol.w1,sol.w2,sol.w3,sol.w4);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2;sol.w3;sol.w4],...
    [aapl_18, pfe_18, aapl_17_option_ts, pfe_17_option_ts]));

%% Heston
aapl_option_payoff_heston = appl_future_heston(end) - aapl_strike;
pfe_option_payoff_heston = pfe_future_heston(end) - pfe_strike;

w1 = optimvar('w1');
w2 = optimvar('w2');
w3 = optimvar('w3');
w4 = optimvar('w4');
%Objective function
obj = -Calculate_sharpe2(w1,w2,w3,w4,aapl_17,pfe_17,appl_future_heston_unsqueze,...
    pfe_future_heston_unsqueze,...
    aapl_option_payoff_heston, pfe_option_payoff_heston,aapl_17_option,pfe_17_option);

%Add constraint
balance = w1+w2+w3+w4==1;
w1greater = w1>=0;
w2greater= w2>=0;
w3greater= w3>=-1;
w4greater= w4>=-1;
prob = optimproblem('Objective',obj);
prob.Constraints.constr1 = balance;
prob.Constraints.constr2 = w1greater;
prob.Constraints.constr3 = w2greater;
prob.Constraints.constr4 = w3greater;
prob.Constraints.constr5 = w4greater;

x0.w1 = 0.25;
x0.w2 = 0.25;
x0.w3 = 0.25;
x0.w4 = 0.25;
[sol,fval] = solve(prob,x0);
fprintf("Sharpe ratio after adding options Heston : %f\n" ,-fval);
fprintf("W1 : %f  W2 : %f  W3 : %f  W4 : %f\n" ,sol.w1,sol.w2,sol.w3,sol.w4);
fprintf("Actual Return: %f\n", realized_return([sol.w1;sol.w2;sol.w3;sol.w4],...
    [aapl_18, pfe_18, aapl_17_option_ts, pfe_17_option_ts]));

%% Which Model represents the future? 

% Taking the standard deviation of the difference
% GBM
error_gbm = sqrt((var(aapl_18 - appl_future_gbm) + var(pfe_18 - pfe_future_gbm))/2)
% CEV
error_cev =  sqrt((var(aapl_18 - appl_future_cev) + var(pfe_18 - pfe_future_cev))/2)
% Merton
error_merton =  sqrt((var(aapl_18 - appl_future_merton) + var(pfe_18 - pfe_future_merton))/2)
% Heston
error_heston =  sqrt((std(aapl_18 - appl_future_heston) + std(pfe_18 - pfe_future_heston))/2)

%% Sharpe 2 for portfolio with option
function sharpe_ratio2 = Calculate_sharpe2(w1,w2,w3,w4,aapl_17,pfe_17,aapl_fut,pfe_fut,...
    aapl_po, pfe_po,aapl_17_option,pfe_17_option)
cost = w1*aapl_17(end) + w2*pfe_17(end)+w3*aapl_17_option(1)+w4*pfe_17_option(1);
portfolio_value = w1*aapl_fut(end,:)+ w2*pfe_fut(end,:)+w3*aapl_po+w4*pfe_po;
portfolio_return = (portfolio_value - cost)/cost;
mu = mean(portfolio_return);
sigma = sqrt(mean(portfolio_return.^2)-mean(portfolio_return)^2); % Standard Deviation

sharpe_ratio2 = mu/sigma;

end

%% Sharpe ratio
function sharpe_ratio = Calculate_sharpe(w1,w2,aapl_17,pfe_17)
portfolio_return = w1*((aapl_17(end)-aapl_17(1))/aapl_17(1)) + ...
    w2*((pfe_17(end)-pfe_17(1))/pfe_17(1));

vol_appl = std(aapl_17);
vol_pfe = std(pfe_17);
cov_aapl_pfe = corr(aapl_17, pfe_17);

% Volatility(Standard Deviation) of Volatility
portfolio_vol = sqrt((w1*vol_appl)^2 + (w2*vol_pfe)^2 + 2*w1*w2*cov_aapl_pfe*vol_appl*vol_pfe);
sharpe_ratio = portfolio_return/portfolio_vol;

end

%% Calculate Return

function return_calc = realized_return(w,P)

ret = P(end,:)./P(1,:)-1;
return_calc = ret*w;
end

