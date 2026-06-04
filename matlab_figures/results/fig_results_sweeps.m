T = readtable('data/sweep_summary.csv');
policies = {'oracle','ml_gcn','attn_ml_gcn','agg_gcn','load_aware','rsrp'};
figure('Color','w','Position',[100 100 1200 760]);
tiledlayout(2,2,'TileSpacing','compact');
settings = {'bandwidth','qos','density','shadowing'};
ylabels = {'Energy saving','Energy saving','Energy saving','Served ratio'};
metrics = {'energy_saving_vs_all_on','energy_saving_vs_all_on','energy_saving_vs_all_on','served_ratio'};
xlabels = {'Bandwidth (MHz)','QoS target (Mbps)','Number of UEs','Shadowing std. (dB)'};
for k=1:4
    nexttile; hold on;
    for p=1:numel(policies)
        S = T(strcmp(T.sweep,settings{k}) & strcmp(T.policy,policies{p}),:);
        S = sortrows(S,'value');
        plot(S.value,S.(metrics{k}),'-o','LineWidth',1.8);
    end
    xlabel(xlabels{k}); ylabel(ylabels{k}); grid on;
end
legend(policies,'Location','bestoutside');
savefig('fig_results_sweeps.fig');
