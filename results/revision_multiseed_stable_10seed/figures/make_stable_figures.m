% Auto-generated from revision_multiseed_stable_10seed summary files
policies = {'Oracle','Flat MLP stable','Attn ML-GCN stable','ML-GCN stable','Greedy sleep','ML-GCN pruned','Agg GCN stable','RSRP sleep'};
energy_saving_with_switching_vs_all_on_mean = [0.52053776 0.50555833 0.50415967 0.50389052 0.48399076 0.48471375 0.49353662 0.45372240];
energy_saving_with_switching_vs_all_on_std = [0.02025398 0.02778804 0.02402305 0.02308019 0.02641476 0.02496935 0.02113364 0.04288124];
switch_events_mean = [2.96500000 2.27500000 2.35000000 2.33500000 3.28500000 2.89000000 2.43500000 3.31000000];
switch_events_std = [0.30736334 0.31024184 0.27182511 0.27289599 0.29911908 0.27264140 0.38733419 0.26645825];
figure; bar(energy_saving_with_switching_vs_all_on_mean); hold on; errorbar(1:numel(policies), energy_saving_with_switching_vs_all_on_mean, energy_saving_with_switching_vs_all_on_std, 'k.', 'LineWidth', 1); hold off; grid on; ylabel('Energy saving with switching penalty'); set(gca, 'XTick', 1:numel(policies), 'XTickLabel', policies, 'XTickLabelRotation', 30);
figure; bar(switch_events_mean); hold on; errorbar(1:numel(policies), switch_events_mean, switch_events_std, 'k.', 'LineWidth', 1); hold off; grid on; ylabel('Mean BS state changes per snapshot'); set(gca, 'XTick', 1:numel(policies), 'XTickLabel', policies, 'XTickLabelRotation', 30);
