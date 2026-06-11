% Auto-generated from results/revision_multiseed_pruned_10seed/summary_mean.csv
policies = {'Oracle','Greedy sleep','ML-GCN\npruned','Attn ML-GCN\npruned','Flat MLP\npruned','Agg GCN\npruned','ML-GCN','Agg GCN','Flat MLP','RSRP sleep','RSRP'};
energy_saving_vs_all_on_mean = [0.62277914 0.59726662 0.58436892 0.58431947 0.58385177 0.58171011 0.52465322 0.52492762 0.52236414 0.56786033 0.17933029];
energy_saving_vs_all_on_std = [0.01977644 0.02599310 0.02413546 0.02527743 0.02532307 0.02785575 0.03002191 0.02519758 0.02928227 0.03829292 0.01194659];
energy_gap_vs_oracle_mean = [0.00000000 0.07197569 0.10645434 0.10574117 0.10440158 0.11102312 0.27423566 0.27124740 0.27614773 0.13858992 1.25490477];
energy_gap_vs_oracle_std = [0.00000000 0.02428160 0.02549005 0.02656195 0.01634398 0.03026064 0.04462213 0.02309110 0.02630536 0.05396391 0.07909999];

figure; bar(energy_saving_vs_all_on_mean); hold on; errorbar(1:numel(policies), energy_saving_vs_all_on_mean, energy_saving_vs_all_on_std, 'k.', 'LineWidth', 1); hold off; grid on; ylabel('Energy saving vs. all-on network'); set(gca, 'XTick', 1:numel(policies), 'XTickLabel', policies, 'XTickLabelRotation', 35);
figure; bar(energy_gap_vs_oracle_mean); hold on; errorbar(1:numel(policies), energy_gap_vs_oracle_mean, energy_gap_vs_oracle_std, 'k.', 'LineWidth', 1); hold off; grid on; ylabel('Energy gap relative to oracle'); set(gca, 'XTick', 1:numel(policies), 'XTickLabel', policies, 'XTickLabelRotation', 35);
