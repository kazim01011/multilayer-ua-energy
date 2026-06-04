A = readtable('data/ablation_attention_summary.csv');
D = readmatrix('data/layer_dissimilarity.csv','NumHeaderLines',1);
figure('Color','w','Position',[100 100 1200 720]);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; bar(categorical(A.layer),A.leave_one_out_gain); ylabel('Full ES - ES without layer'); grid on;
nexttile; bar(categorical(A.layer),A.single_layer_gain); ylabel('Single-layer ES - Flat ES'); grid on;
nexttile; bar(categorical(A.layer),A.attention); ylabel('Attention weight'); grid on;
nexttile; imagesc(D(:,2:end)); colorbar; title('Layer dissimilarity'); axis square;
savefig('fig_results_layer_analysis.fig');
