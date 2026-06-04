S = readmatrix('data/snapshot_sinr.csv');
U = readtable('data/snapshot_ues.csv');
B = readtable('data/snapshot_bs.csv');
figure('Color','w','Position',[100 100 1300 520]);
tiledlayout(1,3,'TileSpacing','compact');
nexttile; imagesc(10*log10(max(S,1e-12))); colorbar; xlabel('BS'); ylabel('UE'); title('All-on SINR (dB)');
nexttile; gscatter(U.x,U.y,U.oracle); hold on; scatter(B.x,B.y,120,'ks','filled'); axis equal; title('Reference association'); grid on;
nexttile; gscatter(U.x,U.y,U.ml_gcn); hold on; scatter(B.x,B.y,120,'ks','filled'); axis equal; title('ML-GCN association'); grid on;
savefig('fig_results_representative_snapshot.fig');
