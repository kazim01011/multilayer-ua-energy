T = readtable('data/surface_energy.csv');
qos = unique(T.qos_mbps)';
ues = unique(T.num_ues)';
shadows = unique(T.shadowing_db)';
[Q,N] = meshgrid(qos, ues);
figure('Color','w','Position',[100 100 1000 720]); hold on;
colors = [1 0.15 0.05; 0 0.80 0.80; 0.05 0.20 0.95];
for s = 1:numel(shadows)
    S = T(T.shadowing_db == shadows(s),:);
    Z = nan(numel(ues), numel(qos));
    for i = 1:numel(ues)
        for j = 1:numel(qos)
            row = S(S.num_ues == ues(i) & S.qos_mbps == qos(j),:);
            Z(i,j) = row.energy_saving_vs_all_on(1);
        end
    end
    surf(Q,N,Z,'FaceColor',colors(s,:),'FaceAlpha',0.78,'EdgeColor',[0.55 0.55 0.55]);
end
xlabel('QoS threshold (Mbps)');
ylabel('Number of UEs');
zlabel('Energy saving');
legend(strcat(string(shadows),' dB shadowing'),'Location','northeast');
grid on; view(42,28);
savefig('fig_results_surface_energy.fig');
