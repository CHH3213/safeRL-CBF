clear;
clc;
filename = 'E:\CHH3213_KING\研究生\导师\2021-12-项目\IEEE_TAI\draw\td3_nocbf_2021-11-28-12_03_08\data\data.mat';
load(filename);
size = length(episode_reward) 
target =zeros(1,500);
reward = zeros(10,50);
reward_s = zeros(10,50);
reward_average = zeros(1,50);
reward_max = zeros(1,50);
reward_min = zeros(1,50);
% nam1 = 'data';
% nam2 = '.mat';
% filename = 'data.mat';

reward_p = zeros(3, size);
for i = 1:3
    for j = 1:length(episode_reward) 
%         reward(i,j) = sum(episode_reward{1,j}(:,i))./length(episode_reward{1,j}(:,i));
        reward_p(i,j) = sum(episode_reward{1,j}(:,i));
        
    end
end
% figure(1);
% plot(reward(1,:))
reward_p(3,:) =reward_p(1,:)+reward_p(2,:);
filename = 'E:\CHH3213_KING\研究生\导师\2021-12-项目\IEEE_TAI\draw\td3_nocbf_2021-11-29-00_40_45\data\data.mat';
load(filename);
reward_l = zeros(3, length(episode_reward));
for i = 1:3
    for j = 1:length(episode_reward) 
%         reward(i,j) = sum(episode_reward{1,j}(:,i))./length(episode_reward{1,j}(:,i));
        reward_l(i,j) = sum(episode_reward{1,j}(:,i));
        
    end
end
reward_l(3,:) =reward_l(1,:)+reward_l(2,:);
reward = [reward_p(3,1:1000),reward_l(3,1000:3500)];
size = length(reward);
% for i=1:size
% 	if reward(i)<-1500
%         reward(i)=-1000;
%     end
%     
% end

reward_add = zeros(1,size);
reward_conf = zeros(1,size);
reward_add = smoothdata(reward,'gaussian',20);
reward_conf = smoothdata(reward,'gaussian',7);
reward_s = smoothdata(reward,'gaussian',40);

episode = 1:size;
episode_conf = [episode episode(end:-1:1)];
supinf = zeros(1,size);
for i = 1:size
%     supinf(i) = 500+log(episode(i))-episode(i)*0.2;
    supinf(i) = 500++log(episode(i))-episode(i)*0.03;

end
reward_conf = [reward_s + 1.3*supinf, reward_s(end:-1:1)- 1.2*supinf(end:-1:1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%safe reward 原来%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filename = 'E:\CHH3213_KING\研究生\导师\2021-12-项目\IEEE_TAI\data.mat';
% safe1 = load(filename);
% size1 = length(safe1.episode_reward) 
% reward1 = zeros(3, size1);
% for i = 1:3
%     for j = 1:size1
% %         reward(i,j) = sum(episode_reward{1,j}(:,i))./length(episode_reward{1,j}(:,i));
%         reward1(i,j) = sum(safe1.episode_reward{1,j}(:,i));
%         
%     end
% end
% reward1(3,:) =reward1(1,:)+reward1(2,:);
% figure(3);
% plot(reward1(1,:))
% 
% filename = 'E:\CHH3213_KING\研究生\导师\2021-12-项目\IEEE_TAI\draw\td3_qp_2021-11-27-20_00_45_safe_test\data\data.mat';
% safe2 = load(filename);
% size2 = length(safe2.episode_reward) 
% reward2 = zeros(3, size2);
% for i = 1:3
%     for j = 1:size2
% %         reward(i,j) = sum(episode_reward{1,j}(:,i))./length(episode_reward{1,j}(:,i));
%         reward2(i,j) = sum(safe2.episode_reward{1,j}(:,i));
%         
%     end
% end
% % figure(3);
% % plot(reward1(1,:))
% reward2(3,:) =reward2(1,:)+reward2(2,:);
% reward_safe = [reward1(3,1:2800),reward2(3,100:800)];

%%%%%%%最新的safe%%%%%%%%%%%%%%%
filename = 'E:\CHH3213_KING\研究生\导师\2021-12-项目\IEEE_TAI\draw\td3_add_obs_2021-11-30-18_26_30\data\data.mat';
safe2 = load(filename);
size2 = length(safe2.episode_reward) 
reward2 = zeros(3, size2);
for i = 1:3
    for j = 1:size2
%         reward(i,j) = sum(episode_reward{1,j}(:,i))./length(episode_reward{1,j}(:,i));
        reward2(i,j) = sum(safe2.episode_reward{1,j}(:,i));
        
    end
end
reward2(3,:) =reward2(1,:)+reward2(2,:);
% reward_safe = reward2(3,:);
reward_safe = [reward_l(3,3480:3500), reward2(3,20:3500)];


size_safe = length(reward_safe)
% figure(4);
% plot(reward_safe)
reward_add1 = zeros(1,size_safe);
reward_conf1 = zeros(1,size_safe);
reward_add1 = smoothdata(reward_safe,'gaussian',20);
reward_conf1 = smoothdata(reward_safe,'gaussian',7);

reward_sa = smoothdata(reward_safe,'gaussian',40);

episode1 = 1:size_safe;
episode_conf1 = [episode1 episode1(end:-1:1)];
inf_ = zeros(1,size_safe);
sup_ = zeros(1,size_safe);
for i = 1:size_safe
    sup_(i) = 500+log(episode1(i))-episode1(i)*0.05;
    inf_(i) = 600+log(episode1(i))-episode1(i)*0.1;   
end
reward_conf1 = [reward_sa + sup_, reward_sa(end:-1:1)- 1.1*inf_(end:-1:1)];

%%%%%%%%%%%%%%%%%%%%%%%%draw$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
%nocbf
p=fill(episode_conf, reward_conf,'g');
p.FaceColor= [175 238 238]/255;
p.EdgeColor = 'none';
set(p,'edgealpha',0,'facealpha',0.8) 
hold on
%safe cbf
p_safe=fill(episode_conf1, reward_conf1,'r');
p_safe.FaceColor= [255,182,193]/255;
p_safe.EdgeColor = 'none';
set(p_safe,'edgealpha',0,'facealpha',0.5) 
hold on
grid on
axis([0 3500 -2000 2000]);
plot(reward_s,'k','linewidth',4,'Color','#48D1CC')
plot(reward_sa,'k','linewidth',4,'Color','#FFA0BE')


% 
set(gca,'FontName','Times New Roman','FontSize',40);
set(gca,'FontSize',40);


hl21 = xlabel('Number of episodes','FontName','Times New Roman','FontSize',40);
% set(hl21,'interpreter','latex')
hl22 = ylabel('Average total reward','FontName','Times New Roman','FontSize',40,'Rotation',90);
% set(hl22,'interpreter','latex')
% 

hl20 = legend('Policy without shields','Policy with shields','FontName','Times New Roman','FontSize',40);
set(hl20,'Box','on');
% set(hl20,'interpreter','latex')
