load("./Robotarium-run26/reward.mat");

size = length(rewards);
reward = zeros(1,size);
for j = 1:size
    reward(j) = sum(rewards{j});
    
end

plot(reward)