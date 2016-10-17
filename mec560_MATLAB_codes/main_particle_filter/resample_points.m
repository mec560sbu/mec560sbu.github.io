function [X_particle2,Y_particle2] = resample_points(weight_P,X_particle,Y_particle,N)

sig2 = .0005;
weight_P = weight_P/sum(weight_P)*2*pi;
w_ang = weight_P;

w_max = max(weight_P)

N = length(X_particle);
beta = 0;
index = randi(N);
P_new = [];
for i_N = 1:(N)
    beta = 2*w_max+beta + rand;
    while w_ang(index) < beta
        beta = beta - w_ang(index);
        index = index + 1;
        if index ==N
            index = 1;
        end
    end
    P_new = [P_new;X_particle(index) Y_particle(index)];
    index;
end

X_particle2 = P_new(:,1) + sqrt(sig2)*randn(size(P_new(:,1) ));
Y_particle2 = P_new(:,2) + sqrt(sig2)*randn(size(P_new(:,1) ));
% 
% X_particle1 = 10*(round(100*rand(1000,1)))/100;
% Y_particle1  = 10*(round(100*rand(1000,1)))/100;
% 


