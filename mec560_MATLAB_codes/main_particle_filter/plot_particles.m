function [X_particle,Y_particle] = plot_dist(obs_locs,X_particle,Y_particle, N)

N = length(X_particle);
ind_rm = [];

for i_pt = 1:N
    
    
    X_particle_L = X_particle(i_pt)-obs_locs(:,1);
    X_particle_R = obs_locs(:,3) - X_particle(i_pt) ;
    Y_particle_B = Y_particle(i_pt) -   obs_locs(:,2) ;
    Y_particle_T = obs_locs(:,4)  -Y_particle(i_pt) ;
    
    ind_in = find( (X_particle_L.*X_particle_R)>=0 & (Y_particle_B.*Y_particle_T)>=0);
    if length(ind_in)~=0
        ind_rm = [ind_rm i_pt];
    end
    
end

X_particle(ind_rm) = [];
Y_particle(ind_rm) = [];
