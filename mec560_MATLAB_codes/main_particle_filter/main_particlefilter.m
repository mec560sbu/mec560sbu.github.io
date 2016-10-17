clc
close all
clear all


x_robot = 2;
y_robot = 1.3;

x = 0:.1:10;
y = 0:.1:10;

figure;
N = 20000;
N_samples = N;
X_particle2 = 10*(round(100*rand(N,1)))/100;
Y_particle2  = 10*(round(100*rand(N,1)))/100;

plot_env;

range = 2.5;
th = pi/10;
sig2 = .4;
[X_particle,Y_particle] = plot_particles(obs_locs,X_particle2,Y_particle2, 2000);

mov = [0 .2;
    0 .2;
    0 .2;
    .2 0;
    .2 0;
    .1 .2;
    .1 .2;
    .1 .2;
    .1 .2;
    .1 0;
    .1 0;
    .2 .2;
    .2 0;
    .2 0;
    .2 0;
    .2 0;
    .2 0;
    0 -.2;
    .2 0;
    .2 0;
    .2 0;
    .2 0;
    .2 0;
    .2 0;
    .2 -.2;
    0 -.2;
    0 -.2;
    0 -.2;
    .2 0;
    .2 0;
    .2 0;
    .2 0;];

filename = 'ParticleFilter2.gif';

for i_update = 1:32
    
    % Plotting obstacles
    % Plotting obstacles
    plot(X_particle,Y_particle, 'r.')
    hold on;
    plot_env;
    plot(x_robot,y_robot,'ko','markerfacecolor','b');
    axis equal;
    
    axis([0 10 0 10]);
    [D_sense,P_sense] = sense_Radar(x_robot,y_robot,range,obs_locs,th,sig2);
    plot(P_sense(:,1),P_sense(:,2),'g*');
    for i =1:size(P_sense,1)
        line([x_robot P_sense(i,1)],[y_robot P_sense(i,2)]);
    end
    plot(X_particle,Y_particle, 'r.')
    
    
    if i_update>1
        x_robot = x_robot +mov(i_update-1,1);
        y_robot = y_robot  +mov(i_update-1,2);
        X_particle = X_particle +mov(i_update-1,1);
        Y_particle = Y_particle +mov(i_update-1,2);
        
    end
    
    % Sense function
    [D_sense,P_sense] = sense_Radar(x_robot,y_robot,range,obs_locs,th,sig2);
    
    
    for i_pt = 1:length(X_particle)
        [D_sense_p,P_sense_p] = sense_Radar(X_particle(i_pt),Y_particle(i_pt),range,obs_locs,th,sig2);
        
        Gauss_weight = exp( -((D_sense_p-D_sense).^2)/(2*sig2^2));
        weight_P(i_pt) = 10^10* prod(Gauss_weight);
        i_pt ;
    end
    
    [X_particle2,Y_particle2] = resample_points(weight_P,X_particle,Y_particle,N_samples);
    [X_particle,Y_particle] = plot_particles(obs_locs,X_particle2,Y_particle2, N_samples);
    
    
    hold off
    pause(0.1)
    
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i_update  == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.25);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.25);
    end
    
end