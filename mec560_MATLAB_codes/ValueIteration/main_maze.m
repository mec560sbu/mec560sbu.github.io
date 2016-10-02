clc
close all
clear all


x = 0:.1:4;
y = 0:.1:4;

Act = zeros(length(x),length(y));
V = zeros(length(x),length(y));
V_new = zeros(length(x),length(y));

[xx,yy] = meshgrid(x,y);

figure;
plot(xx,yy,'k.')
hold on;
x_goal = 2;
y_goal = 2;


x_start = 0;
y_start = 0;

[i_x,i_y]= xy_to_indices(x_goal,y_goal);
ix_goal = i_x;
iy_goal = i_y;

i_do_nth = [i_x,i_y];

Act(i_x,i_y) = 100;
plot(x_goal,y_goal,'ro')

obs_locs = [1 2  1.3 3;
    .5 3  2.5 3.5;
    .5 .5  2.5 1;
    2.25 1  2.5 2.5;
    0.5 1.5  .8 2.5;
    ];


for i = 1:size(obs_locs,1)
    patch( [obs_locs(i,1) obs_locs(i,1)  obs_locs(i,3) obs_locs(i,3)  ], ...
        [obs_locs(i,2) obs_locs(i,4)  obs_locs(i,4) obs_locs(i,2)  ],'g' );
    
    [ix_obs_st,iy_obs_st]= xy_to_indices( obs_locs(i,1), obs_locs(i,2));
    [ix_obs_en,iy_obs_en]= xy_to_indices( obs_locs(i,3), obs_locs(i,4));
    
    Act(ix_obs_st:ix_obs_en,iy_obs_st:iy_obs_en) = -100;
end

value_iteration;