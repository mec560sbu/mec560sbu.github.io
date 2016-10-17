% plot_env

axis([0 10 0 10])




% 
obs_locs = [
    .5 3  2.5 3.5;
    .5 .5  4 1;
     3 3  5.5 3.5;
    4.5 .5  8.5 1;
    0 6.5  7 7;
    4 7.5  10 8;
    8 5.5  10 6;
    2 5.5  7 6;
   1 4.5  10 5;
    4.5 1  5 2.5;
    9.5 1  10 2.5;
    5 6 5.5 6.5
    1.5 8.5  8.5 9;
    2 7  2.5 8.5;

    ];


for i = 1:size(obs_locs,1)
    patch( [obs_locs(i,1) obs_locs(i,1)  obs_locs(i,3) obs_locs(i,3)  ], ...
        [obs_locs(i,2) obs_locs(i,4)  obs_locs(i,4) obs_locs(i,2)  ],'k' );
    
    [ix_obs_st,iy_obs_st]= xy_to_indices( obs_locs(i,1), obs_locs(i,2));
    [ix_obs_en,iy_obs_en]= xy_to_indices( obs_locs(i,3), obs_locs(i,4));
    
end

