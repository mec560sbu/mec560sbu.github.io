function [i_x,i_y]= xy_to_indices(x,y)

i_x = round( x/.1+1);
i_y = round(y/.1+1);