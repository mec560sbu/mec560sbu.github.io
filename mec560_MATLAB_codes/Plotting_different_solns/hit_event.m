function [value, isterminal, direction] = hit_event(t,x)

% value of variable to keep track of
% isterminal, 1 to stop, 0 otherwise. 
% direction, 
% 0- all zeros of the solution
% 1- only zeros where the event function is increasing
% -1 - only zeros where the event function is decreasing

value = x(2);
isterminal = 1; % 
direction = -1; % decreasing 
