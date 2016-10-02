clc
close all
clear all

% Minimize the simulation time
time_min = @(x) x(1);
% The initial parameter guess; 1 second, with a hundred accelerations
% all initialized to 5
x0 = [1; ones(101, 1) * 5];
% No linear inequality or equality constraints
A = [];
b = [];
Aeq = [];
Beq = [];
% Lower bound the simulation time at zero seconds, and bound the
% accelerations between -10 and 30
lb = [0; ones(101, 1) * -10];
ub = [Inf; ones(101, 1) * 30];
% Options for fmincon
options = optimset('TolFun', 0.00000001, 'MaxIter', 100000, ...
                   'MaxFunEvals', 100000);
% Solve for the best simulation time + control input
optimal = fmincon(time_min, x0, A, b, Aeq, Beq, lb, ub, ...
              @double_integrator_constraints, options);
% The simulation time is the first element in the solution
sim_time = optimal(1);
% The time discretization
delta_time = sim_time / (length(optimal) - 1);
times = 0 : delta_time : sim_time - delta_time;
% The accelerations given by the optimizer
accs = optimal(2:end);
% The resulting velocities (integrate accelerations with respect to time)
vels = cumtrapz(times, accs);
% The resulting positions (integrate velocities with respect to time)
positions = cumtrapz(times, vels);
% Make the plots
figure();
plot(times, accs);
title('Control Input (Acceleration) vs Time');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
figure();
plot(times, vels);
title('Velocity vs Time');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
figure();
plot(times, positions);
title('Position vs Time');
xlabel('Time (s)');
ylabel('Position (m)');