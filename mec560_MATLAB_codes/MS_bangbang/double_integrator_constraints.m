function [ c, ceq ] = double_integrator_constraints( x )
    % No nonlinear inequality constraint needed
    c = [];
    % Discretize the times into a vector
    sim_time = x(1);
    delta_time = sim_time / (length(x) - 1);
    times = 0 : delta_time : sim_time - delta_time;
    % Get the accelerations out of the rest of the parameter vector
    accs = x(2:end);
    % Integrate up the velocities and final position
    vels = cumtrapz(times, accs);
    pos = trapz(times, vels);
    % All elements of this vector must be constrained to equal zero
    % This means that final pos must equal 1 (target) and the end velocity
    % must be equal to zero.
    ceq = [pos - 1; vels(end)];
end