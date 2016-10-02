function [ c, ceq ] = double_integrator_constraints2( x )
    global gridN
    % No nonlinear inequality constraint needed
    c = [];
    % Calculate the timestep
    sim_time = x(1);
    delta_time = sim_time / gridN;
    % Get the states / inputs out of the vector
    positions = x(2             : 1 + gridN);
    vels      = x(2 + gridN     : 1 + gridN * 2);
    accs      = x(2 + gridN * 2 : end);
    
    % Constrain initial position and velocity to be zero
    ceq = [positions(1); vels(1)];
    for i = 1 : length(positions) - 1
        % The state at the beginning of the time interval
        x_i = [positions(i); vels(i)];
        % What the state should be at the start of the next time interval
        x_n = [positions(i+1); vels(i+1)];
        % The time derivative of the state at the beginning of the time
        % interval
        xdot_i = [vels(i); accs(i)];
        % The time derivative of the state at the end of the time interval
        xdot_n = [vels(i+1); accs(i+1)];
        
        % The end state of the time interval calculated using quadrature
        xend = x_i + delta_time * (xdot_i + xdot_n) / 2;
        % Constrain the end state of the current time interval to be
        % equal to the starting state of the next time interval
        ceq = [ceq ; x_n - xend];
    end
    % Constrain end position to 1 and end velocity to 0
    ceq = [ceq ; positions(end) - 1; vels(end)];
end