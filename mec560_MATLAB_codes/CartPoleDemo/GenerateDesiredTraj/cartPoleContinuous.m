function phaseout = cartPoleContinuous(input);


x_all   = input.phase.state(:,1);
th_all  = input.phase.state(:,2);
dx_all  = input.phase.state(:,3);
dth_all = input.phase.state(:,4);
u_all   = input.phase.control(:,1);

dq_all = get_dynamics(x_all,th_all,dx_all,dth_all,u_all);


phaseout.dynamics  = dq_all;
phaseout.integrand = u_all.^2;