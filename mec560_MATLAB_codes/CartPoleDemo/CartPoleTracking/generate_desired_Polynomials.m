load solution_p0001_approx.mat

time = solution.phase.time;
states = solution.phase.state;
controls = solution.phase.control;


pp_x = spline(time,states(:,1));
pp_th = spline(time,states(:,2));
pp_dx = spline(time,states(:,3));
pp_dth = spline(time,states(:,4));
pp_u = spline(time,controls);

pp_states.x = pp_x;
pp_states.th = pp_th;
pp_states.dx = pp_dx;
pp_states.dth = pp_dth;
pp_controls.u = pp_u;
