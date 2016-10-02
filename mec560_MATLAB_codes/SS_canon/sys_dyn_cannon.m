function dX = sys_dyn_cannon(t,y,vx,vy)


g = 9.81;

dx  = vx;
dy = 0;
ddy = -g;

dX = [dx;dy;ddy];