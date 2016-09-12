function dx = eqn_bouncing_ball(t,x,g,vx)

x1 = x(1);
x2 = x(2);
x3 = x(3);

dx1 = vx;
dx2 = x(3);
dx3 = -g;

dx = [dx1;dx2;dx3];
