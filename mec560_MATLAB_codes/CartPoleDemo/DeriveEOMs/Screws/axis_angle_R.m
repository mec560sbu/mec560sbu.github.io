function [k,th,th_D] = axis_angle_R(R,r);

l = ([R(3,2)-R(2,3) ; R(1,3) - R(3,1) ; R(2,1) - R(1,2) ]);

th = sign(l'*r)*abs(acos((R(1,1)+R(2,2)+R(3,3)-1)/2));
k = 1/2/sin(th)*l;
th_D = th*180/pi;