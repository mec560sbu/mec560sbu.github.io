function R = Ry(theta)
%Ry : rotation matrix representing rotation about y-axis by an angle THETA
%
%Eric Westervelt
%10/26/2004

R = Raxisangle([0 1 0],theta);
