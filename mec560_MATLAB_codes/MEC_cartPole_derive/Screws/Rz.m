function R = Rz(theta)
%Rz : rotation matrix representing rotation about z-axis by an angle THETA
%
%Eric Westervelt
%10/26/2004

R = Raxisangle([0 0 1],theta);