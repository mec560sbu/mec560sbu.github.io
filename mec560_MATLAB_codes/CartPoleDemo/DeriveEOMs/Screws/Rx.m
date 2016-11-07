function R = Rx(theta)
%Rx : rotation matrix representing rotation about x-axis by an angle THETA
%
%Eric Westervelt
%10/26/2004

R = Raxisangle([1 0 0],theta);