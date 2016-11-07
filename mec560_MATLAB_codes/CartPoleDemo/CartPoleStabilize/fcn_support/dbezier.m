function fcn = dbezier(coeff,s)
% Calculates the value of the first derivative of Bezier polynomials
% Inputs:
%  ~ coeff: an NxM matrix of coefficients.  Each row is one polynomial of
%    order M-1.
%  ~ s: a row vector of length p containing numbers between 0 and 1.
% Output:
%  ~ fcn: a Nxp matrix containing the evaluated polynomials
%
% Modified 10/13/2013

dcoeff = diff_coeff(coeff);
fcn = bezier(dcoeff,s);