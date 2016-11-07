function fcn = d2bezier(coeff,s)
% Calculates the value of the second derivative of Bezier polynomials
% Inputs:
%  ~ coeff: an NxM matrix of coefficients.  Each row is one polynomial of
%    order M-1.
%  ~ s: a row vector of length p containing numbers between 0 and 1.
% Output:
%  ~ fcn: a Nxp matrix containing the evaluated polynomials
%
% Modified 10/13/2013

dcoeff = diff_coeff(coeff);
d2coeff = diff_coeff(dcoeff);

fcn = bezier(d2coeff,s);