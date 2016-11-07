function dcoeff=diff_coeff(coeff)
% A function to automatically calculate the coefficient values of the first
% derivative of a Bezier polynomial
% Input:
%  ~ coeff: an NxM matrix of coefficients.  Each row is one polynomial of
%    order M-1.
% Output:
%  ~ dcoeff: an NxM-1 matrix containing the differintiated polynomial
%    coefficients
%
% Modified 10/13/2013

M = size(coeff,2)-1;
A = zeros(M,M+1);

for i=0:M-1
    A(i+1,i+1) = -(M-i)*nchoosek(M,i)/nchoosek(M-1,i);
    A(i+1,i+2) = (i+1)*nchoosek(M,i+1)/nchoosek(M-1,i);
end

A(M,M+1)=M*nchoosek(M,M);
dcoeff = coeff*A';