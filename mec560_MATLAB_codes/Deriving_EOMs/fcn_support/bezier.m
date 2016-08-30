function fcn = bezier(coeff,s)
% Calculates the value of Bezier polynomials
% Inputs:
%  ~ coeff: an NxM matrix of coefficients.  Each row is one polynomial of
%    order M-1.
%  ~ s: a row vector of length p containing numbers between 0 and 1.
% Output:
%  ~ fcn: a Nxp matrix containing the evaluated polynomials
%
% Modified 10/13/2013

[N,M] = size(coeff);
p = size(s,2);

M=M-1; %Bezier polynomials have M terms for M-1 order

fcn = zeros(N,p);
for k = 0:1:M
    fcn = fcn + coeff(:,k+1)*singleterm_bezier(M,k,s);
end

function val = singleterm_bezier(m,k,s)

if (k == 0)
    val = nchoosek(m,k).*(1-s).^(m-k);
elseif (m == k)
    val = nchoosek(m,k)*s.^(k);
else
    val = nchoosek(m,k)*s.^(k).*(1-s).^(m-k);
end