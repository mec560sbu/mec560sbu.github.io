function exp_w_hat = SkewExp(S,theta)
if isSkewSym3(S)
    w = SkewToAxis(S);
elseif isvector(S) && size(S,1) == 3
    w = S;
    S = AxisToSkew(S);
else
    error('1st input must be a 3 x 3 skew-symmetric matrix');
end
if ~isscalar(theta)
    error('2nd input must be a scalar');
end
if isZero(w) || theta == 0
    exp_w_hat = eye(3);
    return
end
w_norm = sqrt(w.'*w);
S = S/w_norm;
%Use Rodrigues's formula
exp_w_hat = eye(3) + sin(w_norm*theta)*S/w_norm + (1 - cos(w_norm*theta))*S*S/w_norm^2;
