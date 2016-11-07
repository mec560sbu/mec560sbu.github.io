function exp_zeta_hat = TwistExp(zeta,theta)
if isHomogeneousTwist(zeta)
    zeta = HomogeneousToTwist(zeta);
elseif ~isvector(zeta) || size(zeta,1) ~= 6
    error('1st input should be a 6 x 1 vector');
end
if ~isscalar(theta)
    error('2nd input should be a scalar');
end
v = zeta(1:3);
w = zeta(4:6);
if isZero(w)    %Pure translation
    R = eye(3);
    p = theta*v;
else
    R = SkewExp(AxisToSkew(w), theta);
    p = (eye(3) - R)*(AxisToSkew(w)*v) + w*(w.')*v*theta;
end
exp_zeta_hat = RPToHomogeneous(R,p);
