function [zeta_hat] = TwistToHomogeneous(zeta)
if ~isvector(zeta) || size(zeta,1) ~= 6
    error('Input should be a 6 x 1 vector');
end
zeta_hat = [AxisToSkew(zeta(4:6)) zeta(1:3); zeros(1,4)];
