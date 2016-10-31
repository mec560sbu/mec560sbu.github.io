function zeta = HomogeneousToTwist(zeta_hat)
if ~isHomogeneousTwist(zeta_hat)
    error('Input must be a valid homogeneous twist');
end
zeta = [zeta_hat(1:3,4); SkewToAxis(zeta_hat(1:3,1:3))];
