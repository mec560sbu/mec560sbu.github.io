function yes_no = isHomogeneousTwist(zeta_hat)
if size(zeta_hat,1) ~= 4 || size(zeta_hat,2) ~= 4
    yes_no = false;
    return;
end
if ~isZero(zeta_hat(4,:))
    yes_no = false;
    return;
end
if ~isSkewSym3(zeta_hat(1:3,1:3))
    yes_no = false;
    return;
end
yes_no = true;
