function w_not_norm = RotationAxis(R)
if size(R,1) ~= 3 || size(R,2) ~= 3
    error('Input should be a 3 x 3 matrix');
end
J = eye(3);
if isZero(R - J)
    w_not_norm = zeros(3,1);
    return;
end
w_not_norm = vpa(zeros(3,1));
for k = 1:3
    w_not_norm(k) = -trace(AxisToSkew(J(:,k))*R)/sqrt(trace(R)+1)/sqrt(3-trace(R));
end
