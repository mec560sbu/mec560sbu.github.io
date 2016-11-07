function [w_hat] = AxisToSkew(w)
if ~isvector(w) || length(w) ~= 3
    error('Input should be a 3 x 1 vector');
end
w_hat = [   0       -w(3)   w(2);
            w(3)    0       -w(1);
            -w(2)   w(1)    0];
        