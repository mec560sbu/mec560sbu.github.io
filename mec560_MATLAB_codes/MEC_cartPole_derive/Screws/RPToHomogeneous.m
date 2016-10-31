function g = RPToHomogeneous(R,p)
if size(R,1) ~= 3 || size(R,2) ~= 3
    error('1st input should be a 3 x 3 matrix');
end
if ~isvector(p) || size(p,1) ~= 3
    error('2nd input should be a 3 x 1 vector');
end
g = [R, p; zeros(1,3),1];
