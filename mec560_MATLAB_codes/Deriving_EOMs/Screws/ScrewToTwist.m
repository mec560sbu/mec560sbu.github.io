function twist = ScrewToTwist(h,q,w)
if ~isscalar(h)
    error('1st input must be scalar');
end
if ~isvector(q) || size(q,1) ~= 3
    error('2nd input must be a 3 x 1 vector');
end
if ~isvector(w) || size(w,1) ~= 3
    error('3rd input must be a 3 x 1 vector');
end
if isinf(h)
    v = w;
    w = [0;0;0];
else
    v = -AxisToSkew(w)*q + h*w;
end
twist = [v;w];
