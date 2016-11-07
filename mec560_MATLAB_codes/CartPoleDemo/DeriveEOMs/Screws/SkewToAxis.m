function w = SkewToAxis(S)
if ~isSkewSym3(S)
    error('Input must be a 3 x 3 skew-symmetric matrix');
end
w = [-S(2,3); S(1,3); -S(1,2)];
