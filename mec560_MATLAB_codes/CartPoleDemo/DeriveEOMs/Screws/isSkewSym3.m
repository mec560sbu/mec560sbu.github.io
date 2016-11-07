function yes_no = isSkewSym3(S)
if size(S,1) ~= 3 || size(S,2) ~= 3
    yes_no = false;
    return;
end
if (isnumeric(S))
    if ~isZero(S + S.')
        yes_no = false;
        return;
    end
else
    if ~isZero(simple(S + S.'))
        yes_no = false;
        return;
    end
end;
yes_no = true;
