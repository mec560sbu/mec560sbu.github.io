function write_M(M , filename,name)


fid = fopen(filename,'w');

[n_row,n_col] = size(M);
for i = 1:n_row
    for j = 1:n_col
        str_m = [name '(' num2str(i) ',' num2str(j) ') = ' ];
        str_m = [str_m char(M(i,j)) ];
        fprintf(fid,str_m);
        fprintf(fid,';\n');
    end
end

fclose(fid);