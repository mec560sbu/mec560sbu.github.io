function write_M_fn(M , filename,name)


fid = fopen(filename,'w');

var_order = {'q_tr','q_t_r','q_s_r','q_f_r','q_t_l','q_s_l','q_f_l'} ;
dvar_order = {'dq_tr','dq_t_r','dq_s_r','dq_f_r','dq_t_l','dq_s_l','dq_f_l'} ;
var_types = {'I','m','l','lcm',};

head_str =  ['function ' name ' = ' filename(1:end-2) '(q,dq,params)'];
fprintf(fid,head_str);
fprintf(fid,'\n\n\n\n');

for i = 1:length(var_order)
    tag_str =   var_order{i}(2:end);
    for j = 1:length(var_types)
        var_str = [var_types{j} tag_str  ' = params.' var_types{j} tag_str ';'] ;
        fprintf(fid,var_str);
        fprintf(fid,'\n');
    end
end


var_types = {'R','d','h'};
for j = 1:length(var_types)
        var_str = [var_types{j} '_r'  ' = params.' var_types{j} '_r' '; \n'];
        fprintf(fid,var_str);
        var_str = [var_types{j} '_l'  ' = params.' var_types{j} '_l' '; \n'];
        fprintf(fid,var_str);
end        


var_str = ['lcm_f_r_x = params.lcm_f_r_x; \n'];
fprintf(fid,var_str);
var_str = ['lcm_f_r_y = params.lcm_f_r_y; \n'];
fprintf(fid,var_str);
var_str = ['lcm_f_l_x = params.lcm_f_r_x; \n'];
fprintf(fid,var_str);
var_str = ['lcm_f_l_y = params.lcm_f_l_y; \n'];
fprintf(fid,var_str);
var_str = ['g = params.g; \n'];
fprintf(fid,var_str);
fprintf(fid,'\n\n\n\n');

for i = 1:length(var_order)
    var_str =  [ var_order{i} ' = q(' num2str(i) '); \n'];
    dvar_str =  [ dvar_order{i} ' = dq(' num2str(i) '); \n'];
    fprintf(fid,var_str);
    fprintf(fid,dvar_str);
    fprintf(fid,'\n');
end
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