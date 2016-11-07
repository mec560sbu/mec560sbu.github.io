function write_file(p,filename,vars)

matlabFunction(p,...
    'file',filename,...
    'vars',vars);

movefile(filename, ['fcn_models/' filename]);