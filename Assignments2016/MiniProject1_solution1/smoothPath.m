% Path smoothing function
function newPath = smoothPath(origPath)

alpha_smooth = .1;
for i_alpha = 1
    
    newPath = origPath;
    newPath(1,:) = origPath(1,:);
    newPath(end,:)=origPath(end,:);
    
    newPath_old = newPath;
    error = 1;
    i_iter = 0;
    while error > 0.00000001
        i_iter = i_iter+1;
        for i = 2:length(origPath)-1
            newPath(i,:) = newPath(i,:)+0.5*(origPath(i,:)-newPath(i,:))+alpha_smooth*(newPath(i-1,:)-2*newPath(i,:)+newPath(i+1,:));
        end
        error(i_iter) = norm(newPath-newPath_old);
        newPath_old = newPath;
        
    end
    
    plot(newPath(:,1),newPath(:,2),'r', 'LineWidth',2) 
end