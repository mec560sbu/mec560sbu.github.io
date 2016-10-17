close all

filename = ['Final_V.gif'];
close all
figure;


Az = -180:6:180;

for i = 1 :length(Az)
    
    
    surf(yy,xx,V);xlabel('X');ylabel('Y');zlabel('Value')
    title(['Az =  ' num2str(Az(i)) ', El = 30'])
        axis([0 10 0 10 -200 2100])
    view(Az(i),30)
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i  == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.125);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.125);
    end
end