% Time trajectory
function [timeInter, xInter, yInter] = pointToTrajectory(smoothPath)

% Define time vector, initial time & velocity (max)
time = zeros(length(smoothPath),1);
time(1) = 0;
v = 1.5;

% Determine time for each waypoint
for i = 2:length(smoothPath)
    dist = sqrt((smoothPath(i,1)-smoothPath(i-1,1))^2+(smoothPath(i,2)-smoothPath(i-1,2))^2);
    time(i) = time(i-1)+dist/v;
end

% Distribute points/time equally via spline interpolation
pp_X = spline(time,smoothPath(:,1));
pp_Y = spline(time,smoothPath(:,2));

timeInter = (0:1/1000:1)*time(end);
xInter = ppval(pp_X,timeInter);     % x interpolation
yInter = ppval(pp_Y,timeInter);     % y interpolation


% Plot results to verify accuracy
figure
plot(timeInter, xInter, timeInter, yInter)
grid on
grid minor
axis([0 10.5 0 10.5])
legend('x-position', 'y-position','Location','southeast')
xlabel('Time (s)')
ylabel('Postiion')
title('Position vs. Time')

