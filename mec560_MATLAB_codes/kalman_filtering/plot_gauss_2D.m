function plot_gauss_2D(mu,Sigma,x1,x2,color)


[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F,color);
xlabel('Position'); ylabel('Velocity');
