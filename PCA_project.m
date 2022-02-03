clc
clear all
close all
%%% step 2 %%%%
%first chang image data to double 
pic_matrix=imread('pic1.jpg');
rgb = im2double(pic_matrix);

%%% step 3 %%%
% Convert 3-dimensional array to 2D, where each row is a pixel (RGB)
rgb_2D = reshape(rgb, [], 3);
% n is the number of pixels:
n = size(rgb_2D,1);
% Plot pixels in color space with limition of plot every 10th point 
figure(1)
hold on
for i=1:10:n
     colour = rgb_2D(i,:);
     colour = max(colour, [0 0 0]);
     colour = min(colour, [1 1 1]);
     plot3(rgb_2D(i, 1), rgb_2D(i, 2), rgb_2D(i, 3),'.', 'Color', colour);
end
xlabel('red'), ylabel('green'), zlabel('blue');
xlim([0 1]), ylim([0 1]), zlim([0 1]);
hold off
grid on
axis equal   
%plot rotation.
for az=-180:3:180
    view(az,30); 
    drawnow;
end

%%%% step4 %%%%
%finding mean
sum=zeros(1,3);
for i=1:n
    sum(1,1)=sum(1,1)+rgb_2D(i, 1);
    sum(1,2)=sum(1,2)+rgb_2D(i, 2); 
    sum(1,3)=sum(1,3)+rgb_2D(i, 3);
end
mean_of_data =(1/n)*(sum);
disp('mean of data:'),disp(mean_of_data);

%%% step5 %%%
%finding PCA matrix
B=rgb_2D-mean_of_data;
pca_matrix=(1/(n-1))*(B')*(B);
disp('the PCA matrix is:'),disp(pca_matrix);

%%% step6 %%%
% finding variance and correlation of data
var=pca_matrix(1,1)+pca_matrix(2,2)+pca_matrix(3,3);
disp('variance of data:'),disp(var);
correlation=corrcoef(rgb_2D);
disp('correlation of data:'),disp(correlation);

pca_matrix=cov(rgb_2D);
mean_of_data=mean(rgb_2D);

%%% step 7%%%
% Get eigenvalues and eigenvectors of pca_matrix.
% Produces P,D such that pca_matrix*P = P*D.
% So the eigenvectors are the columns of P.
[P,D] = eig(pca_matrix);
e1 = P(:,3);
disp('Eigenvector e1:'), disp(e1);
e2 = P(:,2);
disp('Eigenvector e2:'), disp(e2);
e3 = P(:,1);
disp('Eigenvector e3:'), disp(e3);
d1 = D(3,3);
disp('Eigenvalue d1:'), disp(d1);
d2 = D(2,2);
disp('Eigenvalue d2:'), disp(d2);
d3 = D(1,1);
disp('Eigenvalue d3:'), disp(d3);

A = [e1'; e2'; e3'];
Y = A*(rgb_2D - repmat(mean_of_data,n,1))';
[height,width,depth] = size(rgb);
Y1 = reshape(Y(1,:), height, width);
Y2 = reshape(Y(2,:), height, width);
Y3 = reshape(Y(3,:), height, width);

figure(2);
subplot(1,3,1), imshow(Y1,[]);
subplot(1,3,2), imshow(Y2,[]);
subplot(1,3,3), imshow(Y3,[]);

%%% step 8 %%%
% Reconstruct image using only Y1 and Y2. 
new_pic = ( A(1:2,:)' * Y(1:2,:) )' + repmat(mean_of_data,n,1); 

Ir(:,:,1) = reshape(new_pic(:,1), height, width);
Ir(:,:,2) = reshape(new_pic(:,2), height, width);
Ir(:,:,3) = reshape(new_pic(:,3), height, width);
figure(3)
subplot(1,2,1),imshow(Ir);
subplot(1,2,2),imshow(pic_matrix);

