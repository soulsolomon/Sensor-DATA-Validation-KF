clearvars
clear all
clc
close all

load('sensor_data_project_v2.mat')

[x,y]=grn2eqa(Position.latitude, Position.longitude); %covert geographical coordinates in (latitude, longitude) XY coordinates
subplot(2,2,1)
plot(x,y,'ro','Linewidth', 1);
legend('Position in x and y');
%system model 
x_m = 0:0.01:2.43;
a=2;
y_m = a.^x_m;
Y = [x_m;y_m];%Mean state vector containing position and velocity
Z = Y + 0.1*randn(size(Y));    %Actual Measurments

subplot(2,2,2)
plot(x_m, y_m,'b-')
subplot(2,2,3)
plot(x_m, Z)

%% 
% Defination of input to Kalman Filter Function

T = 1;                 
Z = Z;                     %measurements 
A = [1 0 T 0;
     0 1 0 T;
     0 0 1 0;
     0 0 0 1];             %state transition matrix F
C = [1 0 0 0;
     0 1 0 0];             %measurement matix
R  = eye(4);               %covariance matrix for the state noise
Q  = [0 0; 0 1];           %covariance matrix for the measurement noise
mu_0 = [1 1 1 1]';          %mean of inital belief
sigma_0 = eye(4);          %covariance of the initial belief


[mu,sigma] = kalman(Z,A,C,R,Q,mu_0,sigma_0);
subplot(2,2,4)
plot(mu(1,:),mu(2,:),'bo','Linewidth', 1);
 axis([-0.5 2.5 0.8 5])
hold on
xlabel('x position')
ylabel('y position')
legend('Estimated value')

%The error between the measured and actual position
% x and y position measurement error [m]
m_n = y(:,1)-x(:,1);
t = (0:0.01:1.211)';

% Kalman filter east position error [m]
figure;
% East Position Errors
subplot(2,1,1);
plot(t,m_n,'g');
ylabel('Position Error - x and y [m]');
xlabel('Time [s]');
legend(sprintf('Meas: %.3f',norm(m_n,1)/numel(m_n)));
axis tight;

t2 = (0:0.01:2.44);
subplot(2,1,2)
plot(t2(1,1:121),mu(3,1:121),'b')
ylabel('Estimated velocity');
xlabel('Time [s]');
hold on
plot(t2(1,1:121),diff(x), 'r')
legend(sprintf( 'estimated velocity','actual'));
axis tight;


%% 
% Kalman Filter

function [mu,sigma] = kalman(Z,A,C,R,Q,mu_0,sigma_0)
% [mu,sigma] = kalman(Z,A,C,R,Q,mu_0,sigma_0)
% Matlab function for Kalman filtering

[p,N] = size(Z);                % N = number of samples, p = number of "sensors"
n=length(A);                    % n = system order
mu_pred = zeros(n,N+1);         % Kalman predicted states
mu = zeros(n,N+1);              % Kalman filtered states

% Filter initialization:
mu(:,1) = mu_0;                 % Index 1 means time 0, no measurements
sigma = sigma_0;                % Initial covariance matrix (uncertainty)

% Kalman filter iterations:
for t=2:N
  % Prediction 
   mu_correct=mu_0;
   mu_pre=A*mu_correct;
   sigma_pre = A*sigma *A' + R;

   % Compute Kalman
   K = sigma_pre*C'*inv(C*sigma_pre*C'+ Q);

   % calculation based on observation:
   mu_correct = mu_pre + K*(Z(:,t-1)-C*mu_pre);
   sigma = sigma_pre - K*C*sigma_pre;
   mu(:,t)=mu_correct;
end
end