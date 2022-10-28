clc;
close all;
clear all;

% RTRL

fprintf('Real Time Recurrent Learning (RTRL) Results: \n');

% Generation of Training Data
u = rand(10,1);
y_act = zeros(10,1);
y_act(1) = 1;
for k = 1:10
    y_act(k+1) = (y_act(k)/(1 + y_act(k)^2)) + u(k)^3;
end

no_of_eps = 5e4;
cost = zeros(no_of_eps,1);
eta = 0.01;

% RTRL Algo 
X = zeros(length(u),5);
X(:,1) = u;
X(1,5) = y_act(1);
W2 = rand(5,1);
W3 = rand(5,1);
W4 = rand(5,1);
W5 = rand(5,1);

dW2 = zeros(5,1);
dW3 = zeros(5,1);
dW4 = zeros(5,1);
dW5 = zeros(5,1);

P2_2 = zeros(length(u)+1,5);
P2_3 = zeros(length(u)+1,5);
P2_4 = zeros(length(u)+1,5);
P2_5 = zeros(length(u)+1,5);

P3_2 = zeros(length(u)+1,5);
P3_3 = zeros(length(u)+1,5);
P3_4 = zeros(length(u)+1,5);
P3_5 = zeros(length(u)+1,5);

P4_2 = zeros(length(u)+1,5);
P4_3 = zeros(length(u)+1,5);
P4_4 = zeros(length(u)+1,5);
P4_5 = zeros(length(u)+1,5);

P5_2 = zeros(length(u)+1,5);
P5_3 = zeros(length(u)+1,5);
P5_4 = zeros(length(u)+1,5);
P5_5 = zeros(length(u)+1,5);

for l = 1:no_of_eps
    for i = 1:length(u)
        X(i+1,2) = logsig(X(i,:)*W2);
        X(i+1,3) = logsig(X(i,:)*W3);
        X(i+1,4) = logsig(X(i,:)*W4);
        X(i+1,5) = X(i,:)*W5;
        
        for k = 1:5
            P5_5(i+1,k) = W5(2)*P2_5(i,k) + W5(3)*P3_5(i,k) + W5(4)*P4_5(i,k) + W5(5)*P5_5(i,k) + X(i,k);
            P5_4(i+1,k) = W5(2)*P2_4(i,k) + W5(3)*P3_4(i,k) + W5(4)*P4_4(i,k) + W5(5)*P5_4(i,k);
            P5_3(i+1,k) = W5(2)*P2_3(i,k) + W5(3)*P3_3(i,k) + W5(4)*P4_3(i,k) + W5(5)*P5_3(i,k);
            P5_2(i+1,k) = W5(2)*P2_2(i,k) + W5(3)*P3_2(i,k) + W5(4)*P4_2(i,k) + W5(5)*P5_2(i,k);
            
            P4_5(i+1,k) = X(i+1,4)*(1 - X(i+1,4))*(W4(2)*P2_5(i,k) + W4(3)*P3_5(i,k) + W4(4)*P4_5(i,k) + W4(5)*P5_5(i,k));
            P4_4(i+1,k) = X(i+1,4)*(1 - X(i+1,4))*(W4(2)*P2_4(i,k) + W4(3)*P3_4(i,k) + W4(4)*P4_4(i,k) + W4(5)*P5_4(i,k) + X(i,k));
            P4_3(i+1,k) = X(i+1,4)*(1 - X(i+1,4))*(W4(2)*P2_3(i,k) + W4(3)*P3_3(i,k) + W4(4)*P4_3(i,k) + W4(5)*P5_3(i,k));
            P4_2(i+1,k) = X(i+1,4)*(1 - X(i+1,4))*(W4(2)*P2_2(i,k) + W4(3)*P3_2(i,k) + W4(4)*P4_2(i,k) + W4(5)*P5_2(i,k));
            
            P3_5(i+1,k) = X(i+1,3)*(1 - X(i+1,3))*(W3(2)*P2_5(i,k) + W3(3)*P3_5(i,k) + W3(4)*P4_5(i,k) + W3(5)*P5_5(i,k));
            P3_4(i+1,k) = X(i+1,3)*(1 - X(i+1,3))*(W3(2)*P2_4(i,k) + W3(3)*P3_4(i,k) + W3(4)*P4_4(i,k) + W3(5)*P5_4(i,k));
            P3_3(i+1,k) = X(i+1,3)*(1 - X(i+1,3))*(W3(2)*P2_3(i,k) + W3(3)*P3_3(i,k) + W3(4)*P4_3(i,k) + W3(5)*P5_3(i,k) + X(i,k));
            P3_2(i+1,k) = X(i+1,3)*(1 - X(i+1,3))*(W3(2)*P2_2(i,k) + W3(3)*P3_2(i,k) + W3(4)*P4_2(i,k) + W3(5)*P5_2(i,k));
            
            P2_5(i+1,k) = X(i+1,2)*(1 - X(i+1,2))*(W2(2)*P2_5(i,k) + W2(3)*P3_5(i,k) + W2(4)*P4_5(i,k) + W2(5)*P5_5(i,k));
            P2_4(i+1,k) = X(i+1,2)*(1 - X(i+1,2))*(W2(2)*P2_4(i,k) + W2(3)*P3_4(i,k) + W2(4)*P4_4(i,k) + W2(5)*P5_4(i,k));
            P2_3(i+1,k) = X(i+1,2)*(1 - X(i+1,2))*(W2(2)*P2_3(i,k) + W2(3)*P3_3(i,k) + W2(4)*P4_3(i,k) + W2(5)*P5_3(i,k));
            P2_2(i+1,k) = X(i+1,2)*(1 - X(i+1,2))*(W2(2)*P2_2(i,k) + W2(3)*P3_2(i,k) + W2(4)*P4_2(i,k) + W2(5)*P5_2(i,k) + X(i,k));
            
        end
        
        dW5 = -((y_act(i+1) - X(i+1,5))*P5_5(i+1,:))';
        dW4 = -((y_act(i+1) - X(i+1,5))*P5_4(i+1,:))';
        dW3 = -((y_act(i+1) - X(i+1,5))*P5_3(i+1,:))';
        dW2 = -((y_act(i+1) - X(i+1,5))*P5_2(i+1,:))';
        
        W5 = W5 - eta*dW5;
        W4 = W4 - eta*dW4;
        W3 = W3 - eta*dW3;
        W2 = W2 - eta*dW2;
        
    end
    
    for i = 1:length(u)
        X(i+1,2) = logsig(X(i,:)*W2);
        X(i+1,3) = logsig(X(i,:)*W3);
        X(i+1,4) = logsig(X(i,:)*W4);
        X(i+1,5) = X(i,:)*W5;
        
    end
    
    cost(l) = (1/(length(u)+1))*sum(0.5*((y_act - X(:,5)).^2));
    if cost(l) == 0
        break;
    end
    
end

fprintf('\nTraining Set Error:');
disp(cost(l));
fprintf('Optimal value of W2:\n');
disp(W2);
fprintf('Optimal value of W3:\n');
disp(W3);
fprintf('Optimal value of W4:\n');
disp(W4);
fprintf('Optimal value of W5:\n');
disp(W5);

figure(1);
epoch_no = 1:no_of_eps;
err = cost';
semilogy(epoch_no,err,'color',[0 0.6 0.3],'linewidth',1.5);
xlabel('\bf Number of epochs');
ylabel('\bf Cost (MSE)');
title('\bf RTRL: Learning Curve for 3 Hidden Nodes');

figure(2);
K = 1:length(u)+1;
plot(K,y_act,'ro','MarkerFaceColor','r');
xlabel('\bf {\it k}');
hold on;
plot(K,X(:,5),'bo','MarkerFaceColor','b');
legend('Predicted Output','Target Output');
title('\bf RTRL: Training Set: Predicted v/s Target Output [for 3 Hidden Nodes]');

% BP

fprintf('\nBackpropagation Learning Results: \n');

% u (Random Input Vector) is the same as that used in RTRL  
y = zeros(length(u),1);
target = (0);
y(1) = 1;
for k = 1:length(u)
    y(k+1) = y(k)/(1 + y(k)^2) + u(k)^3;
    target(k) = y(k+1);
end
target = target';
y = y(1:length(u));

eta = 0.01;
cost = zeros(no_of_eps,1);

X_chk = [u,y,ones(length(u),1)];

n = 3;

% Random Initialisation
W1 = rand(n,3);
W2 = rand(n+1,1);

% Training of the Network on the Training Set
for i = 1:no_of_eps
    for j = 1:length(u)
        x = [u(j);y(j);1];
        z1 = W1*x;
        a1 = logsig(z1);
        a1 = [a1;1];
        a2 = W2'*a1;


        delta_W2 = eta*(target(j) - a2)*a1;
        W2 = W2 + delta_W2;

        D_2 = (a2 - target(j))*W2(1:end-1);
        delta_W1 = -1*eta*(D_2.*(1-a1(1:end-1)).*a1(1:end-1))*x';
        W1 = W1 + delta_W1;

    end
    Z1 = W1*X_chk';
    A1 = logsig(Z1);
    A1 = [A1;ones(1,length(u))];
    A2 = W2'*A1;
    A2 = A2';
    cost(i) = (1/(length(u)))*sum(0.5*((target - A2).^2));
    if cost(i) == 0
        break;
    end
end

fprintf('\nFor %d hidden nodes, \n',n);
fprintf('Training Set Error:');
disp(cost(i));
fprintf('Optimal Weight Matrix 1:\n');
disp(W1);
fprintf('Optimal Weight Matrix 2:\n');
disp(W2);

figure(3);
epoch_no = 1:no_of_eps;
err = cost';
semilogy(epoch_no,err,'m','linewidth',1.5);
xlabel('\bf Number of epochs');
ylabel('\bf Cost (MSE)');
heading = sprintf('BP: Learning Curve for %d Hidden Nodes',n);
title(heading);

figure(4);
K = 1:length(u);
plot(K,target,'ro','MarkerFaceColor','r');
xlabel('\bf {\it k}');
hold on;
plot(K,A2,'bo','MarkerFaceColor','b');
legend('Predicted Output','Target Output');
title('\bf BP: Training Set: Predicted v/s Target Output [for 3 Hidden Nodes]');

    




    
    
    
    
        
            
        
            
            
            
            
     
        
        
        
        
        
            
        



