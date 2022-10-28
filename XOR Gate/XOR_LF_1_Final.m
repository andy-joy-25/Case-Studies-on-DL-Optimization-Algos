clc;
clear all;
close all;

X = [0 0;0 1;1 0;1 1];
% Random Initialisation
W1 = [1 -2 1.5;-2.4 2.5 1.7];
W2 = [1;-2;1.5];
Y = [0;1;1;0];
eta2 = zeros(1,4);
eta1 = zeros(1,4);
mu = 0.12;
eps = 1e-6;
no_of_eps = 5e6;

cost = zeros(no_of_eps,1);
lr1 = zeros(no_of_eps,4);
lr2 = zeros(no_of_eps,4);
tol = 1e-8;
X_chk = [0 0 1;0 1 1;1 0 1;1 1 1];
delta_W2 = [0;0;0];
delta_W1 = [0 0 0;0 0 0];

for i = 1:no_of_eps
    for j = 1:size(X,1)
        x = [X(j,:)';1];
        z1 = W1*x;
        a1 = 1./(1 + exp(-z1));
        a1 = [a1;1];
        z2 = W2'*a1;
        a2 = 1./(1 + exp(-z2));
        
        delta_W2 = (a2 - Y(j,:))*(1 - a2)*a2*a1;
        
        y_tilde2 = Y(j,:) - a2;
        J_p2 = (1 - a2)*a2*a1;
        eta2(j) = mu*(((norm(y_tilde2))^2)/(((norm(J_p2*y_tilde2))^2) + eps));

        
        D_2 = (a2 - Y(j,:))*(1 - a2)*a2*W2(1:end-1);
        delta_W1 = (D_2.*(1-a1(1:end-1)).*a1(1:end-1))*x';
        
        y_tilde1 = D_2;
        J_p1 = ((1-a1(1:end-1)).*a1(1:end-1))*x';
        eta1(j) = mu*(((norm(y_tilde1))^2)/(((norm(J_p1.*y_tilde1))^2)+ eps));
        
        W2 = W2 - eta2(j)*delta_W2;
        W1 = W1 - eta1(j)*delta_W1;

    end
    Z1 = W1*X_chk';
    A1 = 1./(1 + exp(-Z1));
    A1 = [A1;ones(1,4)];
    Z2 = W2'*A1;
    A2 = 1./(1 + exp(-Z2));
    A2 = A2';
    cost(i) = 0.25*sum(0.5*((Y - A2).^2));
    lr1(i,:) = eta1;
    lr2(i,:) = eta2;
    if cost(i) == 0
        break;
    end
end

fprintf('Error:');
disp(cost(i));
fprintf('Optimal Weight Matrix 1:\n');
disp(W1);
fprintf('Optimal Weight Matrix 2:\n');
disp(W2);
fprintf('Predicted Output Vector:\n');
disp(A2);

figure(1);
epoch_no = 1:no_of_eps;
err = cost';
semilogy(epoch_no,err,'linewidth',2);
xlabel('\bf Number of epochs');
ylabel('\bf Cost (MSE)');
title('\bf Learning Curve');

figure(2);
lr2_1 = lr2(:,1)';
semilogy(epoch_no,lr2_1,'linewidth',1.25);
xlabel('\bf Number of epochs');
ylabel('\bf Learning Rate');
title('\bf Learning Rate 2 (\eta_2) Variation (Adaptation)');
hold on;
lr2_2 = lr2(:,2)';
semilogy(epoch_no,lr2_2,'linewidth',1.25);
lr2_3 = lr2(:,3)';
semilogy(epoch_no,lr2_3,'linewidth',1.25);
lr2_4 = lr2(:,4)';
semilogy(epoch_no,lr2_4,'linewidth',1.25);
legend('(X_1,y_1)','(X_2,y_2)','(X_3,y_3)','(X_4,y_4)');

figure(3);
lr1_1 = lr1(:,1)';
semilogy(epoch_no,lr1_1,'linewidth',1.25);
xlabel('\bf Number of epochs');
ylabel('\bf Learning Rate');
title('\bf Learning Rate 1 (\eta_1) Variation (Adaptation)');
hold on;
lr1_2 = lr1(:,2)';
semilogy(epoch_no,lr1_2,'linewidth',1.25);
lr1_3 = lr1(:,3)';
semilogy(epoch_no,lr1_3,'linewidth',1.25);
lr1_4 = lr1(:,4)';
semilogy(epoch_no,lr1_4,'linewidth',1.25);
legend('(X_1,y_1)','(X_2,y_2)','(X_3,y_3)','(X_4,y_4)');





