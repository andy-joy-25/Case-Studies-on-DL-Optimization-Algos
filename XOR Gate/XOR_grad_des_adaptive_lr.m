clc;
clear all;
close all;

X = [0 0;0 1;1 0;1 1];
% Random Initialisation
W1 = [1 -2 4;-5 2 3];
W2 = [3;-2;5];
Y = [0;1;1;0];
eta2 = [0.4;0.4;0.4];
eta1 = [0.4 0.4 0.4;0.4 0.4 0.4];
mu = 1.00001;
d = 0.59;
no_of_eps = 5e6;

cost = zeros(no_of_eps,1);
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
        
        delta_W2_prev = delta_W2;
        delta_W2 = (a2 - Y(j,:))*(1 - a2)*a2*a1;
        if i~=1
            for k = 1:size(W2,1)
                if delta_W2_prev(k)*delta_W2(k) >= 0
                        eta2(k) = mu*eta2(k);
                else
                        eta2(k) = d*eta2(k);
                end
                W2(k) = W2(k) - eta2(k)*delta_W2(k);
            end
        else
            W2 = W2 - eta2(1)*delta_W2;
        end
        

        delta_W1_prev = delta_W1;
        D_2 = (a2 - Y(j,:))*(1 - a2)*a2*W2(1:end-1);
        delta_W1 = (D_2.*(1-a1(1:end-1)).*a1(1:end-1))*x';
        if i ~= 1
            for l = 1:size(W1,1)
                for m = 1:size(W1,2)
                    if delta_W1_prev(l,m)*delta_W1(l,m) >= 0
                        eta1(l,m) = mu*eta1(l,m);
                    else
                        eta1(l,m) = d*eta1(l,m);
                    end
                    W1(l,m) = W1(l,m) - eta1(l,m)*delta_W1(l,m);
                end
            end
        else
            W1 = W1 - eta1(1,1)*delta_W1;
        end

    end
    Z1 = W1*X_chk';
    A1 = 1./(1 + exp(-Z1));
    A1 = [A1;ones(1,4)];
    Z2 = W2'*A1;
    A2 = 1./(1 + exp(-Z2));
    A2 = A2';
    cost(i) = 0.25*sum(0.5*((Y - A2).^2));
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

epoch_no = 1:no_of_eps;
err = cost';
semilogy(epoch_no,err,'linewidth',2);
xlabel('\bf Number of epochs');
ylabel('\bf Cost (MSE)');
title('\bf Learning Curve');


