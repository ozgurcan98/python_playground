function [M,K,C] = mass_stiff(N,x)

% Global mass matrix:
M = sparse(N,N);
% Global stiffness matrix:
K = sparse(N,N);
% Global gradient matrix:
C = sparse(N,N);

ngp = 2; % Number of gauss points
[xi_gp,w] = lgwt(ngp,-1,1); % Gauss points and weights

% Create system matrices for x-direction:
for e = 1 : N-1 % Turns in elements - x direction
    Je = (x(e+1)-x(e)) / 2;

    % Elemental mass matrix:
    Me = zeros(2);
    % Elemental stiffness matrix:
    Ke = zeros(2);
    % Elemental convection matrix:
    Ce = zeros(2);

    for i = 1 : ngp % Turns in Gauss points
        xi = xi_gp(i);

        S1 = 1/2 * (1 - xi);
        S2 = 1/2 * (1 + xi);

        dS1 = -1/2;
        dS2 = 1/2;

        Me(1,1) = Me(1,1) + S1 * S1 * Je * w(i);
        Me(1,2) = Me(1,2) + S1 * S2 * Je * w(i);
        Me(2,1) = Me(2,1) + S2 * S1 * Je * w(i);
        Me(2,2) = Me(2,2) + S2 * S2 * Je * w(i);

        Ke(1,1) = Ke(1,1) + dS1 / Je * dS1 / Je * Je * w(i);
        Ke(1,2) = Ke(1,2) + dS1 / Je * dS2 / Je * Je * w(i);
        Ke(2,1) = Ke(2,1) + dS2 / Je * dS1 / Je * Je * w(i);
        Ke(2,2) = Ke(2,2) + dS2 / Je * dS2 / Je * Je * w(i);

        Ce(1,1) = Ce(1,1) + S1  * dS1 / Je * Je * w(i);
        Ce(1,2) = Ce(1,2) + S1  * dS2 / Je * Je * w(i);
        Ce(2,1) = Ce(2,1) + S2  * dS1 / Je * Je * w(i);
        Ce(2,2) = Ce(2,2) + S2  * dS2 / Je * Je * w(i);

    end

    M([e,e+1],[e,e+1]) = M([e,e+1],[e,e+1]) + Me;
    K([e,e+1],[e,e+1]) = K([e,e+1],[e,e+1]) + Ke;
    C([e,e+1],[e,e+1]) = C([e,e+1],[e,e+1]) + Ce;
end