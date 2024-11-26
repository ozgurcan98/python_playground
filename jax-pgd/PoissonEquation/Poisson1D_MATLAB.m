% Here we solve 1D Poisson's equation for comparison.
% -d^2u/dx^2 = f(x)

clear; close all; clc

% Problem parameters:
Lx = 1.0;

Timings = [];
for nx = [1E+1,1E+2,1E+3,1E+4,1E+5]

    N = nx+1;
    x = linspace(0,Lx,N)';
    [M,K,C] = mass_stiff(N,x);

    RHS = M*pi^2*sin(pi*x);
    RHS(1) = 0; RHS(end) = 0;

    LHS = K;
    LHS(1,:) = 0; LHS(1,1) = 1;
    LHS(end,:) = 0;LHS(end,end) = 1;

    tic
    u = LHS \ RHS;
    time = toc;
    Timings = [Timings time];

    Error = norm(sin(pi*x)-u)/norm(sin(pi*x))
    % plot(x,u);

    
end
%%

figure
hold on

plot([1E+1,1E+2,1E+3,1E+4,1E+5],Timings,'-o');
plot([1E+4,1E+5],[1E-3,1E-2],'--');
set(gca,'YScale','log','XScale','log')

%% Errors:
% MATLAB version:
Ne = [1E+1,1+2,1E+3,1E+4,1E+5];
Error = [0.0082,8.2243e-05,8.2246e-07,7.8849e-09,4.7900e-08];

% Python version:
Ne = [1E+1,1+2,1E+3,1E+4,1E+5];
Error = [0.008184,8.224E-5,8.2247E-7,8.216E-9,2.909070826462729e-10];