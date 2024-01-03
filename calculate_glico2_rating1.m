function[rating_new,RD_new,sig_new] = calculate_glico2_rating1(rating,RD,op_rating,op_RD,score,sig,tau)
% Glicko 2 rating update


% rescale

mu=(rating-1500)/173.7178;
phi=RD/173.7178;


op_mu=(op_rating-1500)./173.7178;
op_phi=op_RD/173.7178;

g_phi=1/sqrt(1+3*phi^2/pi^2);

g_phi_opp=1./sqrt(1+3*op_phi.^2/pi^2);


E=1./(1+exp(-g_phi_opp.*(mu-op_mu)));

v=1/sum((g_phi_opp.^2).*(E).*(1-E));


Del=v*sum(g_phi_opp.*(score-E));




a = log(sig^2);
eps=0.000001;
A=a;



if Del^2 > (g_phi^2 + v)
    B = log(Del^2-g_phi^2-v);
else
    k = 1;
    while f(a-k*tau)<0
        k = k + 1;
    end
    B=a-k*tau;
end




fA = f(A);
fB = f(B);


while abs(B - A) > eps
    C = A + (A - B)*fA/(fB - fA);
    fC = f(C);
    if fC*fB <= 0
        A = B;
        fA = fB;
    else
        fA = fA/2;
    end
    B = C;
    fB = fC;
end



% Calculate sigma_0
sigma_0 = exp(A/2);


g_phi_new=sqrt(phi^2+sigma_0^2);




g_phi_new2=1/sqrt(1/g_phi_new^2+1/v);

mu_new=mu+g_phi_new2^2*sum(g_phi_opp.*(score-E));

rating_new=173.7178*mu_new+1500;
RD_new=173.7178*g_phi_new2;
sig_new=sigma_0;


% function f(x) 
function result = f(x)
    result = exp(x)*(Del^2-g_phi^2-v-exp(x))/2/(g_phi+v+exp(x))^2-(x-a)/tau^2;
end
end


