

function[elo_new] = calculate_elo_rating1(elo_old,opp_rating,score)
%opp_ratings = [1200 1400 1350];
%tot_score   = 1.5


qa=power(10,elo_old/400);


[n,~]=size(opp_rating);


%K=.04;
K=4;
%K=10;

for i=1:n

qb=power(10,opp_rating(i,1)/400);

ea(i)=qa/(qa+qb);



end




elo_new=elo_old+K*(sum(score)-sum(ea));






