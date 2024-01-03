function [eee,elo,rrr,rat] = calculate_elo_and_glicko_sequent1(dp)
% calculates statistical results




%[dp,df] = provide_distance_fitness_error;


[~,nn,~]=size(dp);

%sequencing

seq=25;



slices=nn/seq;





for j=1:slices

    dpara=dp(:,1+(j-1)*seq:j*seq,:);






elo=2000*ones(12,1);
thres=1e-4;


rat=1500*ones(12,1);
RD=350*ones(12,1);
sig=0.06*ones(12,1);


tau=0.8;


[n1,n2,n3]=size(dpara);
counter=[1:1:n1];



for ii=1:n2

   for i=1:n1

      
    

       
           x=squeeze(dpara(i,ii,:));
           
           y=squeeze(dpara(counter(counter~=i),ii,:));

         

 
                    for k=1:n1-1
          
                      score(k)=mean(binarize_vector_threshold(y(k,:)',x,thres));

                    end


          ee=elo(counter(counter~=i),ii);

        
                     rr=rat(counter(counter~=i),ii);
          dd=RD(counter(counter~=i),ii); 
        



           
        [rating_new,RD_new,sig_new]=calculate_glico2_rating1(rat(i,ii),RD(i,ii),rr,dd,score',sig(i,ii),tau);
         rat(i,ii+1)=rating_new;
      RD(i,ii+1)=RD_new;
      sig(i,ii+1)=sig_new;    

                

         elo(i,ii+1)=calculate_elo_rating1(elo(i,ii),ee,score);
         score=[];
             
  
 

    
      
   




end




end



eee(:,j) =elo(:,n2+1);

rrr(:,j)=rat(:,n2+1);


elo=[];
rat=[];
RD=[];
sig=[];


end



elo=2000*ones(12,1);
thres=1e-2;


rat=1500*ones(12,1);
RD=350*ones(12,1);
sig=0.06*ones(12,1);


tau=0.8;

dpara=dp;

[n1,n2,n3]=size(dpara);
counter=[1:1:n1];



for ii=1:n2

   for i=1:n1

      
    

       
           x=squeeze(dpara(i,ii,:));
           
           y=squeeze(dpara(counter(counter~=i),ii,:));

         

 
                    for k=1:n1-1
          
                      score(k)=mean(binarize_vector_threshold(y(k,:)',x,thres));

                    end


          ee=elo(counter(counter~=i),ii);

        
           
                rr=rat(counter(counter~=i),ii);
          dd=RD(counter(counter~=i),ii); 
        



           
        [rating_new,RD_new,sig_new]=calculate_glico2_rating1(rat(i,ii),RD(i,ii),rr,dd,score',sig(i,ii),tau);
         rat(i,ii+1)=rating_new;
      RD(i,ii+1)=RD_new;
      sig(i,ii+1)=sig_new;   
                

         elo(i,ii+1)=calculate_elo_rating1(elo(i,ii),ee,score);
         score=[];
             
  
 

    
      
   




end




end
