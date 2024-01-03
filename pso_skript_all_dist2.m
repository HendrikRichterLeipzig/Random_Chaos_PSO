function[]=pso_skript_all_dist2(dd)
%clc; clear all;

population = 5;
functions = [1 2 3 4 5 6 7 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41]; % Funktionsnummern wie in niching_func
min_func = 1;
max_func = 28;          % höchste Testfunktion Nummer (1-16) nicht Name
generations = 20;
c = [0.72, 1.49, 1.49];          % inertia_weight, cognitive_weight, social_weight für PSO
runs = 50;
no_func = max_func - min_func + 1;

all_diversities = 4.04*ones(no_func,runs,10,generations);
all_lyap_exponents = 4.04*ones(no_func,runs,10);

differences_file = strcat('200_differences_1000_03052023',dd,'.mat')
diversities_file = strcat('200_diversities_1000_03052023',dd,'.mat');
lyapunov_file = strcat('200_lyapunov_1000_03052023',dd,'.mat');

save(differences_file,'population','min_func','max_func','generations','c','runs')

for random_source = [0:11]
    disp(num2str(random_source))
    
    for func_index = [min_func:1:max_func]
        disp({'Funktion: ',num2str(functions(func_index))})
        curr_func = functions(func_index);
        max_pos=no_func;
        
        index_for_lists = find(functions==curr_func);
        pos_all_diversities = index_for_lists-min_func+1;
        
        dimensions = get_dimension(curr_func);       %dimension of testfunction
        best_x = ones(dimensions,runs);
        gens_best = zeros(dimensions+1,generations,runs);
        
        switch random_source
            case 0  % chaos_1 logistic
                [random_matrix, random_init] = get_random_beta(dimensions,population,generations,runs,0,0.5,0.5);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 1  % beta0505
                [random_matrix, random_init] = get_random_beta_vs_norm1(dimensions,population,generations,runs,0,0.5,0.5,0.5,0.1);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 2  % norm0501
                [random_matrix, random_init] = get_random_beta_vs_norm1(dimensions,population,generations,runs,1,0.5,0.5,0.5,0.1);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 3  % beta15
                [random_matrix, random_init] = get_random_beta_vs_norm1(dimensions,population,generations,runs,0,1,5,0.5,0.1);  
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 4  % beta1313
                [random_matrix, random_init] = get_random_beta_vs_norm1(dimensions,population,generations,runs,0,13,13,0.5,0.1);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 5  % chaos_4 tent
                [random_matrix, random_init] = get_random_beta(dimensions,population,generations,runs,4,1,5);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
            case 6  % chaos_5 Chebyshev
                [random_matrix, random_init] = get_random_beta(dimensions,population,generations,runs,5,5,1);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 7  % uniform01
                [random_matrix, random_init] = get_random_beta_vs_norm1(dimensions,population,generations,runs,2,0.5,0.5,0.5,0.1);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 8  % chaos_2 cubic
                [random_matrix, random_init] = get_random_beta(dimensions,population,generations,runs,2,0.5,0.5);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 9  % chaos_3 bellows
                [random_matrix, random_init] = get_random_beta(dimensions,population,generations,runs,3,0.5,0.5);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
            case 10  % chaos_6 Weierstrass
                [random_matrix, random_init] = get_random_beta(dimensions,population,generations,runs,6,0.5,0.5);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
             case 11  % beta11
                [random_matrix, random_init] = get_random_beta_vs_norm1(dimensions,population,generations,runs,0,1,1,0.5,0.1);
                 all_lyap_exponents(func_index,:,random_source+1) = get_lyap_from_matrix(random_matrix);
        end

        
        for run = [1:1:runs]

            r_init = random_init(:,:,:,run);
            r_generations = random_matrix(:,:,:,run);

            [best_x(:,run), gens_best(:,:,run), all_diversities(pos_all_diversities,run,random_source+1,:)] = pso(curr_func, population, generations, c, r_init, r_generations, run, random_source);    %eigentlicher PSO
        end

        %% Auswertung der Abstände
        switch curr_func
            case 1
                best_func_1 = best_x;
                max_peak_1 = get_peak(1);     % Peaks der Funktion
                
                if(random_source == 0)        % nur im ersten Durchlauf Variablen vorbereiten
                    diff_para_1 = 0.404*ones(12,runs);
                    diff_fit_1 = 0.404*ones(12,runs);
                end
                
                best_fit_1 = 200;   % Beste Fitness in der Testfunktion
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_1));     % Abst�nde des gefundenen Peaks zu allen m�glichen Peaks der Funktion
                    diff_fit_1(random_source+1,run) = abs(best_fit_1-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_1)]                                % Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_1(i,:)'-best_func_1(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_1(random_source+1,run) = min(list_of_distances);       %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_1','diff_fit_1','-append');
                
            case 2
                best_func_2 = best_x;
                max_peak_2 = get_peak(2);
                
                if(random_source == 0)
                    diff_para_2 = 0.404*ones(12,runs);
                    diff_fit_2 = 0.404*ones(12,runs);
                end
                best_fit_2 = 1;
                
                for run = [1:1:runs]
                    list_of_distances = 4.04*ones(1,length(max_peak_2));
                    diff_fit_2(random_source+1,run) = abs(best_fit_2-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_2)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_2(i,:)'-best_func_2(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_2(random_source+1,run) = min(list_of_distances);                 %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_2','diff_fit_2','-append');
                
            case 3
                best_func_3 = best_x;
                max_peak_3 = get_peak(3);
                
                if(random_source == 0)
                    diff_para_3 = 0.404*ones(12,runs);
                    diff_fit_3 = 0.404*ones(12,runs);
                end
                
                best_fit_3 = 1;
                
                for run = [1:1:runs]
                    list_of_distances = 4.04*ones(1,length(max_peak_3));
                    diff_fit_3(random_source+1,run) = abs(best_fit_3-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_3)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_3(i,:)'-best_func_3(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_3(random_source+1,run) = min(list_of_distances);                 %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_3','diff_fit_3','-append');
                
            case 4
                best_func_4 = best_x;
                max_peak_4 = get_peak(4);
                
                if(random_source == 0)
                    diff_para_4 = 0.404*ones(12,runs);
                    diff_fit_4 = 0.404*ones(12,runs);
                end
                
                best_fit_4 = 200;
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_4));
                    diff_fit_4(random_source+1,run) = abs(best_fit_4-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_4)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_4(i,:)'-best_func_4(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_4(random_source+1,run) = min(list_of_distances);                 %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_4','diff_fit_4','-append');
                
            case 5
                best_func_5 = best_x;
                max_peak_5 = get_peak(5);
                
                if(random_source == 0)
                    diff_para_5 = 0.404*ones(12,runs);
                    diff_fit_5 = 0.404*ones(12,runs);
                end
                
                best_fit_5 =  1.0316;
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_5));
                    diff_fit_5(random_source+1,run) = abs(best_fit_5-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_5)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_5(i,:)'-best_func_5(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_5(random_source+1,run) = min(list_of_distances);                 %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_5','diff_fit_5','-append');
                
            case 6
                best_func_6 = best_x;
                max_peak_6 = get_peak(6);
                
                if(random_source == 0)
                    diff_para_6 = 0.404*ones(12,runs);
                    diff_fit_6 = 0.404*ones(12,runs);
                end
                
                best_fit_6 = 186.7309;
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_6));
                    diff_fit_6(random_source+1,run) = abs(best_fit_6-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_6)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_6(i,:)'-best_func_6(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_6(random_source+1,run) = min(list_of_distances);                %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_6','diff_fit_6','-append');
                
            case 7
                best_func_7 = best_x;
                max_peak_7 = get_peak(7);
                
                if(random_source == 0)
                    diff_para_7 = 0.404*ones(12,runs);
                    diff_fit_7 = 0.404*ones(12,runs);
                end
                
                best_fit_7 = 2;
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_7));
                    diff_fit_7(random_source+1,run) = abs(best_fit_7-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_7)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_7(i,:)'-best_func_7(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_7(random_source+1,run) = min(list_of_distances);                 %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_7','diff_fit_7','-append');
                    
                
            case 8
                best_func_8 = best_x;
                max_peak_8 = get_peak(1);
                
                if(random_source == 0)
                    diff_para_8 = 0.404*ones(12,runs);
                    diff_fit_8 = 0.404*ones(12,runs);
                end
                
                best_fit_8 = niching_func(max_peak_8(1,:),8);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_8));
                    diff_fit_8(random_source+1,run) = abs(best_fit_8-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_8)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_8(i,:)'-best_func_8(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_8(random_source+1,run) = min(list_of_distances);                 %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_8','diff_fit_8','-append');
                
                
            case 9
                best_func_9 = best_x;
                max_peak_9 = get_peak(1);
                
                if(random_source == 0)
                    diff_para_9 = 0.404*ones(12,runs);
                    diff_fit_9 = 0.404*ones(12,runs);
                end
                
                best_fit_9 = niching_func(max_peak_9(1,:),9);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_9));
                    diff_fit_9(random_source+1,run) = abs(best_fit_9-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_9)]                                         %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_9(i,:)'-best_func_9(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_9(random_source+1,run) = min(list_of_distances);                 %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_9','diff_fit_9','-append');
                    
                
            case 10
                best_func_10 = best_x;
                max_peak_10 = get_peak(10);
                
                if(random_source == 0)
                    diff_para_10 = 0.404*ones(12,runs);
                    diff_fit_10 = 0.404*ones(12,runs);
                end
                
                best_fit_10 = niching_func(max_peak_10(1,:),10);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length(max_peak_10));
                    diff_fit_10(random_source+1,run) = abs(best_fit_10-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length(max_peak_10)]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_10(i,:)'-best_func_10(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_10(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_10','diff_fit_10','-append');
                    
               
            case 21
                best_func_21 = best_x;
                max_peak_21 = get_peak(21);
                [length_21, width_21] = size(max_peak_21);
                
                if(random_source == 0)
                    diff_para_21 = 0.404*ones(12,runs);
                    diff_fit_21 = 0.404*ones(12,runs);
                end
                
                best_fit_21 = niching_func(max_peak_21(1,:),21);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_21);
                    diff_fit_21(random_source+1,run) = abs(best_fit_21-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_21]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_21(i,:)'-best_func_21(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_21(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_21','diff_fit_21','-append');
               
            case 22
                best_func_22 = best_x;
                max_peak_22 = get_peak(22);
                [length_22, width_22] = size(max_peak_22);
                
                if(random_source == 0)
                    diff_para_22 = 0.404*ones(12,runs);
                    diff_fit_22 = 0.404*ones(12,runs);
                end
                
                best_fit_22 = niching_func(max_peak_22(1,:),22);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_22);
                    diff_fit_22(random_source+1,run) = abs(best_fit_22-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_22]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_22(i,:)'-best_func_22(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_22(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_22','diff_fit_22','-append');
                    
            case 23
                best_func_23 = best_x;
                max_peak_23 = get_peak(23);
                [length_23, width_23] = size(max_peak_23);
                
                if(random_source == 0)
                    diff_para_23 = 0.404*ones(12,runs);
                    diff_fit_23 = 0.404*ones(12,runs);
                end
                
                best_fit_23 = niching_func(max_peak_23(1,:),23);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_23);
                    diff_fit_23(random_source+1,run) = abs(best_fit_23-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_23]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_23(i,:)'-best_func_23(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_23(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_23','diff_fit_23','-append');
                
                case 24
                best_func_24 = best_x;
                max_peak_24 = get_peak(24);
                [length_24, width_24] = size(max_peak_24);
                
                if(random_source == 0)
                    diff_para_24 = 0.404*ones(12,runs);
                    diff_fit_24 = 0.404*ones(12,runs);
                end
                
                best_fit_24 = niching_func(max_peak_24(1,:),24);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_24);
                    diff_fit_24(random_source+1,run) = abs(best_fit_24-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_24]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_24(i,:)'-best_func_24(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_24(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_24','diff_fit_24','-append');
                
            case 25
                best_func_25 = best_x;
                max_peak_25 = get_peak(25);
                [length_25, width_25] = size(max_peak_25);
                
                if(random_source == 0)
                    diff_para_25 = 0.404*ones(12,runs);
                    diff_fit_25 = 0.404*ones(12,runs);
                end
                
                best_fit_25 = niching_func(max_peak_25(1,:),25);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_25);
                    diff_fit_25(random_source+1,run) = abs(best_fit_25-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_25]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_25(i,:)'-best_func_25(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_25(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_25','diff_fit_25','-append');
                
             case 26
                best_func_26 = best_x;
                max_peak_26 = get_peak(26);
                [length_26, width_26] = size(max_peak_26);
                
                if(random_source == 0)
                    diff_para_26 = 0.404*ones(12,runs);
                    diff_fit_26 = 0.404*ones(12,runs);
                end
                
                best_fit_26 = niching_func(max_peak_26(1,:),26);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_26);
                    diff_fit_26(random_source+1,run) = abs(best_fit_26-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_26]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_26(i,:)'-best_func_26(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_26(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_26','diff_fit_26','-append');
            
            case 27
                best_func_27 = best_x;
                max_peak_27 = get_peak(27);
                [length_27, width_27] = size(max_peak_27);
                
                if(random_source == 0)
                    diff_para_27 = 0.404*ones(10,runs);
                    diff_fit_27 = 0.404*ones(10,runs);
                end
                
                best_fit_27 = niching_func(max_peak_27(1,:),27);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_27);
                    diff_fit_27(random_source+1,run) = abs(best_fit_27-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_27]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_27(i,:)'-best_func_27(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_27(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_27','diff_fit_27','-append');
                
            case 28
                best_func_28 = best_x;
                max_peak_28 = get_peak(28);
                [length_28, width_28] = size(max_peak_28);
                
                if(random_source == 0)
                    diff_para_28 = 0.404*ones(12,runs);
                    diff_fit_28 = 0.404*ones(12,runs);
                end
                
                best_fit_28 = niching_func(max_peak_28(1,:),28);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_28);
                    diff_fit_28(random_source+1,run) = abs(best_fit_28-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_28]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_28(i,:)'-best_func_28(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_28(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_28','diff_fit_28','-append');
                
             case 29
                best_func_29 = best_x;
                max_peak_29 = get_peak(29);
                [length_29, width_29] = size(max_peak_29);
                
                if(random_source == 0)
                    diff_para_29 = 0.404*ones(12,runs);
                    diff_fit_29 = 0.404*ones(12,runs);
                end
                
                best_fit_29 = niching_func(max_peak_29(1,:),29);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_29);
                    diff_fit_29(random_source+1,run) = abs(best_fit_29-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_29]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_29(i,:)'-best_func_29(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_29(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_29','diff_fit_29','-append');

             
             case 30
                best_func_30 = best_x;
                max_peak_30 = get_peak(30);
                [length_30, width_30] = size(max_peak_30);
                
                if(random_source == 0)
                    diff_para_30 = 0.404*ones(12,runs);
                    diff_fit_30 = 0.404*ones(12,runs);
                end
                
                best_fit_30 = niching_func(max_peak_30(1,:),30);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_30);
                    diff_fit_30(random_source+1,run) = abs(best_fit_30-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_30]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_30(i,:)'-best_func_30(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_30(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_30','diff_fit_30','-append');
             
             case 31
                best_func_31 = best_x;
                max_peak_31 = get_peak(31);
                [length_31, width_31] = size(max_peak_31);
                
                if(random_source == 0)
                    diff_para_31 = 0.404*ones(12,runs);
                    diff_fit_31 = 0.404*ones(12,runs);
                end
                
                best_fit_31 = niching_func(max_peak_31(1,:),31);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_31);
                    diff_fit_31(random_source+1,run) = abs(best_fit_31-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_31]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_31(i,:)'-best_func_31(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_31(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_31','diff_fit_31','-append');
             
             case 32
                best_func_32 = best_x;
                max_peak_32 = get_peak(32);
                [length_32, width_32] = size(max_peak_32);
                
                if(random_source == 0)
                    diff_para_32 = 0.404*ones(12,runs);
                    diff_fit_32 = 0.404*ones(12,runs);
                end
                
                best_fit_32 = niching_func(max_peak_32(1,:),32);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_32);
                    diff_fit_32(random_source+1,run) = abs(best_fit_32-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_32]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_32(i,:)'-best_func_32(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_32(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_32','diff_fit_32','-append');
             
             case 33
                best_func_33 = best_x;
                max_peak_33 = get_peak(33);
                [length_33, width_33] = size(max_peak_33);
                
                if(random_source == 0)
                    diff_para_33 = 0.404*ones(12,runs);
                    diff_fit_33 = 0.404*ones(12,runs);
                end
                
                best_fit_33 = niching_func(max_peak_33(1,:),33);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_33);
                    diff_fit_33(random_source+1,run) = abs(best_fit_33-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_33]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_33(i,:)'-best_func_33(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_33(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_33','diff_fit_33','-append');
             
             case 34
                best_func_34 = best_x;
                max_peak_34 = get_peak(34);
                [length_34, width_34] = size(max_peak_34);
                
                if(random_source == 0)
                    diff_para_34 = 0.404*ones(12,runs);
                    diff_fit_34 = 0.404*ones(12,runs);
                end
                
                best_fit_34 = niching_func(max_peak_34(1,:),34);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_34);
                    diff_fit_34(random_source+1,run) = abs(best_fit_34-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_34]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_34(i,:)'-best_func_34(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_34(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_34','diff_fit_34','-append');
             
             case 35
                best_func_35 = best_x;
                max_peak_35 = get_peak(35);
                [length_35, width_35] = size(max_peak_35);
                
                if(random_source == 0)
                    diff_para_35 = 0.404*ones(12,runs);
                    diff_fit_35 = 0.404*ones(12,runs);
                end
                
                best_fit_35 = niching_func(max_peak_35(1,:),35);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_35);
                    diff_fit_35(random_source+1,run) = abs(best_fit_35-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_35]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_35(i,:)'-best_func_35(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_35(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_35','diff_fit_35','-append');
             
             case 36
                best_func_36 = best_x;
                max_peak_36 = get_peak(36);
                [length_36, width_36] = size(max_peak_36);
                
                if(random_source == 0)
                    diff_para_36 = 0.404*ones(12,runs);
                    diff_fit_36 = 0.404*ones(12,runs);
                end
                
                best_fit_36 = niching_func(max_peak_36(1,:),36);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_36);
                    diff_fit_36(random_source+1,run) = abs(best_fit_36-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_36]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_36(i,:)'-best_func_36(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_36(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_36','diff_fit_36','-append');
             
             case 37
                best_func_37 = best_x;
                max_peak_37 = get_peak(37);
                [length_37, width_37] = size(max_peak_37);
                
                if(random_source == 0)
                    diff_para_37 = 0.404*ones(12,runs);
                    diff_fit_37 = 0.404*ones(12,runs);
                end
                
                best_fit_37 = niching_func(max_peak_37(1,:),37);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_37);
                    diff_fit_37(random_source+1,run) = abs(best_fit_37-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_37]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_37(i,:)'-best_func_37(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_37(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_37','diff_fit_37','-append');
             
             case 38
                best_func_38 = best_x;
                max_peak_38 = get_peak(38);
                [length_38, width_38] = size(max_peak_38);
                
                if(random_source == 0)
                    diff_para_38 = 0.404*ones(12,runs);
                    diff_fit_38 = 0.404*ones(12,runs);
                end
                
                best_fit_38 = niching_func(max_peak_38(1,:),38);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_38);
                    diff_fit_38(random_source+1,run) = abs(best_fit_38-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_38]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_38(i,:)'-best_func_38(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_38(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_38','diff_fit_38','-append');
             
             case 39
                best_func_39 = best_x;
                max_peak_39 = get_peak(39);
                [length_39, width_39] = size(max_peak_39);
                
                if(random_source == 0)
                    diff_para_39 = 0.404*ones(12,runs);
                    diff_fit_39 = 0.404*ones(12,runs);
                end
                
                best_fit_39 = niching_func(max_peak_39(1,:),39);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_39);
                    diff_fit_39(random_source+1,run) = abs(best_fit_39-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_39]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_39(i,:)'-best_func_39(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_39(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_39','diff_fit_39','-append');
             
             case 40
                best_func_40 = best_x;
                max_peak_40 = get_peak(40);
                [length_40, width_40] = size(max_peak_40);
                
                if(random_source == 0)
                    diff_para_40 = 0.404*ones(12,runs);
                    diff_fit_40 = 0.404*ones(12,runs);
                end
                
                best_fit_40 = niching_func(max_peak_40(1,:),40);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_40);
                    diff_fit_40(random_source+1,run) = abs(best_fit_40-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_40]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_40(i,:)'-best_func_40(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_40(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_40','diff_fit_40','-append');
             
             case 41
                best_func_41 = best_x;
                max_peak_41 = get_peak(41);
                [length_41, width_41] = size(max_peak_41);
                
                if(random_source == 0)
                    diff_para_41 = 0.404*ones(12,runs);
                    diff_fit_41 = 0.404*ones(12,runs);
                end
                
                best_fit_41 = niching_func(max_peak_41(1,:),41);
                
                for run = [1:1:runs]
                    list_of_distances = ones(1,length_41);
                    diff_fit_41(random_source+1,run) = abs(best_fit_41-max(gens_best(end,:,run))); %Abstand zu besten Fitness
                    
                    for i = [1:1:length_41]                                          %Abstand zu den vorhandenen Maxima als Parameterabstand
                        vek_best2peak = max_peak_41(i,:)'-best_func_41(:,run);
                        list_of_distances(i) = norm(vek_best2peak,2);
                    end
                    diff_para_41(random_source+1,run) = min(list_of_distances);            %Abstand zum nächstgelegenen Maximum als Parameterabstand
                end
                save(differences_file,'diff_para_41','diff_fit_41','-append');




        end %switch
    end %for test_functions
    
    
    
end

%save(diversities_file,'all_diversities')
save(lyapunov_file,'all_lyap_exponents')

