function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = avePA1_K_M(Y, K, options, id_list)
% avePA1: Averge online passive-aggressive algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    the vector of lables
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j)
%  id_list:    a randomized ID list
%  options:    a struct containing rho, sigma, C, n_label and n_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm once
%    mistakes:  a vector of mistake rate 
% mistake_idx:  a vector of number, in which every number corresponds to a
%               mistake rate in the vector above
%         SVs:  a vector records the number of support vectors 
%     size_SV:  the size of final support set
%--------------------------------------------------------------------------

%% initialize parameters
C = options.C; % 1 by default
t_tick = options.t_tick;
alpha = [];
ave_alpha = [];
SV = [];
ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    if (isempty(alpha)), % init stage
        f_t = 0; 
    else
        k_t = K(id,SV(:))';
        f_t = alpha*k_t;            % decision function
    end
    l_t = max(0,1-Y(id)*f_t);   % hinge loss
    hat_y_t = sign(f_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    % count accumulative mistakes
    if (hat_y_t~=Y(id)),
        err_count = err_count + 1;
    end
    
    if (l_t>0)
        % update 
        s_t=K(id,id);
        gamma_t = min(C,l_t/s_t);
        alpha = [alpha Y(id)*gamma_t;];
        SV = [SV id];
        
        ave_alpha = [ave_alpha, 0];
        ave_alpha = ((t-1)/t)*ave_alpha + (1/t)*alpha;
    else
        ave_alpha = ((t-1)/t)*ave_alpha + (1/t)*alpha;
    end
    
    
    run_time=toc;
    if (mod(t,t_tick)==0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end
classifier.SV = SV;
classifier.alpha = ave_alpha;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
