classdef Bayesian_decision
% Class definition which calculates three Bayesian decision method namely Bayesian Parameter Estimation, Maximum A Posteriori Estimation and Maximum Likelihood Estimation   
    properties
        x;FG;BG;mu_0_FG;mu_0_BG;C0;I_ground_double;alpha;W0;

        x_cheetah;x_grass;P_cheetah;P_grass;
        
        P_x_cheetah_BPE;P_x_grass_BPE;P_x_cheetah_MAP;P_x_grass_MAP;P_x_cheetah_ML;P_x_grass_ML;

        P_cheetah_given_x_BPE;P_grass_given_x_BPE;mask_BPE;error_BPE;
        P_cheetah_given_x_MAP;P_grass_given_x_MAP;mask_MAP;error_MAP;
        P_cheetah_given_x_ML;P_grass_given_x_ML;mask_ML;error_ML;

    end

    methods
        function obj = Bayesian_decision(x,FG,BG,mu_0_FG,mu_0_BG,W0,alpha,I_ground_double)
            if nargin > 0
            obj.x=x;
            obj.FG=FG;
            obj.BG=BG;
            obj.mu_0_FG=mu_0_FG;
            obj.mu_0_BG=mu_0_BG;
            obj.I_ground_double=I_ground_double;
            obj.alpha=alpha;
            obj.W0=W0;
            obj.C0=obj.prob_dist_mean_prior(W0,alpha);

            [obj.x_cheetah,obj.x_grass,obj.P_cheetah,obj.P_grass] = obj.Prior_prob_class(FG,BG);

            obj.P_x_cheetah_BPE= obj.class_conditional_prob_bayesian_parameter_estimation(obj.x_cheetah,mu_0_FG);
            obj.P_x_grass_BPE = obj.class_conditional_prob_bayesian_parameter_estimation(obj.x_grass,mu_0_BG);
            
            obj.P_x_cheetah_MAP = obj.class_conditional_prob_MAP_estimation(obj.x_cheetah,mu_0_FG);
            obj.P_x_grass_MAP = obj.class_conditional_prob_MAP_estimation(obj.x_grass,mu_0_BG);
            
            obj.P_x_cheetah_ML = obj.class_conditional_prob_ML_estimation(obj.x_cheetah);
            obj.P_x_grass_ML = obj.class_conditional_prob_ML_estimation(obj.x_grass);

            [obj.P_cheetah_given_x_BPE,obj.P_grass_given_x_BPE,obj.mask_BPE,obj.error_BPE] = obj.prediction(obj.P_x_cheetah_BPE,obj.P_x_grass_BPE);
            [obj.P_cheetah_given_x_MAP,obj.P_grass_given_x_MAP,obj.mask_MAP,obj.error_MAP] = obj.prediction(obj.P_x_cheetah_MAP,obj.P_x_grass_MAP);
            [obj.P_cheetah_given_x_ML,obj.P_grass_given_x_ML,obj.mask_ML,obj.error_ML] = obj.prediction(obj.P_x_cheetah_ML,obj.P_x_grass_ML);
            end
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function G = Gaussian(obj,x,mu,C)
        % Calculates Gausssian pdf of given data
        % x - oxf
        % mu - fx1
        % C - fxf
        % G - ox1
        
        d = size(mu,1);
        n = size(x,1);
        A = zeros(n,1);
        
        if(size(C,1)==1 & size(C,2)==1)
            A = exp(-(x-mu).*(x-mu)/(2*C));
        else
            for o = 1:n
                A(o) = exp(-0.5.*((x(o,:)'-mu)'*(C\(x(o,:)'-mu)))); % Equivalent to C^-1b
            end
        end
        B = sqrt(det(C)*(2*pi)^d);
        G = A./B;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function C = covariance(obj,x)
        % Calculates covariance matrix of given data    
        % x - oxf
        % mu - fx1
        % C - fxf 
        
        o = size(x,1);
        mu = mean(x)';
        C = zeros(size(x,2),size(x,2));
        
        for i = 1:o
            C=C+(x(i,:)'-mu)*(x(i,:)'-mu)';
        end
        
        C=C./o;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function C0 = prob_dist_mean_prior(obj,w0,alpha)
        % Returns covariance matrix of prior distribution of mean
        % Mean mu_0 : to be given
        
        % w0 - fx1
        % alpha - 1x1
        % C0 - fxf
        
        C0 = diag(alpha.*w0);
        
        end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               
        function [mu_n,Cn] = prob_dist_mean_given_data(obj,x,C,C0,mu_0)
        % Returns mean and covariance matrix for probability distribution of mean
        % given data
        
        % n - 1x1
        % C - fxf
        % C0- fxf
        % mu_0 - fx1
        % mu_n - fx1
        % Cn - fx1
        
        n = size(x,1);
        Cx = C/n;
        sum_mat = Cx+C0;
        Cn = C0*(sum_mat\Cx);
        
        mu_n_est = mean(x)';
        mu_n = C0*(sum_mat\mu_n_est)+Cx*(sum_mat\mu_0);
        
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function dist = prob_dist_x_given_data(obj,x,mu_n,Cn,C)
        % Returns Gaussian probability distribution of x given data
        
        % x - oxf
        % mu_n - fx1
        % Cn - fxf
        % C - fxf
        % dist - ox1
        
        dist = obj.Gaussian(x,mu_n,Cn+C);
        
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                     
        function dist = prob_dist_x_given_mean(obj,x,mu,C)
        % Returns Gaussian probability distribution of x given mean and covariance
        % matrix
        
        % x - oxf
        % mu - fx1
        % C - fxf
        % dist - ox1
        
        dist = obj.Gaussian(x,mu,C);
        
        end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               
        % Calculating Bayes Decision Rule for classification based on prior probabilities and 
        function [P_cheetah_given_x,P_grass_given_x,A] = BDR (obj,P_X_cheetah,P_X_grass)
        % Calculates Bayes Decision Rule for classification based on prior
        % probabilities and class conditional probabilities to get posterior
        % probability and mask
        
        % P_x_cheetah - ox1
        % P_x_grass - ox1
        % P_cheetah - 1x1
        % P_grass - 1x1
        % P_cheetah_given_x_prop - ox1 
        % P_grass_given_x_prop - ox1 
        % A - ox1
        
        P_cheetah_given_x = P_X_cheetah.*obj.P_cheetah;% Value proportional to Probability that class is cheetah given feature x
        
        P_grass_given_x = P_X_grass.*obj.P_grass;          %  Value proportional to Probability that class is cheetah given feature x
        
        % P_grass_given_x = P_grass_given_x./sum(P_grass_given_x);  %
        % Normalization - Not required
        A = (P_cheetah_given_x>P_grass_given_x);  % Action - segmentation based on Bayes Decision Rule - choose class with more conditional probability given the feature to reduce risk. 
        end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function [x_cheetah,x_grass,P_cheetah,P_grass] = Prior_prob_class(obj,FG,BG)
        % Prior probability of classes 
        
        % FG - o1xf
        % BG - o2xf
        % x_cheetah - o1xf
        % x_grass - o2xf
        % P_cheetah - 1x1 
        % P_grass - 1x1
        
        x_cheetah = FG; %abs(FG);
        x_grass = BG; %abs(BG);
        P_cheetah = size(x_cheetah,1)/(size(x_cheetah,1)+size(x_grass,1));
        P_grass = 1 - P_cheetah;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function error = mask_error(obj,mask)
        % Function to return the classification error probability 
        
        % Probability of error of algorithmn
        
        error = sum(abs(mask-obj.I_ground_double),"all")/(size(obj.I_ground_double,1)*size(obj.I_ground_double,2));

        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        % Bayesian parameter estimation

        function P_x_class = class_conditional_prob_bayesian_parameter_estimation(obj,x_data,mu_0)
        % Calculates the class_conditional probability using bayesian parameter estimation method
        
        % x - oxf
        % x_data - o1xf
        % mu_0 - fx1
        % C0 - fxf
        % P_x_class - ox1
        
        C = obj.covariance(x_data);
        [mu_n,Cn] = obj.prob_dist_mean_given_data(x_data,C,obj.C0,mu_0);
        P_x_class = obj.prob_dist_x_given_data(obj.x,mu_n,Cn,C);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        % Maximum APosteriori estimation

        function P_x_class = class_conditional_prob_MAP_estimation(obj,x_data,mu_0)
        % Calculates the class_conditional probability using Maximum a Posteriori estimation method
        
        % x - oxf
        % x_data - o1xf
        % mu_0 - fx1
        % C0 - fxf
        % P_x_class - ox1
        
        C = obj.covariance(x_data);
        [mu_MAP,~] = obj.prob_dist_mean_given_data(x_data,C,obj.C0,mu_0);
        P_x_class = obj.prob_dist_x_given_data(obj.x,mu_MAP,0,C);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        % Maximum Likelihood estimation

        function P_x_class = class_conditional_prob_ML_estimation(obj,x_data)
        % Calculates the class_conditional probability using Maximum Likelihood estimation method
        
        % x - oxf
        % x_data - o1xf
        % P_x_class - ox1
        
        C = obj.covariance(x_data);
        mu_ML = mean(x_data)';
        P_x_class = obj.prob_dist_x_given_mean(obj.x,mu_ML,C); 
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function [P_cheetah_given_x,P_grass_given_x,mask,error] = prediction(obj,P_X_cheetah,P_X_grass)
        % Calculates the mask and posterior probability and gets the final results
        
        [P_cheetah_given_x,P_grass_given_x,A]=obj.BDR(P_X_cheetah,P_X_grass);
        mask = reshape(A,size(obj.I_ground_double,2),[])';
        error = obj.mask_error(mask);
        end
    end
end