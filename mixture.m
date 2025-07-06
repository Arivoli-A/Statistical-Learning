classdef mixture
% Class definition which calculates Mixture of Gaussian
    properties
            
        x;FG;BG;C;mu_FG;mu_BG;cov_FG;cov_BG;pi_FG;pi_BG;I_ground_double;

        x_cheetah;x_grass;P_cheetah;P_grass;
        
        log_likelihood_cheetah;log_likelihood_grass;
       
    end

    methods
        function obj = mixture(x,FG,BG,C,I_ground_double)
            if nargin > 0

            obj.x=x;

            d = size(x,2);
   
            obj.FG=FG;
            obj.BG=BG;

            obj.C=C;
            
            obj.mu_FG = rand(d,1,C);
            obj.mu_BG = rand(d,1,C);

            obj.cov_FG = zeros(d,d,C);
            obj.cov_BG = zeros(d,d,C);
            
            % Initialization of covariance matrix is also important for
            % convergence
            for i = 1:C
                obj.cov_FG(:,:,i) = diag(2*(1+rand(d, 1))); 
                obj.cov_BG(:,:,i) = diag(4*(1+rand(d, 1)));
            end

            prob_init=rand(C, 1);
            obj.pi_FG=prob_init./sum(prob_init);

            
            prob_init=rand(C, 1);
            obj.pi_BG=prob_init./sum(prob_init);


            obj.I_ground_double=I_ground_double;
        
            [obj.x_cheetah,obj.x_grass,obj.P_cheetah,obj.P_grass] = obj.Prior_prob_class(FG,BG);
            
            % Run the EM to update the parameters 
            
            disp('Cheetah')
            [obj.mu_FG,obj.cov_FG,obj.pi_FG,obj.log_likelihood_cheetah] = obj.Gaussian_EM(obj.x_cheetah,obj.mu_FG,obj.cov_FG,obj.pi_FG);
            disp('Grass')
            [obj.mu_BG,obj.cov_BG,obj.pi_BG,obj.log_likelihood_grass] = obj.Gaussian_EM(obj.x_grass,obj.mu_BG,obj.cov_BG,obj.pi_BG);
            
            end
        end               
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function C = covariance_EM(obj,x,mu,P_z_x)
        % Calculates covariance matrix of given data    
        % x - oxf
        % mu - fx1
        % C - fxf 
        
        o = size(x,1);
        sigma = zeros(size(x,2),1);
        
        for i = 1:o
            sigma=sigma+P_z_x(i)*(x(i,:)'-mu).^2;
        end
        
        sigma=sigma./sum(P_z_x);
        C = diag(sigma);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function [P_x,P_z_x] = mixture_model(obj,x,mu,cov_mat,pi)
        % Returns posterior distribution of z given x
        
        % mu - fx1xC
        % cov - fxfxC
        % pi - Cx1
        % x - oxf
        % P_x - ox1
        % P_z_x - oxc

        c=size(pi,1);
        o=size(x,1);
        P_z_x = zeros(o,c);
        
        for i = 1:c
            P_z_x(:,i) = pi(i)*mvnpdf(x,mu(:,:,i)',cov_mat(:,:,i));
        end

        P_x = sum(P_z_x,2); 
        P_z_x = P_z_x./P_x;

        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
function [mu,cov_mat,pi,likelihood_new] = Gaussian_EM(obj,x,mu,cov_mat,pi)
                % Returns parameters mu, cov, pi for a class for mixture of
                % gaussian model by running Expectation Maximization algorithmn
                
                % x - oxf
        
              
                % mu - fx1xC
                % cov - fxfxC
                % pi - Cx1
                % x - oxf
                % P_x - ox1
                % P_z_x - oxc
        
                % dist - ox1
                
                c = size(mu,3);
                
                [P_x,P_z_x] = obj.mixture_model(x,mu,cov_mat,pi);
                likelihood_old = sum(log(P_x));
                likelihood_new = sum(log(P_x))+1e2;
                tol = 1e-2;
                
                count = 0;

                while likelihood_new-likelihood_old >=tol
                    
                    count = count +1;
                    likelihood_old=likelihood_new;
                    % Maximization step 
                    for i = 1:c
                        mu(:,:,i) = sum(x'.*P_z_x(:,i)',2)./sum(P_z_x(:,i));
                        cov_mat(:,:,i) = obj.covariance_EM(x,mu(:,:,i),P_z_x(:,i));
                        pi(i) = sum(P_z_x(:,i))./sum(P_z_x,"all");
                    end
        
                    % Expectation step
                    [P_x,P_z_x] = obj.mixture_model(x,mu,cov_mat,pi);
        
                    % Evaluation step 
                    likelihood_new = sum(log(P_x));
                end
                fprintf('Number of iterations : %d \n',count);
        end

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               

        function [A,P_cheetah_given_x,P_grass_given_x] = BDR (obj,P_X_cheetah,P_X_grass)
            
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
        function error = mask_error(obj,mask)
        % Function to return the classification error probability 
        
        % Probability of error of algorithmn
        
        error = sum(abs(mask-obj.I_ground_double),"all")/(size(obj.I_ground_double,1)*size(obj.I_ground_double,2));

        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
function [error] = prediction(obj,obj_sent,d)
        % Calculates the mask and posterior probability and gets the final results
        
        % Class - cheetah conditioned is called from 1st class
        [P_x_cheetah,~] =  obj.mixture_model(obj.x(:,1:d),obj.mu_FG(1:d,:,:),obj.cov_FG(1:d,1:d,:),obj.pi_FG);
        [P_x_grass,~] =  obj_sent.mixture_model(obj_sent.x(:,1:d),obj_sent.mu_BG(1:d,:,:),obj_sent.cov_BG(1:d,1:d,:),obj_sent.pi_BG);

        [A,~]=obj.BDR(P_x_cheetah,P_x_grass);
        mask = reshape(A,size(obj.I_ground_double,2),[])';
        error = obj.mask_error(mask);
        end
    end
end