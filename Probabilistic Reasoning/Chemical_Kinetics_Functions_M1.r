
library(datasets) #
library(gdata)


# The ultimate goal of this function file consists of 
# computing the posterior probability distribution 
# as a function of the three priors and of the likelihood for
# model M1.
# A maximum likelihood estimation is performed at the same time.

 


generate_grid_M1 <- function(lb,ub,n_k)
# The grid is generated for the logarithmic scale 
# and then converted into the normal scale.
# lb: lower bound
#ub: upper bound.
{ print(c("Generate grid - n_k : ",n_k))
  log10_lb = log10(lb)
  log10_ub = log10(ub)
  step = (log10_ub-log10_lb)/(n_k-1)          
  log10_k_grid <- seq(log10_lb,log10_ub,step)

    k_grid <- 10^(log10_k_grid)
    
  return (k_grid)
} 

############################""

# Definition of the priors

f_k <- function(k,lbound,ubound) {k/k/(ubound-lbound)}  # "k/k" : dirty trick to get a sequence of constant numbers.
f_u <- function(k,lbound,ubound) {1/k^2/(1/lbound-1/ubound)}
f_w <- function(k,lbound,ubound) {k^(-1)/(log(10)*(log10(ubound)-log10(lbound))  ) }


############################""

produce_all_priors_M1 <- function(lbound,ubound,n_k)
# Produce 3 uniform priors with respect to k, 
# u = 1/k and w= log10(k), respectively. 
  
# Normally, the function should return a 
# numerical matrix with 3 columns corresponding to the
# 3 priors. 
{ 
  # We start by generating the grid vector 
  
  k <-  generate_grid_M1(lbound,ubound,n_k)
  f_k_k <-  f_k(k,lbound,ubound)     
  f_u_k <-  f_u(k,lbound,ubound)       
  f_w_k <-  f_w(k,lbound,ubound)         
    
# We combine the four vectors into a matrix
   
   priors <- cbind(k,f_k_k,f_u_k,f_w_k)
   colnames(priors) <- c("k","f_k_k","f_u_k","f_w_k")
   
   prior_k_frame = cbind(k,f_k_k)   
   prior_u_frame = cbind(k,f_u_k)   
   prior_w_frame = cbind(k,f_w_k)
   
   write.table(prior_k_frame, file="prior_k_M1.csv", row.names=FALSE, col.names=FALSE, append = FALSE)  
   write.table(prior_u_frame, file="prior_u_M1.csv", row.names=FALSE, col.names=FALSE, append = FALSE)  
   write.table(prior_w_frame, file="prior_w_M1.csv", row.names=FALSE, col.names=FALSE, append = FALSE)  
   # It's also good to have the values in 3 files.
   
   
 return (priors)
}


##########################################################


integrate_density_M1 <- function(k,density_k,lb = 0, ub=1E+99)
# Numerical integration of a probability density.
# We want to check it is equal to 1.
# Because of some bug, the built-in R function "integrate" 
# sometimes diverges so that we can't use it. 
# "lb" and "ub" are optional bounds. 
# If they aren't given to the function, all values of
# k will be used to compute the integral. 
  {
    
  n_k = length(k)

  int = 0
    for (i in 1:(n_k-1))
    { 
      if (   ( k[i] >= lb  )   &    ( k[i+1] <= ub  )           )
    {int=(k[i+1]-k[i])*(density_k[i]+density_k[i+1])/2+int}
    }
  return(int)
}

###################################


integrate_3densities_M1 <- function(densities,lb = 0, ub=1E+99)

# We consider here a family of 3 probability densities
# which could be 3 priors or 3 posteriors. 
  
# Numerical integration. 
# Goal1: checking that the whole integral is
#  equal to 1 in all 3 cases.
  
 # Goal2: compute the integral for narrower
  # intervals given by lb and ub.


{ # The argument 'densities' should be a numerical matrix with 4 columns, so that: 
  k = densities[,1];   f_k_k = densities[,2]
  f_u_k = densities[,3] ; f_w_k = densities[,4]   
 
   I_k <- integrate_density_M1(k,f_k_k,lb,ub)
   I_u <- integrate_density_M1(k,f_u_k,lb,ub)
   I_w <- integrate_density_M1(k,f_w_k,lb,ub)   
   
   int = c(I_k,I_u,I_w)
  
print(c("3 integrals : ",int))
   
  return (int)
}

###################################


test_produce_priors_M1 <- function(lb,ub,n_k)
# Plot the 3 priors. Returns the integrals of the 3 priors.
{
priors <-  produce_all_priors_M1(lb,ub,n_k)

int <- integrate_3densities_M1(priors) # We integrate first the three priors.
  
 # We must now convert the matrix into a data frame 
 # in order to be able to use 'ggplot'.
 
 priors <-  data.frame(priors)
 colnames(priors) <- c("k","f_k_k","f_u_k","f_w_k")
 
# print(priors)    DEBUGGING
 
 p = ggplot() + 
   geom_line(data = priors, aes(x = k, y = f_k_k, color = "black")) +
   geom_line(data = priors, aes(x = k, y = f_u_k, color = "blue")) +
    geom_line(data = priors, aes(x = k, y = f_w_k, color = "red")) +        
   xlab(expression(paste("k(.",s^{-1},")" ) )) + 
    ylab('Probability density')+scale_x_continuous(trans='log10',labels = scales::scientific)+scale_y_continuous(trans='log10',labels = scales::scientific)+
  scale_color_discrete(name = "Prior", labels = c(expression(f[k]),expression(f[u]),expression(f[w]))     )#+        coord_cartesian(xlim =c(1E+04,1E+10))

 #   trans='log10': log scale.
 #   labels = scales::scientific  :  scientific notation for the numbers
 #   scale_color_discrete :  legend 
 
 
 print(p) 

return (int)
  
}



################
# Definition of the profile of R according to M1
# t (us)   k (/s)    R0 (mol/m^3)

# t is a vector/sequence of time values.
# It can be defined in the following way: 
# step = (1000-0)/(n_t-1)
# t_model <- seq(0,1000,step)
# 

# It must be produced before calling this function.

Compute_R_profile_M1 <- function(t,R0,k){
   R = R0*exp(-k*t*1E-06)
  return (R)
  }
#################

plot_profile_M1 <- function(t_model,R_model)
{
# Plot the model profile along with the measurements.

 R_profile_exp <- read.table("R_Exp.csv", header=FALSE) # We first extract the measurements.
 colnames(R_profile_exp) <- c("t","R")       
 # "read.table" produces automatically data.frames
 
 
 
 
 R_profile_model = data.frame(t_model,R_model)  # We can only plot a data frame with ggplot.
 colnames(R_profile_model) <- c("t","R")      
  
 #print(  c("R_profile_model: ",R_profile_model ) )  #DEBUGGING
 
  p = ggplot() + 
    geom_point(data = R_profile_exp, aes(x = t, y = R, color = "blue")) +
    geom_line(data = R_profile_model, aes(x = t, y = R, color = "black")) +
    xlab(expression(paste("t(",mu,"s",")" ) )) + 
    ylab(expression( paste("[R] (mol.",cm^{-3},")" ) ))+
    scale_color_discrete(name = "Profile", labels = c("Optimal profile","Experiment")     )+
    scale_x_continuous(breaks = c(0, 600,1200))+
    scale_y_continuous(breaks = c(0, 300,600,900,1200))+   
    
    theme(legend.position = "none")
    
  print(p)   # Print the graphic
  
  write.csv(R_profile_model, file = "R_M1.csv")   # Save the model predictions.
}

##########################################

# We'll now compute the likelihood function for a given value of k. 

Compute_likelihood_M1 <- function(R_profile_exp,R0,k,epsilon){
      
 # R_profile_exp is a DATA FRAME containing the experimental profile.
  
  t_exp <- R_profile_exp$t   # Extraction from the data frame.
  R_exp <- R_profile_exp$R
  
sigma = epsilon*R_exp     # standard deviations proportional to the measurements. 
# epsilon: proportionality constant.


R <- Compute_R_profile_M1(t_exp,R0,k)   # Profile predicted by k at the 
                                        # experimental time points.

difference_2 = ((R_exp-R)/sigma)^2     # Vector: terms of the chi-square sum.  
chi_square = sum(difference_2)        



factor_local = log(2*pi*sigma^2)      # Write down the likelihood function   
factor= -0.5*sum(factor_local)        # to understand all these steps.  
  
 
l = factor-0.5*chi_square
#l 

L =exp(l)

#bd = cbind(t_exp,R,R_exp,sigma)  Useful for DEBUGGING


return (L)
} # Ende  Compute_likelihood



#####################################################################

Compute_likelihood_all_M1 <- function(R0,lbound,ubound,n_k,epsilon)

{ # Compute the likelihood function between the lower bound and
  # the upper bound. 
  # R0: initial concentration (mol/cm^3).
  
  # It is possible and recommanded to choose 
  # narrower bounds [lbound,ubound] in order to
  # focus on the zone where the likelihood
  # has its highest values for its graphical
  # representation. 
  
 # system("rm Likelihood_M1.csv")  # The file must be empty at the beginning
                                  # of the loop ("rm" = remove).
                                  # It'll be incrementally filled. 


	fn <- "Likelihood_M1.csv"
	#Check its existence
	if (file.exists(fn)) 
	  #Delete file if it exists
	  file.remove(fn)


  
  k_all =  generate_grid_M1(lbound,ubound,n_k)   # We generate the grid.

  # print("k_all in Compute Likelihood: ")   DEBUGGING
  # print(k_all)                             DEBUGGING
   
   
  R_profile_exp <- read.table("R_Exp.csv", header=FALSE) # We first extract the measurements.
  colnames(R_profile_exp) <- c("t","R")     
  t_exp = R_profile_exp[1]  
  R_exp = R_profile_exp[2]
  # Through "read.table", we automatically get a data frame that can 
  # be used for producing plots with ggplot. 
  

  n_k = length(k_all)[1]  # Number of grid points
# print(c("n_k = ",n_k))
  
  L_max = 0           # maximum likelihood estimator 
  k_max = k_all[1]    # maximum likelihood estimator 
  
  # Debugging mode
  # print("Calling  ' Compute_likelihood_all_M1'  ")
  # print(" k_all:   ")
  # print( k_all)
  # print("  R_profile_exp:  ")
  # print(R_profile_exp)  
  # print(c("R0: ", R0) )  
  # print(c("epsilon: ", epsilon) )  
  
  
   for(i in 1:n_k) { 
    
    L =  Compute_likelihood_M1(R_profile_exp,R0,k_all[i],epsilon) 
      d <- cbind(k_all[i],L)
#      print(c("i= ",i))
write.table(d, file="Likelihood_M1.csv", row.names=FALSE, col.names=FALSE, append = TRUE)  
     
      if (L > L_max )   {L_max = L; k_max = k_all[i]}
           # We update the MLE. 
                           }   # End for 
 
   #print(c(k_all[10],L_all)

  result = c(k_max,L_max)  # Only the MLE is returned by the function.
  return(result)          # the profile L(k) has been saved in a file. 
}

#######################################################

Examine_likelihood_M1 <- function(R0,lbound,ubound,n_k,epsilon=0.065)

# Plot the likelihood function and returns the
# maximum likelihood estimator. 
  
{

max_likelihood =  Compute_likelihood_all_M1(R0,lbound,ubound,n_k,epsilon)

print(c("Max lilelihood: ",max_likelihood))
  
L_all <- read.table("Likelihood_M1.csv", header=FALSE)
colnames(L_all) <- c("k_all","L_all")     
# We get a data frame we can directly use with ggplot.

p = ggplot() + 
  geom_line(data = L_all, aes(x = k_all, y = L_all, color = "black")) +
  xlab(expression( paste("k (",s^{-1}, ")" ) )) + 
  ylab("Likelihood")+
  #scale_color_discrete(name = "Prior", labels = c("Optimal profile","Experiment")     )+
  scale_y_continuous(labels = scales::scientific)+   # write number in 
  scale_x_continuous(labels = scales::scientific  )+ # scientific notation.
  theme(legend.position = "none")
# +breaks = c(6E+05, 10E+05,14E+05)  xlim(1E+03,1.6E+03)
print(p)
# ,breaks = c(6E+05, 10E+05,14E+05)

  
}

##################
Compute_posterior_M1 <- function(k,prior,L)
# Compute the posterior through the use of Bayes theorem. 
# "k, "prior" and "L" are all one-dimensional vectors
# which must be extracted from the data frames before calling that function. 
{

# print("Compute_posterior_M1"); 
# print(data.frame(k,prior,L)); 

	
	
  n_k = length(k)         # Number of k values.
                          # Must be the same for all 3 vector arguments.
  
 posterior <-  rep(1,n_k)   # Declaration and initialisation of the vector. 
                            # Without such an initialisation, 
                            # posterior[i] = .... leads to an error message. 
 
  
 normalisation = 0   # Normalisation constant so that
                     # the posterior is a true probability density. 
 
                    # It is actually an integral which will be 
                    # calculated through the loop. 
 
for (i in 1:n_k) {
            posterior[i]=L[i]*prior[i]
  if (2 <= i)  {normalisation =  normalisation + 0.5*(posterior[i-1]+posterior[i])*(k[i]-k[i-1])  }
             }  # End for 
 posterior=posterior/normalisation   # Normalisation of the posterior
 
 write.table(normalisation, file="normalisation.csv", row.names=FALSE, col.names=FALSE, append = FALSE)  
 # The normalisation constant is useful to compute the 
 # Bayes' factor while comparing two models.
 # In this file, the THREE normalisation constants corresponding 
 # to the 3 posterior distributions have been saved.
 
 return(posterior)
  
}

##############



Compute_all_posteriors_M1 <- function(R0,lbound,ubound,n_k,epsilon=0.65,lb_plot,ub_plot)
{  # Compute the 3 posteriors corresponding to the 3 priors.
  # This function performs the ENTIRE Bayesian analysis from the very beginning. 
  # So normally you can call it directly without having to
  # gather intermediate results from older functions. 
  

  # This function returns the three normalisation constants 
  # corresponding to the three priors.
  # [lbound;ubound] is the definition range of k. 
  # [lb_plot;ub_plot] is the range of k which will be used in the plot. 
  # By default, it is the same as the definition range [lbound;ubound]. 

# print(c("R0",R0,"lbound",lbound,"ubound",ubound,"n_k",n_k,"epsilon",epsilon,"lb_plot",lb_plot,"ub_plot",ub_plot)); 
  
  priors <-  produce_all_priors_M1(lbound,ubound,n_k)
# We begin by producing the 3 priors.     
  
  
# integrate_density_M1 <- function(k,density_k,lb = 0, ub=1E+99)  
  
int_prior <- integrate_3densities_M1(priors)
# Integrate the 3 priors over all their
# parameter values. 

print(c("Prior integrals: ",int_prior))
# Must be equal or very close to 1.

k_all = priors[,1]
f_k_k = priors[,2]
f_u_k = priors[,3]
f_w_k = priors[,4]


# Computation of the likelihood function

max_likelihood <- Compute_likelihood_all_M1(R0,lbound,ubound,n_k,epsilon)

print(c("Max likelihood: ",max_likelihood))
# It is advisable to plot both the likelihood function 
# and the posterior around "max_likelihood" 
# as these functions will be very close to 0 
# elsewhere. 

L_all_frame <- read.table("Likelihood_M1.csv", header=FALSE)

colnames(L_all_frame) <- c("k_all","L_all")     
L_all <- L_all_frame$L_all

# print("Likelihood"); print(  data.frame(k_all,L_all)  );


# We'll now compute the posterior corresponding to f_k, f_u_k and f_w_k

posterior_k <- Compute_posterior_M1(k_all,f_k_k,L_all)
norm_k = scan("normalisation.csv")

posterior_u <- Compute_posterior_M1(k_all,f_u_k,L_all)  
norm_u = scan("normalisation.csv")

posterior_w <- Compute_posterior_M1(k_all,f_w_k,L_all)
norm_w = scan("normalisation.csv")


# We check that they're probability densities, 
# i.e. that their integrals are equal to 1. 

I_k <- integrate_density_M1(k_all,posterior_k)

I_u <- integrate_density_M1(k_all,posterior_u)

I_w <- integrate_density_M1(k_all,posterior_w)   

norm <- c(norm_k,norm_u,norm_w)

int = c(I_k,I_u,I_w)

print(c("Posterior integrals: ",int))

# We now save the 3 posteriors in 3 different files

posterior_k_frame = cbind(k_all,posterior_k)
posterior_u_frame = cbind(k_all,posterior_u)
posterior_w_frame = cbind(k_all,posterior_w)

# We save the three posteriors into three files. 
write.table(posterior_k_frame, file="posterior_k_M1.csv", row.names=FALSE, col.names=FALSE, append = FALSE)  
write.table(posterior_u_frame, file="posterior_u_M1.csv", row.names=FALSE, col.names=FALSE, append = FALSE)  
write.table(posterior_w_frame, file="posterior_w_M1.csv", row.names=FALSE, col.names=FALSE, append = FALSE)  



# We'll now plot the three posteriors. 
# We begin by creating a data frame for the graph. 

posteriors <- data.frame(k_all,posterior_k,posterior_u,posterior_w)


p = ggplot() + 
  geom_line(data = posteriors, aes(x = k_all, y = posterior_k, color = "f_k"))+
  geom_line(data = posteriors, aes(x = k_all, y = posterior_u, color = "f_u"))+ 
  geom_line(data = posteriors, aes(x = k_all, y = posterior_w, color = "f_w"))+
 scale_x_continuous(labels = scales::scientific,trans='log10')+
  scale_y_continuous(labels = scales::scientific)+  
  scale_color_discrete(name = "Posterior", labels = c(expression(f[k]),expression(f[u]),expression(f[w]))     )+
  xlab(expression( paste("k (",s^{-1}, ")" ) ))+
  coord_cartesian(xlim =c(lb_plot,ub_plot))

print(p) 

print(c("Normalisation constants : ",norm ) )

return(norm)          # Three normalisation constants


}





