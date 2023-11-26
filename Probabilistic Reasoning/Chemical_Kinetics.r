#install.packages("gdata")
#install.packages("ggplot2")
#install.packages("ggplot")
#install.packages("scales")



library(datasets)
library(ggplot2)
source(file = "Chemical_Kinetics_Functions_M1.r")


##################################

#. Part I: Model M1


# 2)
lb <- 2
ub <- 4.26e+03
n_k <- 1e+04

test_produce_priors_M1(lb, ub, n_k)
####

# 3)
R0 = 1200
n_t = 1000
k = 1.357e+03
step = (1000 - 0) / (n_t - 1)
t_model <- seq(0, 1000, step)

R_model = Compute_R_profile_M1(t_model, R0, k)
plot_profile_M1(t_model, R_model)
####

# 4)
R0 = 1200
lbound = 1e+03
ubound = 1.75e+03
n_k = 1000
epsilon = 0.065

Examine_likelihood_M1(R0, lbound, ubound, n_k, epsilon)
####

#5)
R0 = 1200
lbound = 2
ubound = 4.26e+03
n_k = 1500
epsilon = 0.065
lb_plot = 1.125e+03
ub_plot = 1.625e+03

Compute_all_posteriors_M1(R0, lbound, ubound, n_k, epsilon, lb_plot, ub_plot)
####

# 6)
priors <- produce_all_priors_M1(lbound, ubound, n_k)
lb = 1000
ub = 1500
integrate_3densities_M1(priors, lb, ub)
####

# 7)
A <- read.table("posterior_k_M1.csv", header = FALSE)
k <- A[, 1]
f_k_k <- A[, 2]

A <- read.table("posterior_u_M1.csv", header = FALSE)
k <- A[, 1]
f_u_k <- A[, 2]

A <- read.table("posterior_w_M1.csv", header = FALSE)
k <- A[, 1]
f_w_k <- A[, 2]

posteriors <- cbind(k, f_k_k, f_u_k, f_w_k)

lb = 1000
ub = 1500

integrate_3densities_M1(posteriors, lb, ub)
####

# 8)
R0 = 1200
lbound = 2
ubound = 4.26E+03
n_k = 1500
lb_plot = 0.75E+03
ub_plot = 2.5E+03

epsilon = 0.065
Compute_all_posteriors_M1(R0, lbound, ubound, n_k, epsilon, lb_plot, ub_plot)

epsilon = 0.30
Compute_all_posteriors_M1(R0, lbound, ubound, n_k, epsilon, lb_plot, ub_plot)
####

# 9)
A <- read.table("posterior_k_M1.csv", header = FALSE)
k <- A[, 1]
f_k_k <- A[, 2]

A <- read.table("posterior_u_M1.csv", header = FALSE)
k <- A[, 1]
f_u_k <- A[, 2]

A <- read.table("posterior_w_M1.csv", header = FALSE)
k <- A[, 1]
f_w_k <- A[, 2]

posteriors <- cbind(k, f_k_k, f_u_k, f_w_k)

lb = 1000
ub = 1500

integrate_3densities_M1(posteriors, lb , ub)


lb_plot = 0.75E+03
ub_plot = 2.5E+03

epsilon = 0.065
Examine_likelihood_M1(R0, lbound, ubound, n_k, epsilon)

epsilon = 0.30
Examine_likelihood_M1(R0, lbound, ubound, n_k, epsilon)
####




###################################################################"

# Part II: Model M2


# 2)
source(file = "Chemical_Kinetics_Functions_M2.r")
lb <- 5e+04
ub <- 5e+09
n_k <- 1000

test_produce_priors_M2(lb, ub, n_k)
###


# 3)
R0 = 1200
lbound = 5.8E+05
ubound = 1.4E+06
n_k = 1000
epsilon = 0.065
Examine_likelihood_M2(R0, lbound, ubound, n_k, epsilon)
###

# 4)
R0 = 1200
n_t = 1000
k = 1004821.17971131
step = (1000 - 0) / (n_t - 1)
t_mod <- seq(0, 1000, step)
R_mod <- Compute_R_profile_M2(t_mod, R0, k)
plot_profile_M2(t_mod, R_mod)
###

# 5)
lbound = 5e+05
ubound = 5e+09
lb_plot = 7e+05
ub_plot = 1.4e+06
Compute_all_posteriors_M2(R0, lbound, ubound, n_k, epsilon, lb_plot, ub_plot)
###



###################################################################"

#  III. Bayes factor


# 1)
lbound1 = 2
ubound1 = 4.26e+03
n_k = 1000
epsilon = 0.065
lb_plot1 = 1e+03
ub_plot1 = 1.6e+03
norm1 <-
  Compute_all_posteriors_M1(R0, lbound1, ubound1, n_k, epsilon, lb_plot1, ub_plot1)

lbound2 = 5e+05
ubound2 = 5e+09
lb_plot2 = 5.8e+05
ub_plot2 = 1.4e+06
norm2 <-
  Compute_all_posteriors_M2(R0, lbound2, ubound2, n_k, epsilon, lb_plot2, ub_plot2)

B_k <- norm2[1] / norm1[1]
B_k
###


#  2)
B = norm2 / norm1
B
###


#  3)
###
