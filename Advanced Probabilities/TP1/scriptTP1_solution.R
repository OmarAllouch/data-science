# TP1 Probabilités Avancées - UP1
# Majeure science des données 2023-2024
# Solution R

################################################################
# Partie 1. Loi normale et approximation de la loi binomiale #
################################################################

# Loi normale centrée-réduite

xx <- seq(-3,3,0.01)
pxx <- dnorm(xx, mean=0, sd=1) # densité de la loi N(0,1) calculée aux points xx
plot(xx, pxx,'l', col="red", lwd=2, xlab='x', ylab='f(x)',
     main="Densité de la loi normale N(0,1)", cex.main=1)
grid()
abline(h=0,lty=2)
abline(v=0,lty=2)

# Approximation de la loi binomiale B(n,p) par la loi normale N(np,np(1-p)) 
# en vertu du Théorème Central-Limite (TCL)

n <- 12
p <- 0.5
xx <- seq(0,n,1)
pxx <- dbinom(xx,size=n,prob=p) # probabilités discrètes de la loi binomiale B(n,p)

xx <- seq(0,n,0.01)
plot(xx, dnorm(xx, mean=n*p, sd=sqrt(n*p*(1-p))), 'l', col="red", lwd=2,
     ylab="densité", xlab="x")
grid()
abline(h=0, lty=2)
for (k in 0:n){
  lines(c(k-0.5,k-0.5), c(0,pxx[k+1]), 'l')
  lines(c(k-0.5,k+0.5), c(pxx[k+1],pxx[k+1]), 'l')
  lines(c(k+0.5,k+0.5), c(0,pxx[k+1]), 'l')
}
axis(1, 0:n,0:n)
title('Loi binomiale B(n=12, p=0.5) et loi normale N(np, np(1-p)', cex.main=1)

# Comparaison loi binomiale et loi normale 
# On simule la loi binomiale B(n,p) et on compare avec la distribution normale

n <- 100 
p <- 0.5
N <- 4000 # nombre de simulations de la loi binomiale
s <- rbinom(N, size=n,prob=p) # s = échantillon de taille N de la loi binomiale B(n,p) 
z <- (s - n*p)/sqrt(n*p*(1-p)) # z = échantillon après centrage et réduction

# Comparaison 1: histogramme de l'échantillon z et densité de la loi normale N(0,1)

hist(z, prob=TRUE, xlab="z", ylab = "", 
     main="Histogramme et densité N(0,1)", breaks=30, cex.main=1)
xx <- seq(-3.5,3.5,0.01)
lines(xx, dnorm(xx,mean=0,sd=1),'l',col="red",lwd=2)

# Comparaison 2: quantiles empiriques contre quantiles de la loi normale N(0,1)

qqnorm(z) # sur les données centrées-réduites
qqline(z, col="red", probs=c(0.1,0.9), lwd=2)
grid()

qqnorm(s) # sur les données brutes
qqline(s, col="red", probs=c(0.1,0.9), lwd=2)
grid()

# Conclusion : le TCL est bien vérifié  

#########################################################
# Partie 2. Loi trinomiale et loi gaussienne bivariée #
#########################################################

# Extension de la loi binomiale : loi trinomiale associée à une variable
# à 3 modalités (ou 3 classes) numérotées 0, 1 et 2

p0 <- 0.5 # probabilité de la classe 0 
p1 <- 0.25 # probabilité de la classe 1
p2 <- 0.25 # probabilité de la classe 2

if (p0+p1+p2 != 1) print("Erreur : la somme des probabilités est différente de 1")

# Simulation de la loi trinomiale (loi multinomiale à 3 classes) 

N <- 5 # nombre de simulations de la loi trinomiale
n <- 100 # nombre d'épreuves = nombre de tirages d'une variable à 3 modalités
s <- rmultinom(N, size=n, prob=c(p0,p1,p2))
print(s)
apply(s, 2,"sum") # somme sur les colonnes

n0 <- s[1,] # échantillon loi B(n,p0) = loi du nombre N0 d'éléments dans la classe 0
n1 <- s[2,] # échantillon loi B(n,p1) = loi du nombre N1 d'éléments dans la classe 1
n2 <- s[3,] # échantillon loi B(n,p2) = loi du nombre N2 d'éléments dans la classe 2

# Question 1
p0 <- 0.1   # probabilité de la classe 0 
p1 <- 0.5   # probabilité de la classe 1
p2 <- 0.4   # probabilité de la classe 2
n <- 500    # nombre d'épreuves = nombre de tirages d'une variable à 3 modalités
N <- 100    # nombre de simulations de la loi trinomiale

s <- rmultinom(N, size=n, prob=c(p0,p1,p2))
dim(s)

n1 <- s[2,] # échantillon loi B(n,p1) = loi du nombre N1 d'éléments dans la classe 1
n2 <- s[3,] # échantillon loi B(n,p2) = loi du nombre N2 d'éléments dans la classe 2

z <- matrix(0, nrow=2, ncol=N) # initialisation des variables centrées-réduites
z[1,] <- (n1 - n*p1)/sqrt(n*p1*(1-p1))
z[2,] <- (n2 - n*p2)/sqrt(n*p2*(1-p2))
 
# Question 2 : normalité de Z1 et Z2
qqnorm(z[1,]); qqline(z[1,], col="red", probs=c(0.1,0.9), lwd=2); grid()
qqnorm(z[2,]); qqline(z[2,], col="red", probs=c(0.1,0.9), lwd=2); grid()

# Question 3 : corrélation entre Z1 et Z2
plot(z[1,], z[2,], type='p', pch='+', asp=1, 
     xlab="z1", ylab="z2")
abline(h=0, lty=2)
abline(v=0, lty=2)

# Corrélation théorique  
rho <- - p1*p2/sqrt(p1*(1-p1)*p2*(1-p2)) 

# Droite de régression théorique de Y2 sur Y1 : pente = rho
abline(a=0, b=rho, col="red", lwd=2) 

# Corrélation empirique (estimée sur les données simulées)
rho.emp <- cor(z[1,], z[2,]) 

# Comparaison 
cat(" Corrélation théorique : ", round(rho,3), "\n",
    "Corrélation empirique : ", round(rho.emp,3), "\n")

# Cas limite : p0 proche de 0 -> N2 ~ n - N1 (car N0 ~ 0)
# donc rho proche de - 1
# Vérification par simulation
p0 <- 0.01   # probabilité de la classe 0 
p1 <- 0.5   
p2 <- 0.49  
n <- 500    # nombre d'épreuves 
N <- 100    # nombre de simulations 

s <- rmultinom(N, size=n, prob=c(p0,p1,p2))
n1 <- s[2,] # échantillon loi B(n,p1) 
n2 <- s[3,] # échantillon loi B(n,p2) 

cat("\n Corrélation empirique avec p0 = 0.01 :", 
    round(cor(n1, n2),3))

# Question 4 : changement de repère orthonormé
# Nouvelles variables C1 et C2
# combinaisons linéaires des variables initiales
C1 <- (sqrt(2)/2)*(z[1,]+z[2,]) 
C2 <- (sqrt(2)/2)*(-z[1,]+z[2,])

# On intervertit les 2 nouveaux axes pour plus de
# lisibilité
plot(C2, C1, type='p', pch='+', asp=1,
     xlab="C2", ylab="C1") 
grid()
abline(h=0, lty=2)
abline(v=0, lty=2)

# Il semble que les nouvelles variables soient non corrélées,
# voire indépendantes 

# Question 5
# Normalité de C1 et variance estimée
qqnorm(C1); qqline(C1,col="red", probs=c(0.1,0.9), lwd=2); grid()
v1 <- var(C1)
cat("  Variance estimée de C1  : ", round(v1,3), "\n",
    " Comparaison avec 1+rho  : ", round(1+rho,3))

# Normalité de C2 et variance estimée
qqnorm(C2); qqline(C2, col="red", probs=c(0.1,0.9), lwd=2)
v2 <- var(C2)
cat("  Variance estimée de C2  : ", round(v2,3), "\n",
    " Comparaison avec 1-rho  : ", round(1-rho,3))

# Question 6 : on trace les lignes de niveau de la densité
# gaussienne en question sachant que ce sont des ellipses  
plot(z[1,], z[2,], type='p',pch='+', asp=1, 
     xlab="z1", ylab="z2")
grid()
abline(h=0,lty=2)
abline(v=0,lty=2)

library(mixtools)  # pour la fonction ellipse()
mu <- c(0,0) # Moyenne
# Matrice de covariance
sigma <- matrix(c(1, rho, rho, 1),nrow = 2) 

for (niv in 1:5){
  ellipse(mu,sigma, alpha=.5/niv, col="red")
}
abline(a=0, b=1, col="blue") # petit axe des ellipses
abline(a=0, b=-1, col="blue") # grand axe
title("Données simulées et courbes d'iso-probabilité",
      cex.main=1)

