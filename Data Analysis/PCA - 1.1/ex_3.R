# 3
# 3.1
X = c(0, 1, 0, 1, 1, 0)
Y = c(0, 1, 1, 0, 1, 0)

arr <- array(c(X,Y), dim = c(6,2,1))
print(arr)

# Poids statistiques
poids <- 1 / length(X)

# Matrice diagonale D des poids des individus
D <- diag(poids, length(X), length(X))
print(D)

# Moyenne:
X.mean <- mean(X)
Y.mean <- mean(Y)

Z <- data.frame(X, Y)
Z$X = Z$X - X.mean
Z$Y = Z$Y - Y.mean
print(Z)

A <- cov(Z)
print(A)

# 3.2
L <- eigen(as.matrix(A))
L$values

# For ease of use
var_x = A[1, 1]
var_y = A[2, 2]
covariance = A[1, 2]
lambda1 = L$values[1]
lambda2 = L$values[2]

trace <- var_x + var_y
sum <- lambda1 + lambda2
if (trace == sum) {
  print("Equal")
} else {
  print("Not equal")
}

det <- var_x * var_y - covariance * covariance * covariance
mult <- lambda1 * lambda2
if (det == mult) {
  print("Equal")
} else {
  print("Not equal")
}

# 3.3
c1 = 1/sqrt(((var_y) - lambda1)^2 + (covariance)^2)
c2 = 1/sqrt(((var_y) - lambda2)^2 + (covariance)^2)
u1 <- -c1 * c(var_y - lambda1, -covariance)
u2 <- -c2 * c(var_y - lambda2, -covariance)
u1
u2
L$vectors

# 3.4
u = cbind(u1,u2)
Is = t(u) %*% A %*% u
Is
L$values

# 3.5
Is1 = t(u1) %*% A %*% u1
Is2 = t(u2) %*% A %*% u2
It = var_x + var_y
print(It)
print(Is1 + Is2)

# 3.6
plot(Z, pch=19)
abline(a=0, b=u1[2]/u1[1], col="blue")
abline(a=0, b=u2[2]/u2[1], col="red")

# 3.7
total <- lambda1 + lambda2
TI1 <- lambda1 / total
TI2 <- lambda1 / total
print(paste("Taux d’inertie expliqué par la composante 1:", TI1))
print(paste("Taux d’inertie expliqué par la composante 2:", TI2))

# 4
GMi1 <- rep(0, nrow(Z))
for (i in 1:nrow(Z)) {
  GMi1[i] = c(Z[i,1], Z[i,2]) %*% u1
}
GMi1

GMi2 <- rep(0, nrow(Z))
for (i in 1:nrow(Z)) {
  GMi2[i] = c(Z[i,1], Z[i,2]) %*% u2
}
GMi2
GMi = cbind(GMi1, GMi2)

# 5
Qi = rep(0, nrow(Z))
for (i in 1:nrow(Z)) {
  Qi[i] = sum(GMi^2)/sum(Z^2)
}
Qi

# 6
gammai_1 = rep(0, nrow(Z))
for (i in 1:nrow(Z)) {
  gammai_1[i] = (1/nrow(Z)) * (GMi1[i]^2/L$values[1])
}
gammai_1

gammai_2 = rep(0, nrow(Z))
for (i in 1:nrow(Z)) {
  gammai_2[i] = (1/nrow(Z)) * (GMi2[i]^2/L$values[2])
}
gammai_2

# 7
# Les points 3 et 4 sont mieux projetés et contribuent le plus:
# Les points (1 et 2), et (5 et 6) sont confondues sur les nouveaux axes.
# Les points (3 et 4) présentent une variance plus grande et les autres.


# TP
# Ce code prend une matrice de la forme p*n et fait les modifications necessaires
ACP_function <- function(matrice, norme) {
  # Standardization
  m_ce <- NULL

  if (norme) {
    for(i in 1:nrow(matrice)) {
      m_ce <- rbind(m_ce, (matrice[i,] - mean(matrice[i,])) / (sd(matrice[i,])))
    }
  } else {
    m_ce <- matrice - mean(matrice)
  }

  # Covariance matrix
  m_ce <- as.data.frame(t(m_ce))
  m_cov <- cov(m_ce)

  # Eigen values and Eigen vectors
  vp <- eigen(m_cov)
  vectorP <- vp$vectors

  # PCA
  m_ce_t <- t(m_ce)
  var <- rep(0 , nrow(m_ce))
  var <- t(vectorP) %*% m_ce_t
  var <- as.data.frame(t(var))

  print(var)

  # Contributions
  valeursP <- vp$values
  percentage <- valeursP/(nrow(matrice)-1)
  percentage <- round((percentage/sum(percentage))*100, 1)

  return(var)
}

# Testing
data <- data.frame(X, Y)
df <- ACP_function(t(data), FALSE)
