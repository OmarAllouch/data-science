library("readxl")

# Read data from excel
X <- read_excel("TP_AFC_majeur1718_travail.xlsx", na = " ")
X <- as.data.frame(X[, 2:3])
X <- na.omit(X)

# Add column names
V0 <- with(X, table(Sexe, Fonction))
rownames(V0) <- c("Non répondu", "H", "F")
colnames(V0) <- c(
  "Non répondu", "Administratif", "Technicien (OS)",
  "Ingénieur", "Technicien supérieur", "Direction",
  "Contractuel S1", "Contractuel S2"
)

k <- 248
V <- V0 / k
print(V)

I <- length(unique(X[, 1]))
J <- length(unique(X[, 2]))

Dn <- diag(I)

rownames(Dn) <- c("Non répondu", "H", "F")
colnames(Dn) <- c("Non répondu", "H", "F")

Dp <- diag(J)

rownames(Dp) <- c(
  "Non répondu", "Administratif", "Technicien (OS)",
  "Ingénieur", "Technicien supérieur", "Direction",
  "Contractuel S1", "Contractuel S2"
)
colnames(Dp) <- c(
  "Non répondu", "Administratif", "Technicien (OS)",
  "Ingénieur", "Technicien supérieur", "Direction",
  "Contractuel S1", "Contractuel S2"
)

for (i in 1:I) {
  S <- 0
  for (j in 1:J) {
    S <- S + V[i, j]
  }
  Dn[i, i] <- S
}

for (j in 1:J) {
  S <- 0
  for (i in 1:I) {
    S <- S + V[i, j]
  }
  Dp[j, j] <- S
}

print(Dn)
print(Dp)

line_profiles <- solve(Dn) %*% V
print(line_profiles)

column_profiles <- solve(Dp) %*% t(V)
print(column_profiles)

S <- t(V) %*% line_profiles %*% solve(Dp)

A <- sqrt(solve(Dp)) %*% t(V) %*% line_profiles %*% sqrt(solve(Dp))
print(A)

decomp1 <- eigen(A)
eigen_values1 <- decomp1$values
eigen_vectors1 <- decomp1$vectors

print(eigen_values1)
print(eigen_vectors1)
u1 <- sqrt(Dp) %*% eigen_vectors1[, 1]
u2 <- sqrt(Dp) %*% eigen_vectors1[, 2]

C1 <- V %*% u1
C2 <- V %*% u2

plot(C1, C2, type = "p", xlim = c(-0.15, 0.05), ylim = c(-0.1, 0.1), col = "blue")
text(C1, C2, rownames(V), cex = 0.7, pos = 3, col = "blue")
abline(h = 0, col = "gray")
abline(v = 0, col = "gray")

T <- V %*% column_profiles %*% solve(Dn)

A2 <- sqrt(solve(Dn)) %*% V %*% column_profiles %*% sqrt(solve(Dn))
print(A2)

decomp2 <- eigen(A2)
eigen_values2 <- decomp2$values
eigen_vectors2 <- decomp2$vectors

print(eigen_values2)
print(eigen_vectors2)
u1_2 <- sqrt(Dn) %*% eigen_vectors2[, 1]
u2_2 <- sqrt(Dn) %*% eigen_vectors2[, 2]

C1_2 <- t(V) %*% u1_2
C2_2 <- t(V) %*% u2_2

plot(C1_2, C2_2, type = "p", xlim = c(-0.15, 0.05), ylim = c(-0.1, 0.1), col = "blue")
text(C1_2, C2_2, colnames(V), cex = 0.7, pos = 3, col = "blue")
abline(h = 0, col = "gray")
abline(v = 0, col = "gray")

# All points in the same plot
plot(C1, C2, pch = 24, cex = 1, col = "blue", lwd = 2, xlim = c(-0.15, 0.05), ylim = c(-0.1, 0.1))
text(C1, C2, rownames(V), cex = 0.75, col = "blue", pos = 3)
points(C1_2, C2_2, pch = 25, cex = 1, col = "red", lwd = 2)
text(C1_2, C2_2, colnames(V), cex = 0.75, col = "red", pos = 1)
abline(h = 0, col = "gray")
abline(v = 0, col = "gray")


# Nouvelles coordonnées dans Rp
co_lignes2 <- V %*% sqrt(Dp) %*% eigen_vectors1

# Nouvelles coordonnées dans Rn
co_colonnes2 <- t(V) %*% sqrt(Dn) %*% eigen_vectors2

quality <- function(n, i, k) {
  A <- sum((n[i, ]^2))
  B <- sum((n[i, 1:k]^2))
  return(B / A)
}

# Qualité des projections sur Rp
quality1 <- rep(0, 3)

for (i in 1:3) {
  quality1[i] <- quality(co_lignes2, i, 2)
}

# Qualité des projections sur Rp
quality2 <- rep(0, 7)

for (i in 1:7) {
  quality2[i] <- quality(co_colonnes2, i, 2)
}

print(quality1)
print(quality2)
