#create a matrix with the desired eigenvalues
A <- diag(c(-1, -2 , -2 , -5/2))
#make some complex eigenvalues
A[[2,3]] <- -1/2
A[[3,2]] <- 1/2

#eigenvalues of A = -5/2, -2 +- 0.5i, -1
#note that you could randomize the values in A

#use a random matrix for perturbation
B <- matrix(runif(16, -1, 1), nrow = 4)

eps <- 0.3

eigen(A + eps*B) #eigenvalues are still good

#so the actual matrix is

C <- A + eps*B

X_eq <- runif(4, 10,20)

Coef <- matrix(0,4,4)

offDiag <- diag(1/X_eq) %*% C

Coef[ upper.tri(Coef) | lower.tri(Coef) ] <- offDiag[ upper.tri(Coef) | lower.tri(Coef) ] #beta[i,j]
# Remove main diag

alphas <- -diag(C) - as.vector(Coef %*% X_eq)

diag(Coef) <- -(alphas + as.vector(Coef %*% X_eq))/X_eq #beta[i,i]

#equilibrium check
alphas*X_eq + X_eq * as.vector(Coef %*% X_eq) #good

#verify jacobian at equilibrium is C
range(diag(alphas) + diag(as.vector(Coef %*% X_eq)) + diag(X_eq) %*% Coef - C) #looks good, numeric error


finalGLV <- list("alpha" = alphas, "beta" = Coef, "Equilibrium" = X_eq, "JacobianAtEq" = C, "Eigenvalues" = eigen(C)$values)




