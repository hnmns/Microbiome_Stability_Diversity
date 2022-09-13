#create a matrix with the desired eigenvalues
#A <- diag(c(-1, -2 , -2 , -5/2))

A <- diag(c(runif(4,-4,-1)))

#make some complex eigenvalues
A[[2,3]] <- runif(1,1,4)
A[[3,2]] <- -1 * A[[2,3]]

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

alphas <- -diag(C) - as.vector(Coef %*% X_eq)

diag(Coef) <- -(alphas + as.vector(Coef %*% X_eq))/X_eq #beta[i,i]

#equilibrium check
alphas*X_eq + X_eq * as.vector(Coef %*% X_eq) #good

#verify jacobian at equilibrium is C
range(diag(alphas) + diag(as.vector(Coef %*% X_eq)) + diag(X_eq) %*% Coef - C) #looks good, numeric error


finalGLV <- list("alpha" = alphas, "beta" = Coef, "Equilibrium" = X_eq, "JacobianAtEq" = C, "Eigenvalues" = eigen(C)$values)

library(deSolve)


GLVmod <- function(Time, State, Pars){
  with(as.list(c(State,Pars)), {
    dW <- alpha[1]*W + beta[1,1]*W*W + beta[1,2]*W*X + beta[1,3]*W*Y + beta[1,4]*W*Z
    dX <- alpha[2]*X + beta[2,1]*W*X + beta[2,2]*X*X + beta[2,3]*X*Y + beta[2,4]*X*Z
    dY <- alpha[3]*Y + beta[3,1]*W*Y + beta[3,2]*X*Y + beta[3,3]*Y*Y + beta[3,4]*Y*Z
    dZ <- alpha[4]*Z + beta[4,1]*W*Z + beta[4,2]*X*Z + beta[4,3]*Y*Z + beta[4,4]*Z*Z
    return(list(c(dW,dX,dY,dZ)))
       })
}

GLVmod(0, State = c(W=1,X=1,Y=1,Z=1), Pars = list(alpha = finalGLV$alpha, beta = finalGLV$beta) )

results <- ode(c(W=1,X=1,Y=1,Z=1),seq(0,10,by=0.05),GLVmod, list(alpha = finalGLV$alpha, beta = finalGLV$beta))

matplot(results[,1], results[,2:5], type = "l")
tail(results)
finalGLV$Equilibrium
finalGLV$Eigenvalues 
