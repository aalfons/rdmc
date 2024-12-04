## load packages
library("rdmc")

## additional functions
# map a vector to an interval (to be used in MNAR settings)
map_to_interval <- function (x, min, max) {
  # map to interval [0, 1]
  r <- range(x)
  z <- (x - r[1]) / diff(r)
  # map to interval [min, max]
  z * (max - min) + min
}
# function to compute mode, with random selection in case of multiple modes 
# (currently commented out in package rdmc)
randomized_mode <- function(x) {
  # compute contingency table
  tab <- table(x)
  # determine the maximum frequency
  max <- max(tab)
  # keep all values that occur with the maximum frequency
  out <- as.numeric(names(tab)[tab == max])
  # if there are multiple modes, randomly select one to be returned
  if (length(out) > 1) sample(out, 1)
  else out
}

## control paramaters for data generation
n <- 300    # number of observations
p <- 200    # number of variables 
q <- 20     # rank of latent continuous variables
sigma <- 1  # standard deviation of noise

## control parameters for discretization
nb_cat <- 5

## control parameters for missing values
pct_min <- 0.4   # for variable with largest mean shift
pct_max <- 0.99  # for variable with smallest mean shift

## other control parameters
R <- 100          # number of simulation runs
seed <- 20240806  # seed of the random number generator

## control parameters for methods
loss <- "pseudo_huber"
type <- "svd"
svd_tol <- 1e-4
delta <- 1.05
mu <- 0.1
conv_tol <- 1e-4
max_iter_RDMC <- 100


## generate data

# set seed of random number generator
set.seed(seed)

# generate data from Gaussian factor model with Gaussian noise
X_continuous <-
  matrix(rnorm(n*q), nrow = n, ncol = q) %*%
  matrix(rnorm(q*p), nrow = q, ncol = p) +
  sigma * matrix(rnorm(n*p), nrow = n, ncol = p)
# scale data matrix by population standard deviation (same for all elements)
X_continuous <- X_continuous / sqrt(q + sigma^2)

# Draw random mean shifts so that the discretized matrix contains popular 
# and unpopular items in the columns (with distributions that are left- or 
# right-skewed). Here we draw the shifts from the interval [-1, 1], which 
# will later be scaled according to a maximum shift depending on the number 
# of categories.
shifts <- runif(p, min = -1, max = 1)

# additional control parameters for discretization
values <- 1:nb_cat            # rating-scale values
midpoint <- (1 + nb_cat) / 2  # rating-scale midpoint
breaks <- c(-Inf, seq(from = -1.5, to = 1.5, by = 1), Inf)

# additional control parameter for random mean shift to create popular and 
# unpopular items (40% of cells in most popular item are expected to 
# receive highest rating, same for least popular item with lowest rating)
max_shift <- breaks[nb_cat] + qnorm(0.4)

# add a random shift to the continuous data so that the discretized matrix 
# contains popular and unpopular items in the columns (with distributions 
# that are left- or right-skewed)
X_shifted <- sweep(X_continuous, 2, shifts * max_shift, "+")

# discretize the generated data
X <- apply(X_shifted, 2, function(x) as.numeric(cut(x, breaks = breaks)))

# map mean shifts to interval [pct_min, pct_max] so that the 
# percentage of missingness in a variable depends on its shift 
# (most popular items having lowest percentage)
pct_NA_MNAR <- map_to_interval(-shifts, min = pct_min, max = pct_max)
# list of index vectors of observations to be set to NA for each variable
# (for smaller percentages of missing values, we only take the first few)
# random sampling within each variable (still creates MNAR as cells with 
# lower scores will have higher probability of being missing due to
# overrepresentation in unpopular items with high percentage of missingness)
set_NA_MNAR <- lapply(pct_NA_MNAR, function(pct) sample.int(n, pct * n))

# loop over variables and take the first few elements of the prepared 
# list of index vectors according to the current percentage, and convert
# to an index vector for the matrix to be compatible with MCAR setting
set_NA <- unlist(lapply(
  seq_len(p), 
  function(j) {
    keep <- seq_len(pct_NA_MNAR[j] * n)
    (j-1) * n + set_NA_MNAR[[j]][keep]
  } 
))

# set the corresponding cells to NA
X_NA <- X
X_NA[set_NA] <- NA_real_

# robust low-rank matrix completion for rating-scale data
fit_RDMC <- rdmc(X_NA, values = values, type = type, svd_tol = svd_tol, 
                 loss = loss, delta = delta, mu = mu, conv_tol = conv_tol, 
                 max_iter = max_iter_RDMC)
