import argparse
import numpy as np
import math
from scipy.stats import norm, chi
from z3 import *

# Compute the number of feasible partition combinations for the dimensions
# of a d-dimensional spherical Gaussian
#
# Partitions are uniform across dimensions and encoded as an array where
# each partition is the interval [partitions[i], partitions[i+1]); closed
# on the left and open on the right with the final interval closed on the
# right with the maximal value.
#
# Computation is via a DFS of a partition assignment tree
#  - a level of the tree corresponds to an arbitrary dimension
#  - vertices at a level of the tree correspond to assignments of partitions to the dimension at that level
#  - a vertex encodes a combination of partition assignments, i.e., the combination along the tree path from the root
#  - a vertex count stores the number of feasible combinations that can be constructed from the given combination of the vertex


# Parameters of the problem
#dim = 9
#parts = list()
#numparts = 4
#radin = 2.8
#radout = 3.0
#way = 2

# visited stores canonical representatives of tree paths and the counts reachable from those paths
visited = dict()

# for debugging we record the set of pruned and infeasible path prefixes
verbose = False
pruned = set()
infeasible = set()

# Constraint and Z3 solver related structures
s = Solver()
Z = list()
disableconstraints = False

parser = argparse.ArgumentParser('Generate count of t-way combinations of partitions of spherical Gaussian')


# Generate an encoding of the distance constraint
#   - initialize non-negative variables for squares of each dimension
#   - establish lower bound as radin^2
#   - establish upper bound as radout^2
def distanceconstraint():
    global Z, s, partmax, dim
    Z = [Real('z_%s' % (i + 1)) for i in range(dim)]
    zlowerbound = [Z[i] >= 0 for i in range(dim)]
    s.add(zlowerbound)
    zupperbound = [Z[i] < (partmax*partmax) for i in range(dim)]
    s.add(zupperbound)
    zsum = 0
    for i in range(dim):
        zsum = zsum + Z[i]
    s.add(zsum >= (radin*radin), zsum < (radout*radout))
    return


def checkstateconstraint(depth, partition):
    global Z, s, disableconstraints, verbose

    if not disableconstraints:
        s.push()
        if partition[0] < 0:
            s.add(Z[depth] <= (partition[0]*partition[0]), Z[depth] > (partition[1]*partition[1]))
        else:
            s.add(Z[depth] >= (partition[0]*partition[0]), Z[depth] < (partition[1]*partition[1]))
        if verbose:
            print(f"checking partition {partition} at depth {depth}")
            print(f"   with constraint {s.assertions()}")
        return s.check() == sat
    else:
        return True


def dropstateconstraint():
    global s
    if not disableconstraints:
        s.pop()
    return


# Perform a DFS through levels of the partition assignment tree
def dfs(depth, state):
    global parts, way, pruned, visited, unsatstates

    count = 0

    if depth < way:
        for p in range(2*numparts):
            nextstate = list(state)
            nextstate.append(parts[p][0])
            nextstate.sort()
            nextstr = str(nextstate)

            subcount = 0

            # check to see if the next state has been visited
            # and if so skip it
            if nextstr not in visited:
                state.append(parts[p][0])

                # check the satisfiability of the state with the accumulated distance and partition constraints
                if checkstateconstraint(depth, parts[p]):
                    subcount = dfs(depth+1, state)
                else:
                    infeasible.add(nextstr)

                dropstateconstraint()
                state.pop()
                visited[nextstr] = subcount
            else:
                prunedstate = list(state)
                prunedstate.append(parts[p][0])
                prunedstr = str(prunedstate)
                pruned.add(prunedstr)
                subcount = visited[nextstr]

            count = count + subcount
    else:
        count = 1

    return count


def main():
    global s, dim, radin, radout, numparts, partmax, parts, way, disableconstraints, verbose

    parser.add_argument('--dim', metavar='Dim', type=int, default=9,
                        help='number of dimensions; defaults to 9')
    parser.add_argument('--numparts', metavar='Parts', type=int, default=4,
                        help='number of partitions of positive coordinates; defaults to 4 meaning that there are 8 total partitions')
    parser.add_argument('--way', metavar='Way', type=int, default=2,
                        help='size of combinatorial sub-domain; defaults to 2')
    parser.add_argument('--verbose', default=False, action="store_true",
                        help='print information about search to generate count')
    parser.add_argument('--unconstrained', default=False, action="store_true",
                        help='do not enforce constraints')
    args = parser.parse_args()

    dim = args.dim
    numparts = args.numparts
    way = args.way
    disableconstraints = args.unconstrained
    verbose = args.verbose
    
    # high dimensional multivariate standard normal distribution density is concentrated
    # in a shell
    #Using chi-distribution
    radin, radout = chi.interval(density, dim)
    partmax = radout
    
    #Below logic divides probability [0,1] into equal density intervals
    #result will be in parts which is a list of tuples
    #each tuple contains lower and upper bounds of random variable for that division
    
    #probability of each division
    partition_density = 1.0/(2*numparts)
    
    #For each of the divisions, convert probability density bounds into random variable bounds
    density_low = 0
    density_high = partition_density
    parts = list()
    for i in range(2*numparts):
        #density can be greater than 1 by a negligible amount due to 
        #density_high variable rhs in the below logic
        if density_high > 1:
            density_high = 1
            
        #below code converts probability bounds to random variable 
        #bounds using quantile function
        rv_low = norm.ppf(density_low)
        rv_high = norm.ppf(density_high)
        
        #below logic limits the random variable partitions to 
        #lie in [-partsmax, partsmax] range
        if rv_low == -np.inf:
            rv_low = -partmax
        elif rv_high > partmax:
            rv_high = partmax
            
        if rv_low >= rv_high:
            print("Error in dividing probability densities")
            exit(0)
        
        if math.isclose(rv_low, 0, abs_tol=1e-5):
            rv_low = 0
        if math.isclose(rv_high, 0, abs_tol=1e-5):
            rv_high = 0
        parts.append((rv_low, rv_high))
            
        density_low = density_high
        density_high = density_high + partition_density  

    if verbose:
        print(f"dim = {dim}, radin = {radin}, radout = {radout}, numparts = {numparts}, parts = {parts}, way = {way}")
    
    # define the distance constraint
    distanceconstraint()

    # perform the search for feasible combinations
    count = dfs(0, list())

    numcombinations = len(parts) ** way
    print(f"Out of {numcombinations} possible {way}-way combinations, {count} are feasible")

    print(f"The number of {way}-way partition combinations, for {dim} dimensions each with {2*numparts} partitions, is {count * math.comb(dim,way)}")

    if verbose:
        print(f"The generated tree paths are:")
        print(*visited, sep="\n")

        print(f"The pruned path prefixes are:")
        print(*pruned, sep="\n")

        print(f"The infeasible path prefixes are:")
        print(*infeasible, sep="\n")


if __name__ == "__main__":
    main()
