# FeasiblePartitions
Compute the number of feasible t-way combinations of partitions of spherical Gaussian

This code addresses the problem of partitioning the volume of a high-dimensional spherical Gaussian based on a partition of each of its dimensions using a Cartesian coordinate system.
The challenge in computing this partition is that the dimensions are coupled, so that not all possible combinations of dimensional partitions are feasible.

Due to concentration, the volume of a spherical Gaussian will comprise an annulus whose center radius is the square-root of its dimension, d.
We assume that the annulus will be partitioned, radially, into a shells with an inner and outer radius; denoted [radin,radout)
This gives rise to a constraint on the dimensions based on the distance formulae:

     radin^2 \le \sum (d_i)^2 \lt radout^2

We assume a partition, P, of each dimension into a set of intervals that are closed on the left and open on the right.
We further assume that the partition is symmetric about 0.  
In other words, we partition a dimension, d_i, by decomposing its range [-max,max) into a set of intervals dividing [0,max) and a complementary set of intervals dividing [-max,0).
Thus the number of partitions is even and no partition spans 0.

A combination of partitions, one per dimension, is feasible with respect to the distance constraint if there exists a point in the set defined by the combination that lies within the target shell.

For example, if the shell has radii [1,2) for a 3-dimensional annulus then the following partition combinations are feasible:

     d_1 = [0,1)
     
this 1-way combination leaves d_2 and d_3 unconstrained and their values could be chosen to define a point in the shell

     d_1 = [0,1), d_2 = [0,1), d_3 = [1,2)
     
this 3-way combination defines value sets for all 3 dimensions and if we choose the value 0 for d_1 and d_2, then we we end up with a version of the distance constraint of the form:

    1^2 \lt [1,2)^2 \le 2^2
    
which is clearly true.
On the other hand the following partition combination is infeasible:

     d_1 = [2,3), d_2 = [1,2)

this 2-way combination leads to the following sum of squares, in interval arithmetic:

     [5,13) = [2,3)^2 + [1,2)^2
     
which falsifies the distance constraint

     1 \lt [5,13) \le 4
     
We can solve these constraints by encoding them as an SMT problem and using a solver such as Z3.
Note that while the distance formulae is quadratic, we don't need to encode this as a non-linear constraint.
Instead we can use variables that encode the squares of the dimensional coordinates - as shown above when squaring the interval [2,3).
This gives rise to a linear constraint involving the summation of the squares of the dimensions.

Choosing partitions that do not span 0 allows us to recover the sign of the underlying dimension if we want to work backwards from the squared dimension variables to produce the coordinates.
We don't need this for coverage, but for test generation this will be necessary.

A strength of combination, t \le d, defines the set of t-way combinations of partition elements across dimensions.
For example, if t = 2, d = 3, and there are 2 partitions, labeled A and B, for each dimension, then the number of t-way combinations is 12 and those combinations are:

     d_1=A,d_2=A
     d_1=A,d_2=B
     d_1=A,d_3=A
     d_1=A,d_3=B
     d_1=B,d_2=A
     d_1=B,d_2=B
     d_1=B,d_3=A
     d_1=B,d_3=B
     d_2=A,d_3=A
     d_2=A,d_3=B
     d_2=B,d_3=A
     d_2=B,d_3=B

There are two degrees of freedom to resolve in computing these combinations:
- which of the d-dimensions are included in t
- of the included dimensions what are their assigned partitions

We are only interested in computing the number of combinations, as opposed to generating the combinations.
This allows us to simplify the problem by computing the number of combinations for an arbitrary choice of t dimensions and then multiplying by the number of possible choises of t-dimensions; which is d choose t (which can be computed using math.comb(d,t)).

In the example above, pick any pair of dimensions (recall t=2) and you will find 4 combinations of partition assignments for those, e.g.

     d_1=A,d_2=A
     d_1=A,d_2=B
     d_1=B,d_2=A
     d_1=B,d_2=B

We then multiply by 3 = comb(3,2) to get the count of 12.

Given an arbitrary choice of t dimensions, we are interested in computing the number of feasible partition assignments to those dimensions.
The above example showed the full set of assignments, but depending on the definition of the partitions and the radial constraints some will be infeasible.

We can compute the number of assignment by performing the exhaustive search of a tree of combinations.
Each level in the tree corresponds to one of the t dimensions.
The branching at each level corresponds to the possble partition choices for that level's dimension.
A tree path thereby defines a combination of partitions for a set of dimensions whose size is equal to the length of the path.

The combinatorics of building this tree is daunting for large t and |P|, but one can exploit the symmetry of this problem to reduce complexity.
Since we are only interested in computing the number of combinations and not the combinations themselves, we can use the fact that addition is commutative and of course the squared-distance constraint involves a summation.

For example, consider the following tree path:

     d_1=[0,1), d_2=[1,2), d_3=[0,1)

This will generate a constraint with a term:

     [0,1)^ + [1,2)^2 + [0,1)^2
     
which is representative of an equivalence class of terms that can be generated by applying commuting the operands of the additions.
They key point is that all of the paths in the tree associated with the members of this equivalence class yield sub-trees of the same size.
So if we compute the size of one such sub-tree, and store it, we can reuse it for any other path in the equivalence class and thereby skip generating the sub-tree, which will yield a combinatorial speedup as t gets big.

To achieve this we use the usual trick of defining a canonical representative of the equivalence class, e.g., the sorted set of terms.
     
