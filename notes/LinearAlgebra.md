# Notations
## Point
- Points are represented via $$()$$
	- e.g. $$p_1 = (x_1, y_1) $$ (2D), $$p_2 = (x_2, y_2, z_2) $$

## Vector
- Scalar: $$ s \in \Re $$
- Vectors are represented via $$<>$$ as a row vector or in column form $$[]$$ as a column vector with name $$\vec v$$ 
	- $$ \vec v \in \Re^n $$ i.e. it has n elements
	- Each element in $$\vec v$$ is called a component, an entry, or a coordinate, and it's denoted as $$v_i$$ for ith element where 0 < i <= n
	- e.g. $$ \vec v = <1, -2, 3> \equiv \vec v = i - 2j + 3k \equiv \vec v = \hat v - 2\hat j + 3\hat k $$. In this case $$v_1 = 1, v_2 = -2$$
- Unit vectors are represented as $$\hat v $$
- Length would be denoted as $$|\vec v|$$

## Matrix
- $$ A \in \Re^{m*n} $$ where m is the number of rows, and n is the number of columns
- Each element: $$ A_{i,j} $$, ith row: $$ A_{i,:} $$, jth column: $$ A_{:,j} $$

## Tensor
- Tensor: array with more than 2 axes (3D comparing to matrix which is 2-D)
	- To get specific element: $$ A_{i,j,k} $$



# Vector
## Basics
- Each entry / component $$v_i$$ in $$\vec v$$ is a real number scalar i.e. $$v_i \in \Re $$
- $$ x_S $$ where S is {1,3,6} represents {$$ x_1, x_3, x_6 $$}

### Length


## Operations
### Vector addition
- $$\vec v + \vec u = <v_1 + u_1, v_2 + u_2, ...> $$

### Scalar Multiply
- $$c\vec v = <cv_1, cv_2, ...>$$
- $$c\vec v$$ is a **scalar multiple** of $$\vec v$$
- $$c\vec v$$ and $$\vec v$$ are said to be **colinear**

### Dot product
- $$\vec v \cdot \vec u = v_1 * u_1 + v_2 * u_2 + ... + v_n * u_n = |\vec v||\vec u|\cos(\theta) $$


### Cross Product
- Cross product would result in a vector which is orthogonal to the 2 vectors
- $$\vec v \times \vec u = $$ 
	1. $$ |\vec v||\vec u|\sin(\theta)\hat{n} $$ where $$\hat{n}$$ represents the normal vector to the plane constructed by u and v.
	1. $$ 	\begin{pmatrix}
  				u_2v_3 - u_3v_2\\ 
  				u_3v_1 - u_1v_3\\ 
  				u_1v_2 - u_2v_1\\ 
  			\end{pmatrix} $$
- Unit vector cross product: $$\hat i \times \hat j = 1 * 1 * \sin(90) * \hat{k} = \hat{k} $$
	- $$ \hat i \times \hat i = 0 $$
- Cross product is not commutative, and it follows right-hand rule: cross product in counter clockwise direction would result in positive direction.
	- e.g. $$ \hat i \times \hat j = \hat k $$, $$\hat j \times \hat i = -\hat k $$
- $$\tan{\theta} = \frac{|\vec u \times \vec v|}{\vec v \cdot \vec u} $$


## Linear Combinations
- A linear combinations of vectors is a sum of scalar multiples of vectors.

### Linearly Independence
- Definition
	- The vectors v1, v2,...., vk are said to be linearly independent if the only way that $$a_1\vec v_1 + a_2\vec v_2 + ... a_k\vec v_k = 0 $$ can hold true is if 􏱘a1, a􏱘2, ..., a􏱘k are all zeroes.
- Characteristics
	1. A set of vectors are linearly independent iff no vector in M is a linear combination of others
	1. If $$\vec v $$ is non-zero, then {$$\vec v$$} is linearly independent

## Orthogonal (Perpendicular)
- 2 vectors are **orthogonal** if the dot product is 0 i.e. $$ \vec x \cdot \vec y = \vec x^T * \vec y = 0 $$
	- 2 vectors are **orthonormal** if they are orthogonal and unit vectors

## Projection
- Projection of a on b = $$ |\vec a| * cos(\theta) = \frac{\vec a\cdot \vec b}{|\vec b|} $$

### Height (distance)
- Distance from point a to $$\vec b = |\vec a| * sin(\theta) = \frac{|\vec v \times \vec u|}{|\vec b|} $$ 
	- Area of triangle bounded by points a, b, c $$ = \frac{|\vec{ab} \times \vec{ac}|}{|\vec{ac}|} * \vec{ac} / 2 = |\vec{ab} \times \vec{ac}| / 2 $$
	- Area of parallelogram bounded by points a, b, c, d $$ = |\vec{ab} \times \vec{ac}| $$

## Line
- A set of all vectors $$\in \Re^3$$ in form $$\vec v = \vec a + t\vec d$$ is referred to a line L containing vector $$\vec a$$ and is parallel to $$\vec d$$
	- $$\vec v = \vec a + t\vec d$$ is the **vector equation** of line L
	- $$\vec d$$ is called the **direction vector** of line L
	- Line L will pass through point a and point (a + d)
	- Parametric equation of a line:
		- $$ \begin{aligned} 
			v_1 &= a_1 + t * d_1 \\
			v_2 &= a_2 + t * d_2 \\
			v_3 &= a_3 + t * d_3 \\
		  \end{aligned} $$
	- Cartesian equation of a line: 
		- $$ t = \frac{v_1 - a_1}{d_1} = \frac{v_2 - a_2}{d_2} = \frac{v_3 - a_3}{d_3} $$
- To find angle between 2 lines == angle between 2 direction vectors. Thus we can use dot product equations to get the angle 
- 2 lines are parallel if $$\vec d_1 $$ is a scalar multiple of $$\vec d_2 $$

## Plane
- Vector equations of a plane
	1. $$ \vec z = \vec b + s\vec v + t\vec w $$ where $$\vec b$$ is a non-zero position vector, v and w are linearly independent
	1. $$\vec n \cdot (\vec r - \vec a) = 0 $$ where n is normal, a is a specific point on plane (vector), and r is referring to any vectors on the plane (as dot product of orthogonal vectors is 0)
		- or $$\vec n \cdot vec r = d $$
			- If $$\vec n$$ is a unit vector, then d is the distance from the plane to origin
- Cartesian equation
	- $$\begin{aligned} 
	&\vec n \cdot (\vec r - \vec a) = 0 \rarr \\ 
	&n_1 * (x - a_1) + n_2 * (y - a_2) + n_3 * (z - a_3) = 0 \rarr \\ 
	&n_1x + n_2y + n_3z = d  \\
	\end{aligned} $$

### Intersection between a line and a plane
- Substitude the line's cartesian equation into the cartesian equation of a plane
	- $$ n_1 * (x - a_1) + n_2 * (y - a_2) + n_3 * (z - a_3) = 0  $$

### Angle of intersection between 2 planes
- Equals to the angle between the normals of the 2 planes
	- $$\theta = cos^{-1}(\frac{\vec n_1 \cdot \vec n_2}{|\vec n_1||\vec n_2|}) $$

### Hyperplane

## Unit vector
- unit vector: vector with unit norms i.e. $$ ||x|| = 1  $$




# Matrix

## Overview
- $$A \in \Re^{m*n} $$
- Main diagonal
	- The diagonal line
- Transpose
	- The mirror image of the matrix across a diagonal line
	- $$ (A_{ij})^T = A_{ji} $$
- Matrix + Vector: $$ C_{i,j} = A_{i,j} + b_j $$ (implicit copying of b to many locations is called broadcasting)


## Special Matrices
- Identity
- Inverse: should not be calculated explicitly in practice
- Diagonal
- Symmetric
- Orthogonal matrix: $$ A^{-1} = A^T $$

## Operations
### Element-wise product (Hadamard product)
- element-wise product of 2 matrices
- $$ C_{ij} = A_{ij} * A_{ij} $$
- Denoted as $$ A \odot B $$

### Matrices Multiplication
- $$ A \in \Re^{m \times n}, B \in \Re^{n \times p}, C = AB \in \Re^{m \times p} $$ 
	- complexity O(mnp)
	- Each entry of result C[i][j] is the dot product of ith row in A and jth column in B: $$ C_{ij} = A_{i,:} \cdot B_{:,j} $$
- Properties
	1. Distributive: A(B+C) == AB + AC
	1. Associative: A(BC) == (AB)C
	1. NOT commutative: AB != BA  
- Strassen's alg. $$ O(n^{2.89}) $$


## Matrix Inverse
- A is invertable if and only if $$ A \in \Re^{n*n} $$ and $$ A^{-1}*A = A * A^{-1} = I $$
- Thm (Sherman-Morrisen-Woodbury): $$ A\in\Re^{n*n}, B\in\Re^{m*m}, U\in\Re^{n*m}, V\in\Re^{n*m}, (A + uBV)^{-1} = A^{-1} - A^{-1}U(B^{-1}+V^TA^{-1}U)^{-1}V^TA^{-1} $$
	- In particular, $$ (A + \vec u \vec v)^{-1} = A^{-1} - \frac{1}{1+v^TA^{-1}u}A^{-1}uv^TA^{-1} $$
	- Proof: verify that the product is 1

## Determinant

## System of linear equations
- Format: $$ Ax = \vec b $$ where $$ A \in \Re^{m*n}, x \in \Re^{n}, b \in \Re^{m} $$
- Solution approaches
	1. Inverse: $$ x = A^{-1}\vec b $$
- Pivot
- Free variable
- **rank** of a matrix: maximum number of linearly independent column vectors

### RREF

### Gaussian Elimination 

### Linear dependence and Span
- Ax = b has a solution iff b is in the **column space** of A (span of A's column vectors)
- **Linearly independent** no vector in the set is a linear combination of other vectors

### Goal: solve $$ A\vec x = \vec b, A \in \Re^{n*n} $$, A is invertible
- $$ \vec x = A^{-1}\vec b $$ is non-trivial for computer implementation
- e.g. $$ 	\begin{cases} 
				2x + y + 2z = 1 \\
				4x + 2y -z = 2 \\
				x + y + 2z = 3
			\end{cases} $$
- to solve a linear system, try to transform the matrix 
	$$ \begin{bmatrix}
		2 & 1 & 2 \\ 
		4 & 2 & -1 \\
		1 & 1 & 2
	\end{bmatrix} $$ to an upper triangular matrix
- Psedo code for solving lower triangular matrix $$ O(n^2) $$
	- `for n = 1, 2, ... n do`  
			$$ y_i = (b_i - l^T_{i,1:(i-1)}y_{1:(i-1)})/l_{ii}; $$
	- **Problem** division of $$ l_{i,i} $$ could be 0. 
		- solution: add a small number to all diagonals to avoid such issues / pivoting
- During implementation, try to use vector instead of for-loop for calculation (avoid loop as much as possible)
- Gaussian Elimination psedo code
	- Input: $$ A \in \Re^{n*n}, \vec b \in \Re^{n*1} $$
	- output: in-place transform A to upper triangular matrix
	- code:
		```
		for j = 1,2 ... n-1 do
			for i = j + 1, ..., n do
		```    
			$$	a_{i,j} <- a_{i,j} / a_{j,j} $$
			$$	\vec a_{i,j+1:n} <- \vec a_{i,j+1:n} - a_{ij} $$
	- still have the problem of division
		- how to guarantee that the diagnals are never zero? 
		- solution: compute the **determinent**: $$ det(A) = \sum_{\sigma}(-1)^{sgn(\sigma)} * \prod^{n}_{i=1} A_{i, \sigma(i)} $$

### Thm (LU factorization)
- let $$\widetilde{L}$$ and u be the strict lower triangular and upper triangular part of the output of GE, then A = L*u, $$ L = \widetilde{L} + I $$

### To solve linear system Ax = b
- note: don't use inverse ever!
- $$ \begin{aligned}
		A\vec x = \vec b, A &= LU \\
 		LU\vec x &= \vec b \\
		L\vec y &= \vec b, U\vec x = \vec y
\end{aligned} $$
- Steps:
	- run GE on A and b
	- forward solve to get y from Ly = b
	- backward solve to get x from Ux = y
- time complexity is $$ O(n^3) $$


## Eigen Decomposition
- $$ Av = \lambda v $$ where v is an eigenvector and $$\lambda $$ is an eigenvalue
- **Characteristic equation** $$det(A - \lambda I) = 0 $$
	- $$\lambda $$ is an eigenvalue of $$A$$ iff it is a root of characteristic equation
		- proof:
			- if $$ \lambda $$ is an eigenvalue, then $$ (\lambda I - A)u = 0 $$ has a non-trivial solution
			- so $$ \lambda I - A $$ is singular -> $$ det(A - \lambda I) = 0 $$ -> $$ \lambda $$ is a root
- Upon similarity transform, eigenvalues do not change
- If $$ A \in \Re^{n*n} $$ is a square matrix with rank n, then $$ A = V \Lambda V^{-1} $$, where $$V_{:,i}$$ is ith eigenvector, and $$ \Lambda_{i,i} $$ is ith eigenvalue


## Singular Value Decomposition






# References
- MATH 136, University of Waterloo
- CS 370, University of Waterloo
- CS 475, Univeristy of Waterloo
- https://www.deeplearningbook.org/contents/linear_algebra.html
