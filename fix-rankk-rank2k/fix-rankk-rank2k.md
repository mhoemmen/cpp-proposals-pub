
---
title: "Fix C++26 by making the rank-1, rank-2, rank-k, and rank-2k updates consistent with the BLAS"
document: P3371R2
date: today
audience: LEWG
author:
  - name: Mark Hoemmen
    email: <mhoemmen@nvidia.com>
  - name: Ilya Burylov
    email: <burylov@gmail.com>
toc: true
---

# Authors

* Mark Hoemmen (mhoemmen@nvidia.com) (NVIDIA)
* Ilya Burylov (burylov@gmail.com) (NVIDIA)

# Revision history

* Revision 0 to be submitted 2024-08-15

* Revision 1 to be submitted 2024-09-15

    * Main wording changes from R0:

        * Instead of just changing the symmetric and Hermitian rank-k and rank-2k updates to have both overwriting and updating overloads, change *all* the update functions in this way: rank-1 (general, symmetric, and Hermitian), rank-2, rank-k, and rank-2k

        * For all the new updating overloads, specify that `C` (or `A`) may alias `E`

        * For the new symmetric and Hermitian updating overloads, specify that the functions access the new `E` parameter in the same way (e.g., with respect to the lower or upper triangle) as the `C` (or `A`) parameter

        * Add exposition-only concept _`noncomplex`_ to constrain a scaling factor to be noncomplex, as needed for Hermitian rank-1 and rank-k functions

    * Add Ilya Burylov as coauthor

    * Change title and abstract to express the wording changes

    * Add nonwording section explaining why we change rank-1 and rank-2 updates to be consistent with rank-k and rank-2k updates

    * Add nonwording sections explaining why we don't change `hermitian_matrix_vector_product`, `hermitian_matrix_product`, `triangular_matrix_product`, or the triangular solves

    * Add nonwording section explaining why we constrain some scaling factors to be noncomplex at compile time, instead of taking a run-time approach

    * Reorganize and expand nonwording sections

* Revision 2 to be submitted 2024-10-15

    * For Hermitian matrix rank-1 and rank-k updates, do not constrain the type of the scaling factor `alpha`, as R1 did.  Instead, define the algorithms to use _`real-if-needed`_`(alpha)`.  Remove exposition-only concept _`noncomplex`_.

# Abstract

We propose the following changes to [linalg] that improve consistency of the rank-1, rank-2, rank-k, and rank-2k update functions with the BLAS.

1. Add "updating" overloads to all the rank-1, rank-2, rank-k, and rank-2k update functions: general, symmetric, and Hermitian.  The new overloads are analogous to the updating overloads of `matrix_product`.  For example, `symmetric_matrix_rank_k_update(A, scaled(beta, C), C, upper_triangle)` will perform $C := \beta C + A A^T$.  This makes the functions consistent with the BLAS's behavior for nonzero `beta`, and also more consistent with the behavior of `matrix_product` (of which they are mathematically a special case).

2. Change the behavior of all the existing rank-1, rank-2, rank-k, and rank-2k update functions (general, symmetric, and Hermitian) to be "overwriting" instead of "unconditionally updating."  For example, `symmetric_matrix_rank_k_update(A, C, upper_triangle)` will perform $C = A A^T$ instead of $C := C + A A^T$.  This makes them consistent with the BLAS's behavior when `beta` is zero.

3. For the overloads of `hermitian_rank_1_update` and `hermitian_rank_k_update` that have an `alpha` scaling factor parameter, only use _`real-if-needed`_`(alpha)` in the update.  This ensures that the update will be mathematically Hermitian, and makes the behavior well defined if `alpha` has nonzero imaginary part.  The change is also consistent with our proposed resolution for LWG 4136 ("Specify behavior of [linalg] Hermitian algorithms on diagonal with nonzero imaginary part").

Items (2) and (3) are breaking changes to the current Working Draft.  Thus, we must finish this before finalization of C++26.

# Discussion of proposed changes

## Support both overwriting and updating rank-k and rank-2k updates

1. For rank-k and rank-2k updates (general, symmetric, and Hermitian), BLAS routines support both overwriting and updating behavior by exposing a scaling factor `beta`.  The corresponding [linalg] algorithms currently do not expose the equivalent functionality.  Instead, they are unconditionally updating, as if `beta` is one.

2. The rank-k and rank-2k updates are special cases of `matrix_product`, but as a result of (1), their behavior is not consistent with `matrix_product`.

3. Therefore, we need to add updating overloads of the rank-k and rank-2k updates, and change the existing overloads to be overwriting.

4. The change to existing overloads is a breaking change and thus must be finished before C++26.

5. To simplify wording, we add new exposition-only concepts _`possibly-packed-in-matrix`_ and _`possibly-packed-out-matrix`_ for symmetric and Hermitian matrix update algorithms.

### For rank-k and rank-2k updates, BLAS supports scaling factor beta, while std::linalg currently does not

Each function in any section whose label begins with "linalg.algs" generally corresponds to one or more routines or functions in the original BLAS (Basic Linear Algebra Subroutines).  Every computation that the BLAS can do, a function in the C++ Standard Library should be able to do.

One `std::linalg` user <a href="https://github.com/kokkos/stdBLAS/issues/272#issuecomment-2248273146">reported</a> an exception to this rule.  The BLAS routine `DSYRK` (Double-precision SYmmetric Rank-K update) computes $C := \beta C + \alpha A A^T$, but the corresponding `std::linalg` function `symmetric_matrix_rank_k_update` only computes $C := C + \alpha A A^T$.  That is, `std::linalg` currently has no way to express this BLAS operation with a general $\beta$ scaling factor.  This issue applies to all of the symmetric and Hermitian rank-k and rank-2k update functions.

* `symmetric_matrix_rank_k_update`: computes $C := C + \alpha A A^T$ 
* `hermitian_matrix_rank_k_update`: computes $C := C + \alpha A A^H$
* `symmetric_matrix_rank_2k_update`: computes $C := C + \alpha A B^H + \alpha B A^H$
* `hermitian_matrix_rank_2k_update`: computes $C := C + \alpha A B^H + \bar{\alpha} B A^H$, where $\bar{\alpha}$ denotes the complex conjugate of $\alpha$

### Make these functions consistent with general matrix product

These functions implement special cases of matrix-matrix products.  The `matrix_product` function in `std::linalg` implements the general case of matrix-matrix products.  This function corresponds to the BLAS's `SGEMM`, `DGEMM`, `CGEMM`, and `ZGEMM`, which compute $C := \beta C + \alpha A B$, where $\alpha$ and $\beta$ are scaling factors.  The `matrix_product` function has two kinds of overloads:

1. *overwriting* ($C = A B$) and

2. *updating* ($C = E + A B$).

The updating overloads handle the general $\alpha$ and $\beta$ case by `matrix_product(scaled(alpha, A), B, scaled(beta, C), C)`.  The specification explicitly permits the input `scaled(beta, C)` to alias the output `C` (**[linalg.algs.blas3.gemm]** 10: "*Remarks*: `C` may alias `E`").  The `std::linalg` library provides overwriting and updating overloads so that it can do everything that the BLAS does, just in a more idiomatically C++ way.  Please see <a href="https://isocpp.org/files/papers/P1673R13.html#function-argument-aliasing-and-zero-scalar-multipliers">P1673R13 Section 10.3</a> ("Function argument aliasing and zero scalar multipliers") for a more detailed explanation.

### Fix requires changing behavior of existing overloads

The problem with the current symmetric and Hermitian rank-k and rank-2k functions is that they have the same _calling syntax_ as the overwriting version of `matrix_product`, but _semantics_ that differ from both the overwriting and the updating versions of `matrix_product`.  For example,
```c++
hermitian_matrix_rank_k_update(alpha, A, C);
```
updates $C$ with $C + \alpha A A^H$, but
```c++
matrix_product(scaled(alpha, A), conjugate_transposed(A), C);
```
overwrites $C$ with $\alpha A A^H$.  The current rank-k and rank-2k overloads are not overwriting, so we can't just fix this problem by introducing an "updating" overload for each function.  

Incidentally, the fact that these functions have "update" in their name is not relevant, because that naming choice is original to the BLAS.  The BLAS calls its corresponding `xSYRK`, `xHERK`, `xSYR2K`, and `xHER2K` routines "{Symmetric, Hermitian} rank {one, two} update," even though setting $\beta = 0$ makes these routines "overwriting" in the sense of `std::linalg`.

### Add new updating overloads; make existing ones overwriting

We propose to fix this by making the functions work just like `matrix_vector_product` or `matrix_product`.  This entails three changes.

1. Add two new exposition-only concepts _`possibly-packed-in-matrix`_ and _`possibly-packed-out-matrix`_ for constraining input and output parameters of the changed or new symmetric and Hermitian update functions.

2. Add "updating" overloads of the symmetric and Hermitian rank-k and rank-2k update functions.

    a. The updating overloads take a new input matrix parameter `E`, analogous to the updating overloads of `matrix_product`, and make `C` an output parameter instead of an in/out parameter.  For example, `symmetric_matrix_rank_k_update(A, E, C, upper_triangle)` computes $C = E + A A^T$.
    
    b. Explicitly permit `C` and `E` to alias, thus permitting the desired case where `E` is `scaled(beta, C)`.
    
    c. The updating overloads take `E` as a _`possibly-packed-in-matrix`_, and take `C` as a _`possibly-packed-out-matrix`_ (instead of a _`possibly-packed-inout-matrix`_).
    
    d. `E` must be accessed as a symmetric or Hermitian matrix (depending on the function name) and such accesses must use the same triangle as `C`.  (The existing [linalg.general] 4 wording for symmetric and Hermitian behavior does not cover `E`.)

3. Change the behavior of the existing symmetric and Hermitian rank-k and rank-2k overloads to be overwriting instead of updating.

    a. For example, `symmetric_matrix_rank_k_update(A, C, upper_triangle)` will compute $C = A A^T$ instead of $C := C + A A^T$.
    
    b. Change `C` from a _`possibly-packed-inout-matrix`_ to a _`possibly-packed-out-matrix`_.

Items (2) and (3) are breaking changes to the current Working Draft.  This needs to be so that we can provide the overwriting behavior $C := \alpha A A^T$ or $C := \alpha A A^H$ that the corresponding BLAS routines already provide.  Thus, we must finish this before finalization of C++26.

Both sets of overloads still only write to the specified triangle (lower or upper) of the output matrix `C`.  As a result, the new updating overloads only read from that triangle of the input matrix `E`.  Therefore, even though `E` may be a different matrix than `C`, the updating overloads do not need an additional `Triangle t_E` parameter for `E`.  The `symmetric_*` functions interpret `E` as symmetric in the same way that they interpret `C` as symmetric, and the `hermitian_*` functions interpret `E` as Hermitian in the same way that they interpret `C` as Hermitian.  Nevertheless, we do need new wording to explain how the functions may interpret and access `E`.

## Change rank-1 and rank-2 updates to be consistent with rank-k and rank-2k

1. Currently, the rank-1 and rank-2 updates unconditionally update and do not take a $\beta$ scaling factor.  This behavior deviates from the BLAS Standard and is inconsistent with the rank-k and rank-2k updates.

2. We propose making all the rank-1 and rank-2 update functions consistent with the proposed change to the rank-k and rank-2k updates.  This means both changing the meaning of the current overloads to be overwriting, and adding new overloads that are updating.  This includes general (nonsymmetric), symmetric, and Hermitian rank-1 update functions, as well as symmetric and Hermitian rank-2 update functions.

3. The exposition-only concept _`possibly-packed-inout-matrix`_ is no longer needed.  We propose removing it.

### Current std::linalg behavior

The rank-k and rank-2k update functions have the following rank-1 and rank-2 analogs, where $A$ denotes a symmetric or Hermitian matrix (depending on the function's name) and $x$ and $y$ denote vectors.

* `symmetric_matrix_rank_1_update`: computes $A := A + \alpha x x^T$
* `hermetian_matrix_rank_1_update`: computes $A := A + \alpha x x^H$
* `symmetric_matrix_rank_2_update`: computes $A := A + \alpha x y^T + \alpha y x^T$
* `hermitian_matrix_rank_2_update`: computes $A := A + \alpha x y^H + \bar{\alpha} x y^H$

These functions *unconditionally* update the matrix $A$.  They do not have an overwriting option.  In this, they follow the "general" (not necessarily symmetric or Hermitian) rank-1 update functions.

* `matrix_rank_1_update`: computes $A := A + x y^T$
* `matrix_rank_1_update_c`: computes $A := A + x y^H$

### Current behavior is inconsistent with BLAS Standard and rank-k and rank-2k updates

These six rank-1 and rank-2 update functions map to BLAS routines as follows.

* `matrix_rank_1_update`: `xGER`
* `matrix_rank_1_update`: `xGERC`
* `symmetric_matrix_rank_1_update`: `xSYR`, `xSPR`
* `hermitian_matrix_rank_1_update`: `xHER`, `xHPR`
* `hermitian_matrix_rank_1_update`: `xSYR2`, `xSPR2`
* `hermitian_matrix_rank_1_update`: `xHER2`, `xHPR2`

The Reference BLAS and the BLAS Standard (see Chapter 2, pp. 64 - 68) differ here.  The Reference BLAS and the original 1988 BLAS 2 paper specify all of the rank-1 and rank-2 update routines listed above as unconditionally updating, and not taking a $\beta$ scaling factor.  However, the (2002) BLAS Standard specifies all of these rank-1 and rank-2 update functions as taking a $\beta$ scaling factor.  We consider the latter to express our design intent.  It is also consistent with the corresponding rank-k and rank-2k update functions in the BLAS, which all take a $\beta$ scaling factor and thus can do either overwriting (with zero $\beta$) or updating (with nonzero $\beta$).  These routines include `xSYRK`, `xHERK`, `xSYR2K`, and `xHER2K`.  One could also include the general matrix-matrix product `xGEMM` among these, as `xGEMM` also takes a $\beta$ scaling factor.

### This change would remove a special case in std::linalg's design

<a href="https://isocpp.org/files/papers/P1673R13.html#function-argument-aliasing-and-zero-scalar-multipliers">Section 10.3 of P1673R13</a> explains the three ways that the std::linalg design translates Fortran `INTENT(INOUT)` arguments into a C++ idiom.

1. Provide in-place and not-in-place overloads for triangular solve and triangular multiply.

2. "Else, if the BLAS function unconditionally updates (like `xGER`), we retain read-and-write behavior for that argument."

3. "Else, if the BLAS function uses a scalar beta argument to decide whether to read the output argument as well as write to it (like `xGEMM`), we provide two versions: a write-only version (as if `beta` is zero), and a read-and-write version (as if `beta` is nonzero)."

Our design goal was for functions with vector or matrix output to imitate `std::transform` as much as possible.  This favors Way (3) as the default approach, which turns `INTENT(INOUT)` arguments on the Fortran side into separate input and output parameters on the C++ side.  Way (2) is really an awkward special case.  The BLAS Standard effectively eliminates this special case on the Fortran side by making the rank-1 and rank-2 updates work just like the rank-k and rank-2k updates, with a $\beta$ scaling factor.  This makes it natural to eliminate the Way (2) special case on the C++ side as well.

### Exposition-only concept no longer needed

These changes make the exposition-only concept _`possibly-packed-inout-matrix`_ superfluous.  We propose removing it.

Note that this would not eliminate all uses of the exposition-only concept _`inout-matrix`_.  The in-place triangular matrix product functions `triangular_matrix_left_product` and `triangular_matrix_right_product`, and the in-place triangular linear system solve functions `triangular_matrix_matrix_left_solve` and `triangular_matrix_matrix_right_solve` would still use _`inout-matrix`_.

## Use only the real part of scaling factor `alpha` for Hermitian matrix rank-1 and rank-k updates

For Hermitian rank-1 and rank-k matrix updates, if users provide a scaling factor `alpha`, it must have zero imaginary part.  Otherwise, the matrix update will not be Hermitian, because all elements on the diagonal of a Hermitian matrix must have nonzero imaginary part.  Even though $A A^H$ is mathematically always Hermitian, if $\alpha$ has nonzero imaginary part, then $\alpha A A^H$ may no longer be a Hermitian matrix.  For example, if $A$ is the identity matrix (with ones on the diagonal and zeros elsewhere) and $\alpha = i$ (the imaginary unit, which is the square root of negative one), then $\alpha A A^H$ is the diagonal matrix whose diagonal elements are all $i$, and thus has nonzero imaginary part.

The specification of `hermitian_matrix_rank_1_update` and `hermitian_matrix_rank_k_update` does not currently require that `alpha` have nonzero imaginary part.  We propose fixing this by making these update algorithms only use the real part of `alpha`, as in _`real-if-needed`_`(alpha)`.  This solution is consistent with our proposed resolution of <a href="https://cplusplus.github.io/LWG/issue4136">LWG Issue 4136</a>, "Specify behavior of [linalg] Hermitian algorithms on diagonal with nonzero imaginary part," where we make Hermitian rank-1 and rank-k matrix updates use only the real part of matrices' diagonals.

We begin with a summary of all the Hermitian matrix BLAS routines, how scaling factors influence their mathematical correctness.  Then, we explain how these scaling factor concerns translate into [linalg] function concerns.  Finally, we discuss alternative solutions.

### Survey of scaling factors in Hermitian matrix BLAS routines

The BLAS's Hermitian matrix routines take `alpha` and `beta` scaling factors.  The BLAS addresses the resulting correctness concerns in different ways, depending on what each routine computes.  For routines where a nonzero imaginary part could make the result incorrect, the routine restricts the scaling factor to have a noncomplex number type.  Otherwise, the routine takes the scaling factor as a complex number type.  We discuss all the Hermitian routines here.

#### `HEMM`: Hermitian matrix-matrix multiply

`HEMM` (HErmitian Matrix-matrix Multiply) computes either $C := \alpha A B + \beta C$ or $C := \alpha B A + \beta C$, where $A$ is a Hermitian matrix, and neither $B$ nor $C$ need to be Hermitian.  The products $A B$ and $B A$ thus need not be Hermitian, so the scaling factors $\alpha$ and $\beta$ can have nonzero imaginary parts.  The BLAS takes them both as complex numbers.

#### `HEMV`: HErmitian Matrix-Vector multiply

`HEMV` (HErmitian Matrix-Vector multiply) computes $y := \alpha A x + \beta y$, where $A$ is a Hermitian matrix and $x$ and $y$ are vectors.  The scaled matrix $\alpha A$ does not need to be Hermitian.  Thus, $\alpha$ and $\beta$ can have nonzero imaginary parts.  The BLAS takes them both as complex numbers.

#### `HER`: HErmitian Rank-1 update

`HER` (HErmitian Rank-1 update) differs between the Reference BLAS (which computes $A := \alpha x x^H + A$) and the BLAS Standard (which computes $A := \alpha x x^H + \beta A$).  The matrix $A$ must be Hermitian, and the rank-1 matrix $x x^H$ is always mathematically Hermitian, so both $\alpha$ and $\beta$ need to have zero imaginary part in order for the update to preserve $A$'s Hermitian property.  The BLAS takes them both as real (noncomplex) numbers.

#### `HER2`: HErmitian Rank-2 update

`HER2` (HErmitian Rank-2 update) differs between the Reference BLAS (which computes $A := \alpha x y^H + \bar{\alpha} y x^H + A$, where $\bar{\alpha}$ denotes the complex conjugate of $\alpha$) and the BLAS Standard (which computes $A := \alpha x y^H + \bar{\alpha} y x^H + \beta A$).  The matrix $A$ must be Hermitian, and the rank-2 matrix $\alpha x y^H + \bar{\alpha} y x^H$ is always mathematically Hermitian, no matter the value of $\alpha$.  Thus, $\alpha$ can have nonzero imaginary part, but $\beta$ cannot.  The BLAS thus takes `alpha` as a complex number, but `beta` as a real (noncomplex) number.  (There is likely a typo in the BLAS Standard's description of the Fortran 95 binding.  It says that both `alpha` and `beta` are complex (have type `COMPLEX(<wp>)`), even though in the Fortran 77 binding, `beta` is real (`<rtype>`).  The BLAS Standard's description of `HER2K` (see below) says that `alpha` is complex but `beta` is real.  `HER2` needs to be consistent with `HER2K`.)

#### `HERK`: HErmitian Rank-K update

`HERK` (HErmitian Rank-K update) computes either $C := \alpha A A^H + \beta C$ or $C := \alpha A^H A + \beta C$, where $C$ must be Hermitian.  This is a generalization of `HER` and thus both $\alpha$ and $\beta$ need to have zero imaginary part.  The BLAS takes them both as real (noncomplex) numbers.

#### `HER2K`: HErmitian Rank-2k update

`HER2K` (HErmitian Rank-2k update) computes either $C := \alpha A B^H + \bar{\alpha} B A^H + \beta C$ or $C := \alpha A^H B + \bar{\alpha} B^H A + \beta C$.  This is a generalization of `HER2`: $\alpha$ can have nonzero imaginary part, but $\beta$ cannot.  The BLAS thus takes `alpha` as a complex number, but `beta` as a real (noncomplex) number.

#### Summary of BLAS routine restrictions

The following table lists, for all the Hermitian matrix update BLAS routines, whether the routine restricts `alpha` and/or `beta` to have zero imaginary part, and whether the routine is a generalization of some other routine in the list (N/A, "not applicable," means that it is not).

<table>
  <tr>
    <th> BLAS routine      </th>
    <th> Restricts `alpha` </th>
    <th> Restricts `beta`  </th>
    <th> Generalizes       </th>
  </tr>
  <tr>
    <th> `HEMM`  </th>
    <th> No      </th>
    <th> No      </th>
    <th> N/A     </th>
  </tr>
  <tr>
    <th> `HER`   </th>
    <th> Yes     </th>
    <th> Yes     </th>
    <th> N/A     </th>
  </tr>
  <tr>
    <th> `HER2`  </th>
    <th> No      </th>
    <th> Yes     </th>
    <th> N/A     </th>
  </tr>
  <tr>
    <th> `HERK`  </th>
    <th> Yes     </th>
    <th> Yes     </th>
    <th> `HER`   </th>
  </tr>
  <tr>
    <th> `HER2K` </th>
    <th> No      </th>
    <th> Yes     </th>
    <th> `HER2`  </th>
  </tr>
</table>

### Translation of Hermitian BLAS concerns to std::linalg

#### Assume changes proposed in previous sections

We assume here the changes proposed in previous sections that remove inout matrix parameters from the rank-1, rank-2, rank-k, and rank-2k algorithms, and separate these algorithms into overwriting and updating overloads.  This lets us only consider input matrix and vector parameters.

#### std::linalg and the BLAS treat scaling factors differently

The [linalg] library and the BLAS treat scaling factors in different ways.  First, [linalg] treats the result of `scaled` just like any other matrix or vector parameter.  It applies any mathematical requirements (like being Hermitian) to the parameter, regardless of whether the corresponding argument results from `scaled`.  It also does not forbid any input argument from being the result of `scaled`.  Second, the BLAS always exposes `alpha` and `beta` scaling factor parameters separately from the matrix or vector parameters to which they are applied.  In contrast, [linalg] only exposes a separate `alpha` scaling factor (never `beta`) if it would otherwise be mathematically impossible to express an operation that the BLAS can express.  For example, for matrices and scaling factors that are noncomplex, `symmetric_matrix_rank_1_update` cannot express $A := A - x x^T$ with a noncomplex scaling factor (because the square root of $-1$ is $i$).

#### Defer fixing places where the BLAS can do what std::linalg cannot

In some cases, [linalg] does not expose a separate scaling factor parameter, even when this prevents [linalg] from doing some things that the BLAS can do.  We give an example below of triangular matrix solves with multiple right-hand sides and `alpha` scaling not equal to one, where the matrix has an implicit unit diagonal.

Even though this means that the BLAS can do some things that [linalg] cannot do, it does not cause [linalg] to violate mathematical consistency.  More importantly, as we show later, [linalg] can be extended to do whatever the BLAS can do without breaking backwards compatibility.  Thus, we consider the status quo acceptable for C++26.

#### Scaling factor `beta` is not a concern

The [linalg] library never exposes a scaling factor `beta`.  For BLAS routines that perform an update with `beta` times an inout matrix or vector parameter (e.g., $\beta y$ or $\beta C$), [linalg] instead takes an input matrix or vector parameter (e.g., `E`) that can be separate from the output matrix or vector (e.g., `C`).  For Hermitian BLAS routines where `beta` needs to have zero imaginary part, [linalg] simply requires that `E` be Hermitian -- a strictly more general requirement.  For example, for the new updating overloads of `hermitian_rank_1_update` and `hermitian_rank_k_update` proposed above, [linalg] expresses a `beta` scaling factor by letting users supply `scaled(beta, C)` as the argument for `E`.  The wording only requires that `E` be Hermitian.  If `E` is `scaled(beta, C)`, this concerns only the product of `beta` and `C`.  It would be incorrect to constrain `beta` or `C` separately.  For example, if $\beta = -i$ and $C$ is the matrix whose elements are all $i$, then $C$ is not Hermitian but $\beta C$ (and therefore `scaled(beta, C)`) is Hermitian.  The same reasoning applies for the rank-2 and rank-2k updates.

#### What this section proposes to fix, and the proposed solution

The above arguments help us restrict our concerns.  This section of our proposal concerns itself with Hermitian matrix update algorithms where

* the algorithm exposes a separate scaling factor parameter `alpha`, and

* `alpha` needs to have zero imaginary part, but

* nothing in the wording currently prevents `alpha` from having nonzero imaginary part.

These correspond exactly to the BLAS's Hermitian matrix update routines where the type of `alpha` is real: `HER` and `HERK`.  This strongly suggests solving the problem in [linalg] by constraining the type of `alpha` to be noncomplex.  However, as we explain in "Alternative solutions" below, it is hard to define a "noncomplex number" constraint that works well for user-defined number types.  Instead, we propose fixing this in a way that is consistent with our proposed resolution of <a href="https://cplusplus.github.io/LWG/issue4136">LWG Issue 4136</a>, "Specify behavior of [linalg] Hermitian algorithms on diagonal with nonzero imaginary part."  That is, the Hermitian rank-1 and rank-k update algorithms will simply use _`real-if-needed`_`(alpha)` and ignore any nonzero imaginary part of `alpha`.

### Alternative solutions

We can think of at least four different ways to solve this problem, and will explain why we did not choose those solutions.

1. Constrain `alpha` by defining a generic "noncomplex number type" constraint.

2. Only constrain `alpha` not to be `std::complex<T>`; do not try to define a generic "noncomplex number" constraint.

3. Constrain `alpha` by default using a generic "noncomplex number type" constraint, but let users specialize an "opt-out trait" to tell the library that their number type is noncomplex.

4. Impose a precondition that the imaginary part of `alpha` is zero.

If we had to pick one of these solutions, we would favor first (4), then (3), and then (1).  We would object most to (2).

#### Alternative: Constrain `alpha` via generic "noncomplex" constraint

The BLAS solves this problem by having the Hermitian rank-1 update routines `xHER` and rank-k update routines `xHERK` take the scaling factor $\alpha$ as a noncomplex number.  This suggests constraining `alpha`'s type to be noncomplex.  However, [linalg] accepts user-defined number types, including user-defined complex number types.  How do we define a "noncomplex number type"?  If we get it wrong and say that a number type is complex when its "imaginary part" is always zero, we end up excluding a perfectly valid custom number type.

For number types that have an additive identity (like zero for the integers and reals), it's mathematically correct to treat those as "complex numbers" whose imaginary parts are always zero.  This is what `conjugated_accessor` does if you give it a number type for which ADL cannot find `conj`.  P3050 optimizes `conjugated` for such number types by bypassing `conjugated_accessor` and using `default_accessor` instead, but this is just a code optimization.  Therefore, it can afford to be conservative: if a number type _might_ be complex, then `conjugated` needs to use `conjugated_accessor`.  This is why P3050's approach doesn't apply here.  If a number type has ADL-findable `conj`, `real`, and `imag`, then it _might_ be complex.  However, if we define the opposite of that as "noncomplex," then we might be preventing users from using otherwise reasonable number types.  

The following `MyRealNumber` example always has zero imaginary part, but nevertheless has ADL-findable `conj`, `real`, and `imag`.  Furthermore, it has a constructor for which `MyRealNumber(1.2, 3.4)` is well-formed.  (This is an unfortunate design choice; making `precision` have class type that is not implicitly convertible from `double` would be wiser, so that users would have to type `MyRealNumber(1.2, Precision(42))`.)  As a result, there is no reasonable way to tell at compile time if `MyRealNumber` might represent a complex number.

```c++
class MyRealNumber {
public:
  explicit MyRealNumber(double initial_value);
  // precision represents the amount of storage
  // used by an arbitrary-precision real number object
  explicit MyRealNumber(double initial_value, int precision);

  MyRealNumber(); // result is the additive identity "zero"

  // ... other members to make MyRealNumber regular ...
  // ... hidden friend overloaded arithmetic operators ...

  friend MyRealNumber conj(MyRealNumber x) { return  x; }
  friend MyRealNumber real(MyRealNumber x) { return  x; }
  friend MyRealNumber imag(MyRealNumber)   { return {}; }
};
```

It's reasonable to write custom noncomplex number types that define ADL-findable `conj`, `real`, and `imag`.  First, users may want to write or use libraries of generic numerical algorithms that work for both complex and noncomplex number types.  P1673 argues that defining `conj` to be type-preserving (unlike `std::conj` in the C++ Standard Library) makes this possible.  For example, Appendix A below shows how to implement a generic two-norm absolute value (or magnitude, for complex numbers) function, using this interface.  Second, the Standard Library might lead users to write a type like this.  `std::conj` accepts arguments of any integer or floating-point type, none of which represent complex numbers.  The Standard makes the unfortunate choice for `std::conj` of an integer type to return `std::complex<double>`.  However, users who try to make `conj(MyRealNumber)` return `std::complex<MyRealNumber>` would find out that `std::complex<MyRealNumber>` does not compile, because `std::complex<T>` requires that `T` be a floating-point type.  The least-effort next step would be to make `conj(MyRealNumber)` return `MyRealNumber`.

We want rank-1 and rank-k Hermitian matrix updates to work with types like `MyRealNumber`, but any reasonable way to constrain the type of `alpha` would exclude `MyRealNumber`.

#### Alternative: Only constrain `alpha` not to be `std::complex`

A variant of this suggestion would be only to constrain `alpha` not to be `std::complex<T>`, and not try to define a generic "noncomplex number" constraint.  However, this would break generic numerical algorithms by making valid code for the non-Standard complex number case invalid code for `std::complex<T>`.  We do not want [linalg] to treat custom complex number types differently than `std::complex`.

#### Alternative: Constrain `alpha` by default, but let users "opt out"

Another option would be to constrain `alpha` by default using the same generic "noncomplex number type" constraint as in Option (1), but let users specialize an "opt-out trait" to tell the library that their number type is noncomplex.  We would add a new "tag" type trait `is_noncomplex` that users can specialize.  By default, `is_noncomplex<T>::value` is `false` for any type `T`.  This does _not_ mean that the type is complex, just that the user declares their type to be noncomplex.  The distinction matters, because a noncomplex number type might still provide ADL-findable `conj`, `real`, and `imag`, as we showed above.  Users must take positive action to declare their type `U` as "not a complex number type," by specializing `is_noncomplex<U>` so that `is_noncomplex<U>::value` is `true`.  If users do that, then the library will ignore any ADL-findable functions `conj`, `real`, and `imag` (whether or not they exist), and will assume that the number type is noncomplex.

Standard Library precedent for this approach is in heterogeneous lookup for associative containers (see N3657 for ordered associative containers, and P0919 and P1690 for unordered containers).  User-defined hash functions and key equality comparison functions can tell the container to provide heterogeneous comparisons by exposing a `static constexpr bool is_transparent` whose value is `true`.  Default behavior does not expose heterogeneous comparisons.  Thus, users must opt in at compile time to assert something about their user-defined types.  Another example is `uses_allocator<T, alloc>`, whose `value` member defaults to `false` unless `T` has a nested type `allocator_type` that is convertible from `Alloc`.  Standard Library types like `tuple` use `uses_allocator` to determine if a user-defined type `T` is allocator-aware.

Of the three constraint-based approaches discussed in this proposal, we favor this one the most.  It still treats types "as they are" and does not permit users to claim that a type is complex when it lacks the needed operations, but it lets users optimize by giving the Standard Library a single bit of compile-time information.  By default, any linear algebra value type (see [linalg.reqs.val]) that meets the `maybe_complex` concept below would be considered "possibly complex."  Types that do not meet this concept would result in compilation errors; users would then be able to search documentation or web pages to find out that they need to specialize `is_noncomplex`.

```c++
template<class T>
concept maybe_complex =
  std::semiregular<T> &&
  requires(T t) {
    {conj(t)} -> T;
    {real(t)} -> std::convertible_to<T>;
    {imag(t)} -> std::convertible_to<T>;
```

P1673 generally avoids approaches based on specializing traits.  Its design philosophy favors treating types as they are.  Users should not need to do something to get correct behavior.  We based this on our past experiences in generic numerical algorithms development.  In the 2010's, one of the authors maintained a generic mathematical algorithms library called Trilinos.  The Teuchos (pronounced "TEFF-os") package of Trilinos provides a monolithic `ScalarTraits` class template that defines different properties of a number type.  It combines the features of `std::numeric_limits` with generic complex arithmetic operations like `conjugate`, `real`, and `imag`.  Trilinos' generic algorithms assume that number types are regular and define overloaded `+`, `-`, `*`, and `/`, but use `ScalarTraits<T>::conjugate`, `ScalarTraits<T>::real`, and `ScalarTraits<T>::imag`.  As a result, users with a custom complex number type had to specialize `ScalarTraits` and provide all these operations.  Even if users had imitated `std::complex`'s interface perfectly and provided ADL-findable `conj`, `real`, and `imag`, users had to do extra work to make Trilinos compile and run correctly for their numbers.  With P1673, we decided instead that users who define a custom complex number type with an interface sufficiently like `std::complex` should get reasonable behavior without needing to do anything else.

As a tangent, we would like to comment on the monolithic design of `Teuchos::ScalarTraits`.  The monolithic design was partly an imitation of `std::numeric_limits`, and partly a side effect of a requirement to support pre-C++11 compilers that did not permit partial specialization of function templates.  (The typical pre-C++11 work-around is to define an unspecialized function template that dispatches to a possibly specialized class template.)  C++11 permits partial specialization of function templates and C++14 introduces variable templates; these features have encouraged "breaking up" monolithic traits classes into separate traits.  Our paper P1370R1 ("Generic numerical algorithm development with(out) `numeric_limits`") aligns with this trend.

#### Alternative: Impose precondition on `alpha`

Another option would be to impose a precondition that _`imag-if-needed`_`(alpha)` is zero.  However, this would be inconsistent with our proposed resolution of <a href="https://cplusplus.github.io/LWG/issue4136">LWG Issue 4136</a>, "Specify behavior of [linalg] Hermitian algorithms on diagonal with nonzero imaginary part".  WG21 members have expressed wanting _fewer_ preconditions and _less_ undefined behavior in the Standard Library.

If users call Hermitian matrix rank-1 or rank-k updates with `alpha` being `std::complex<float>` or `std::complex<double>`, implementations of [linalg] that call an underlying C or Fortran BLAS would have to get the real part of `alpha` anyway, because these BLAS routines only take `alpha` as a real type.  Thus, our proposed solution -- to _define_ the behavior of the update algorithms as using _`real-if-needed`_`(alpha)` -- would not add overhead.

## Things relating to scaling factors that we do not propose changing

### Hermitian matrix-vector and matrix-matrix products

We pointed out above that `hermitian_matrix_vector_product` and `hermitian_matrix_product` expect that the (possibly scaled) input matrix is Hermitian, while the corresponding BLAS routines `HEMV` and `HEMM` expect that the unscaled input matrix is Hermitian and permit the scaling factor `alpha` to have nonzero imaginary part.  However, this does not affect the ability of these [linalg] algorithms to compute what the BLAS can compute.  Users who want to supply `alpha` with nonzero imaginary part should *not* scale the matrix `A` (as in `scaled(alpha, A)`).  Instead, they should scale the input vector `x`, as in the following.
```c++
auto alpha = std::complex{0.0, 1.0};
hermitian_matrix_vector_product(A, upper_triangle, scaled(alpha, x), y);
```
Therefore, `hermitian_matrix_vector_product` and `hermitian_matrix_product` do *not* need extra overloads with a scaling factor `alpha` parameter.

#### In BLAS, matrix is Hermitian, but scaled matrix need not be

In Chapter 2 of the BLAS Standard, both `xHEMV` and `xHEMM` take the scaling factors $\alpha$ and $\beta$ as complex numbers (`COMPLEX<wp>`, where `<wp>` represents the current working precision).  The BLAS permits `xHEMV` or `xHEMM` to be called with `alpha` whose imaginary part is nonzero.  The matrix that the BLAS assumes to be Hermitian is $A$, not $\alpha A$.  Even if $A$ is Hermitian, $\alpha A$ might not necessarily be Hermitian.  For example, if $A$ is the identity matrix (diagonal all ones) and $\alpha$ is $i$, then $\alpha A$ is not Hermitian but skew-Hermitian.

The current [linalg] wording requires that the input matrix be Hermitian.  This excludes using `scaled(alpha, A)` as the input matrix, where `alpha` has nonzero imaginary part.  For example, the following gives mathematically incorrect results.
```c++
auto alpha = std::complex{0.0, 1.0};
hermitian_matrix_vector_product(scaled(alpha, A), upper_triangle, x, y);
```
Note that the behavior of this is still otherwise well defined, at least after applying the fix proposed in LWG4136 for diagonal elements with nonzero imaginary part.  It does not violate a precondition.  Therefore, the Standard has no way to tell the user that they did something wrong.

#### Status quo permits scaling via the input vector

The current wording permits introducing the scaling factor `alpha` through the input vector.
```c++
auto alpha = std::complex{0.0, 1.0};
hermitian_matrix_vector_product(A, upper_triangle, scaled(alpha, x), y);
```
This is fine as long as $\alpha A x$ equals $A \alpha x$, that is, as long as $\alpha$ commutes with the elements of A.  This issue would only be of concern if those multiplications might be noncommutative.  We're already well outside the world of "types that do ordinary arithmetic with `std::complex`."  This scenario would only arise given a user-defined complex type, like `user_complex<user_noncommutative>` in the example below, whose real parts have noncommutative multiplication.

```c++
template<class T>
class user_complex {
public:
  user_complex(T re, T im) : re_(re), im_(im) {}

  // ... overloaded arithmetic operators ...

  friend T real(user_complex<T> z) { return z.re_; }
  friend T imag(user_complex<T> z) { return z.im_; } 
  friend user_complex<T> conj(user_complex<T> z) {
    return {real(z), -imag(z)};
  }

private:
  T re_, im_;
};

auto alpha = user_complex<user_noncommutative>{something, something_else};
hermitian_matrix_vector_product(N, upper_triangle, scaled(alpha, x), y);
```

The [linalg] library was designed to support element types with noncommutative multiplication.  On the other hand, generally, if we speak of Hermitian matrices or even of inner products (which are used to define Hermitian matrices), we're working in a vector space.  This means that multiplication of the matrix's elements is commutative.  Anything more general than that is far beyond what the BLAS can do.  Thus, we think restricting use of `alpha` with nonzero imaginary part to `scaled(alpha, x)` is not so onerous.

#### Scaling via the input vector is weird, but the least bad choice

Many users may not like the status quo of needing to scale `x` instead of `A`.  First, it differs from the BLAS, which puts `alpha` before `A` in its `xHEMV` and `xHEMM` function arguments.  Second, users would get no compile-time feedback and likely no run-time feedback if they scale `A` instead of `x`.  Their code would compile and produce correct results for almost all matrix-vector or matrix-matrix products, *except* for the Hermitian case, and *only* when the scaling factor has a nonzero imaginary part.  However, we still think the status quo is the least bad choice.  We explain why by discussing the following alternatives.

1. Treat `scaled(alpha, A)` as a special case: expect `A` to be Hermitian and permit `alpha` to have nonzero imaginary part

2. Forbid `scaled(alpha, A)` at compile time, so that users must scale `x` instead

3. Add overloads that take `alpha`, analogous to the rank-1 and rank-k update functions

The first choice is mathematically incorrect, as we will explain below.  The second is not incorrect, but could only catch some errors.  The third is likewise not incorrect, but would add a lot of overloads for an uncommon use case, and would still not prevent users from scaling the matrix in mathematically incorrect ways.

##### Bad choice: Special cases for scaling the matrix

"Treat `scaled(alpha, A)` as a special case" actually means three special cases, given some nested accessor type `Acc` and a scaling factor `alpha` of type `Scalar`.

a. `scaled(alpha, A)`, whose accessor type is `scaled_accessor<Scalar, Acc>`

b. `conjugated(scaled(alpha, A))`, whose accessor type is `conjugated_accessor<scaled_accessor<Scalar, Acc>>`

c. `scaled(alpha, conjugated(A))`, whose accessor type is `scaled_accessor<Scalar, conjugated_accessor<Acc>>`

One could replace `conjugated` with `conjugate_transposed` (which we expect to be more common in practice) without changing the accessor type.

This approach violates the fundamental [linalg] principle that "... each `mdspan` parameter of a function behaves as itself and is not otherwise 'modified' by other parameters" (Section 10.2.5, P1673R13).  The behavior of [linalg] is agnostic of specific accessor types, even though implementations likely have optimizations for specific accessor types.  [linalg] takes this approach for consistency, even where it results in different behavior from the BLAS (see Section 10.5.2 of P1673R13).  The application of this principle here is "the input parameter `A` is always Hermitian."

In this case, the consistency matters for mathematical correctness.  What if `scaled(alpha, A)` is Hermitian, but `A` itself is not?  An example is $\alpha = -i$ and $A$ is the 2 x 2 matrix whose elements are all $i$.  If we treat `scaled_accessor` as a special case, then `hermitian_matrix_vector_product` will compute different results.

Another problem is that users are permitted to define their own accessor types, including nested accessors.  Arbitrary user accessors might have `scaled_accessor` somewhere in that nesting, or they might have the *effect* of `scaled_accessor` without being called that.  Thus, we can't detect all possible ways that the matrix might be scaled.

##### Not-so-good choice: Forbid scaling the matrix at compile time

Hermitian matrix-matrix and matrix-vector product functions could instead *forbid* scaling the matrix at compile time, at least for the three accessor type cases listed in the previous option.

a. `scaled_accessor<Scalar, Acc>`

b. `conjugated_accessor<scaled_accessor<Scalar, Acc>>`

c. `scaled_accessor<Scalar, conjugated_accessor<Acc>>`

Doing this would certainly encourage correct behavior for the most common cases.  However, as mentioned above, users are permitted to define their own accessor types, including nested accessors.  Thus, we can't detect all ways that an arbitrary, possibly user-defined accessor might scale the matrix.  Furthermore, scaling the matrix might still be mathematically correct.  A scaling factor with nonzero imaginary part might even *make* the matrix Hermitian.  Applying $i$ as a scaling factor twice would give a perfectly valid scaling factor of $-1$.

##### Not-so-good choice: Add alpha overloads

One could imagine adding overloads that take `alpha`, analogous to the rank-1 and rank-k update overloads that take `alpha`.  Users could then write
```c++
hermitian_matrix_vector_product(alpha, A, upper_triangle, x, y);
```
instead of
```c++
hermitian_matrix_vector_product(A, upper_triangle, scaled(alpha, x), y);
```

We do not support this approach.  First, it would introduce many overloads, without adding to what the existing overloads can accomplish.  (Users can already introduce the scaling factor `alpha` through `x`.)  Contrast this with the rank-1 and rank-k Hermitian update functions, where not having `alpha` overloads might break simple cases.  Here are two examples.

1. If the matrix and vector element types and the scaling factor $\alpha = 2$ are all integers, then the update $C := C - 2 x x^H$ can be computed using integer types and arithmetic with an `alpha` overload.  However, without an `alpha` overload, the user would need to use `scaled(sqrt(2), x)` as the input vector, thus forcing use of floating-point arithmetic that may give inexact results.

2. The update $C := C - x x^H$ would require resorting to complex arithmetic, as the only way to express $-x x^H$ with the same scaling factor for both vectors is $(i x) (i x)^H$.

Second, `alpha` overloads would not prevent users from *also* supplying `scaled(gamma, A)` as the matrix for some other scaling factor `gamma`.  Thus, instead of solving the problem, the overloads would introduce more possibilities for errors.

### Triangular matrix products with implicit unit diagonals

1. In BLAS, triangular matrix-vector and matrix-matrix products apply `alpha` scaling to the implicit unit diagonal.  In [linalg], the scaling factor `alpha` is not applied to the implicit unit diagonal.  This is because the library does not interpret `scaled(alpha, A)` differently than any other `mdspan`.

2. Users of triangular matrix-vector products can recover BLAS functionality by scaling the input vector instead of the input matrix, so this only matters for triangular matrix-matrix products.

3. All calls of the BLAS's triangular matrix-matrix product routine `xTRMM` in LAPACK (other than in testing routines) use `alpha` equal to one.

4. Straightforward approaches for fixing this issue would not break backwards compatibility.

5. Therefore, we do not consider fixing this a high-priority issue, and we do not propose a fix for it in this paper.

#### BLAS applies alpha after unit diagonal; linalg applies it before

The `triangular_matrix_vector_product` and `triangular_matrix_product` algorithms have an `implicit_unit_diagonal` option.  This makes the algorithm not access the diagonal of the matrix, and compute as if the diagonal were all ones.  The option corresponds to the BLAS's "Unit" flag.  BLAS routines that take both a "Unit" flag and an `alpha` scaling factor apply "Unit" *before* scaling by `alpha`, so that the matrix is treated as if it has a diagonal of all `alpha` values.  In contrast, [linalg] follows the general principle that `scaled(alpha, A)` should be treated like any other kind of `mdspan`.  As a result, algorithms interpret `implicit_unit_diagonal` as applied to the matrix *after* scaling by `alpha`, so that the matrix still has a diagonal of all ones.

#### Triangular solve algorithms not affected

The triangular solve algorithms in std::linalg are not affected, because their BLAS analogs either do not take an `alpha` argument (as with `xTRSV`), or the `alpha` argument does not affect the triangular matrix (with `xTRSM`, `alpha` affects the right-hand sides `B`, not the triangular matrix `A`).

#### Triangular matrix-vector product work-around

This issue only reduces functionality of `triangular_matrix_product`.  Users of `triangular_matrix_vector_product` who wish to replicate the original BLAS functionality can scale the input matrix (by supplying `scaled(alpha, x)` instead of `x` as the input argument) instead of the triangular matrix.

#### Triangular matrix-matrix product example

The following example computes $A := 2 A B$ where $A$ is a lower triangular matrix, but it makes the diagonal of $A$ all ones on the input (right-hand) side.
```c++
triangular_matrix_product(scaled(2.0, A), lower_triangle, implicit_unit_diagonal, B, A);
```
Contrast with the analogous BLAS routine `DTRMM`, which has the effect of making the diagonal elements all `2.0`.
```fortran
dtrmm('Left side', 'Lower triangular', 'No transpose', 'Unit diagonal', m, n, 2.0, A, lda, B, ldb)
```
If we want to use [linalg] to express what the BLAS call expresses, we need to perform a separate scaling.
```c++
triangular_matrix_product(A, lower_triangle, implicit_unit_diagonal, B, A);
scale(2.0, A);
```
This is counterintuitive, and may also affect performance.  Performance of `scale` is typically bound by memory bandwidth and/or latency, but if the work done by `scale` could be fused with the work done by the `triangular_matrix_product`, then `scale`'s memory operations could be "hidden" in the cost of the matrix product. 

#### LAPACK never calls `xTRMM` with the implicit unit diagonal option and `alpha` not equal to one

How much might users care about this missing [linalg] feature?  P1673R13 explains that the BLAS was codesigned with LAPACK and that every reference BLAS routine is used by some LAPACK routine.  "The BLAS does not aim to provide a complete set of mathematical operations.  Every function in the BLAS exists because some LINPACK or LAPACK algorithm needs it" (Section 10.6.1).  Therefore, to judge the urgency of adding new functionality to [linalg], we can ask whether the functionality would be needed by a C++ re-implementation of LAPACK.  We think not much, because the highest-priority target audience of the BLAS is LAPACK developers, and LAPACK routines (other than testing routines) never use a scaling factor alpha other than one.  

We survey calls to `xTRMM` in the latest version of LAPACK as of the publication date of R1 of this proposal, LAPACK 3.12.0.  It suffices to survey `DTRMM`, the double-precision real case, since for all the routines of interest, the complex case follows the same pattern.  (We did survey `ZTRMM`, the double-precision complex case, just in case.)  LAPACK has 24 routines that call `DTRMM` directly.  They fall into five categories.

1. Test routines: `DCHK3`, `DCHKE`, `DLARHS`

2. Routines relating to QR factorization or using the result of a QR factorization (especially with block Householder reflectors): `DGELQT3`, `DLARFB`, `DGEQRT3`, `DLARFB_GETT`, `DLARZB`, `DORM22`

3. Routines relating to computing an inverse of a triangular matrix or of a matrix that has been factored into triangular matrices: `DLAUUM`, `DTRITRI`, `DTFTRI`, `DPFTRI`

4. Routines relating to solving eigenvalue (or generalized eigenvalue) problems: `DLAHR2`, `DSYGST`, `DGEHRD`, `DSYGV`, `DSYGV_2STAGE`, `DSYGVD`, `DSYGVX` (note that `DLAQR5` depends on `DTRMM` via `EXTERNAL` declaration, but doesn't actually call it)

5. Routines relating to symmetric indefinite factorizations: `DSYT01_AA`, `DSYTRI2X`, `DSYTRI_3X`

The only routines that call `DTRMM` with `alpha` equal to anything other than one or negative one are the testing routines.  Some calls in `DGELQT3` and `DLARFB_GETT` use negative one, but these calls never specify an implicit unit diagonal (they use the explicit diagonal option).  The only routine that might possibly call `DTRMM` with both negative one as alpha and the implicit unit diagonal is `DTFTRI`.  (This routine "computes the inverse of a triangular matrix A stored in RFP [Rectangular Full Packed] format."  RFP format was introduced to LAPACK in the late 2000's, well after the BLAS Standard was published.  See <a href="http://www.netlib.org/lapack/lawnspdf/lawn199.pdf">LAPACK Working Note 199</a>, which was published in 2008.)  `DTFTRI` passes its caller's `diag` argument (which specifies either implicit unit diagonal or explicit diagonal) to `DTRMM`.  The only two LAPACK routines that call `DTFTRI` are `DERRRFP` (a testing routine) and `DPFTRI`.  `DPFTRI` only ever calls `DTFTRI` with `diag` *not* specifying the implicit unit diagonal option.  Therefore, LAPACK never needs both `alpha` not equal to one and the implicit unit diagonal option, so adding the ability to "scale the implicit diagonal" in [linalg] is a low-priority feature.

#### Fixes would not break backwards compatibility

We can think of two ways to fix this issue.  First, we could add an `alpha` scaling parameter, analogous to the symmetric and Hermitian rank-1 and rank-k update functions.  Second, we could add a new kind of `Diagonal` template parameter type that expresses a "diagonal value."  For example, `implicit_diagonal_t{alpha}` (or a function form, `implicit_diagonal(alpha)`) would tell the algorithm not to access the diagonal elements, but instead to assume that their value is `alpha`.  Both of these solutions would let users specify the diagonal's scaling factor separately from the scaling factor for the rest of the matrix.  Those two scaling factors could differ, which is new functionality not offered by the BLAS.  More importantly, both of these solutions could be added later, after C++26, without breaking backwards compatibility.

### Triangular solves with implicit unit diagonals

1. In BLAS, triangular solves with possibly multiple right-hand sides (`xTRSM`) apply `alpha` scaling to the implicit unit diagonal.  In [linalg], the scaling factor `alpha` is not applied to the implicit unit diagonal.  This is because the library does not interpret `scaled(alpha, A)` differently than any other `mdspan`.

2. Users of triangular solves would need a separate `scale` call to recover BLAS functionality.

3. LAPACK sometimes calls `xTRSM` with `alpha` not equal to one.

4. Straightforward approaches for fixing this issue would not break backwards compatibility.

5. Therefore, we do not consider fixing this a high-priority issue, and we do not propose a fix for it in this paper.

#### BLAS applies alpha after unit diagonal; linalg applies it before

Triangular solves have a similar issue to the one explained in the previous section.  The BLAS routine `xTRSM` applies `alpha` "after" the implicit unit diagonal, while std::linalg applies `alpha` "before."  (`xTRSV` does not take an `alpha` scaling factor.)  As a result, the BLAS solves with a different matrix than std::linalg.

In mathematical terms, `xTRSM` solves the equation $\alpha (A + I) X = B$ for $X$, where $A$ is the user's input matrix (without implicit unit diagonal) and $I$ is the identity matrix (with ones on the diagonal and zeros everywhere else).  `triangular_matrix_matrix_left_solve` solves the equation $(\alpha A + I) Y = B$ for $Y$.  The two results $X$ and $Y$ are not equal in general.

#### Work-around requires changing all elements of the matrix

Users could work around this problem by first scaling the matrix $A$ by $\alpha$, and then solving for $Y$.  In the common case where the "other triangle" of $A$ holds another triangular matrix, users could not call `scale(alpha, A)`.  They would instead need to iterate over the elements of $A$ manually.  Users might also need to "unscale" the matrix after the solve.  Another option would be to copy the matrix $A$ before scaling.
```c++
for (size_t i = 0; i < A.extent(0); ++i) {
  for (size_t j = i + 1; j < A.extent(1); ++j) {
    A[i, j] *= alpha;
  }
}
triangular_matrix_matrix_left_solve(A, lower_triangle, implicit_unit_diagonal, B, Y);
for (size_t i = 0; i < A.extent(0); ++i) {
  for (size_t j = i + 1; j < A.extent(1); ++j) {
    A[i, j] /= alpha;
  }
}
```
Users cannot solve this problem by scaling $B$ (either with `scaled(1.0 / alpha, B)` or with `scale(1.0 / alpha, B)`).  Transforming $X$ into $Y$ or vice versa is mathematically nontrivial in general, and may introduce new failure conditions.  This issue occurs with both the in-place and out-of-place triangular solves.

#### Unsupported case occurs in LAPACK

The common case in LAPACK is calling `xTRSM` with `alpha` equal to one, but other values of `alpha` occur.  For example, `xTRTRI` calls `xTRSM` with `alpha` equal to $-1$.  Thus, we cannot dismiss this issue, as we could with `xTRMM`.

#### Fixes would not break backwards compatibility

As with triangular matrix products above, we can think of two ways to fix this issue.  First, we could add an `alpha` scaling parameter, analogous to the symmetric and Hermitian rank-1 and rank-k update functions.  Second, we could add a new kind of `Diagonal` template parameter type that expresses a "diagonal value."  For example, `implicit_diagonal_t{alpha}` (or a function form, `implicit_diagonal(alpha)`) would tell the algorithm not to access the diagonal elements, but instead to assume that their value is `alpha`.  Both of these solutions would let users specify the diagonal's scaling factor separately from the scaling factor for the rest of the matrix.  Those two scaling factors could differ, which is new functionality not offered by the BLAS.  More importantly, both of these solutions could be added later, after C++26, without breaking backwards compatibility.

# Ordering with respect to other proposals and LWG issues

We currently have two other `std::linalg` fix papers in review.

* P3222: Fix C++26 by adding `transposed` special cases for P2642 layouts (forwarded by LEWG to LWG on 2024-08-27 pending electronic poll results)

* P3050: "Fix C++26 by optimizing `linalg::conjugated` for noncomplex value types" (forwarded by LEWG to LWG on 2024-09-03 pending electronic poll results)

LEWG was aware of these two papers and this pending paper P3371 in its 2024-09-03 review of P3050R2.  All three of these papers increment the value of the `__cpp_lib_linalg` macro.  While this technically causes a conflict between the papers, advice from LEWG on 2024-09-03 was not to introduce special wording changes to avoid this conflict.

We also have two outstanding LWG issues.

* <a href="https://cplusplus.github.io/LWG/lwg-active.html#4136">LWG4136</a> specifies the behavior of Hermitian algorithms on diagonal matrix elements with nonzero imaginary part.  (As the BLAS Standard specifies and the Reference BLAS implements, the Hermitian algorithms do not access the imaginary parts of diagonal elements, and assume they are zero.)  In our view, P3371 does not conflict with LWG4136.

* <a href="https://cplusplus.github.io/LWG/lwg-active.html#4137">LWG4137</a>, "Fix Mandates, Preconditions, and Complexity elements of [linalg] algorithms," affects several sections touched by this proposal, including [linalg.algs.blas3.rankk] and [linalg.algs.blas3.rank2k].  We consider P3371 rebased atop the wording changes proposed by LWG4137.  While the wording changes may conflict in a formal ("diff") sense, it is our view that they do not conflict in a mathematical or specification sense.

# Implementation status

The following function overload sets need changing.

* `matrix_rank_1_update`
* `matrix_rank_1_update_c`
* `symmetric_matrix_rank_1_update`
* `hermitian_matrix_rank_1_update`
* `symmetric_matrix_rank_2_update`
* `hermitian_matrix_rank_2_update`
* `symmetric_matrix_rank_k_update`
* `hermitian_matrix_rank_k_update`
* `symmetric_matrix_rank_2k_update`
* `hermitian_matrix_rank_2k_update`

As of 2024/10/04, <a href="https://github.com/kokkos/stdBLAS/pull/293">Pull request 293</a> in the reference std::linalg implementation implements changes to the following functions, and adds tests to ensure test coverage of the new overloads.

* `matrix_rank_1_update`
* `matrix_rank_1_update_c`
* `symmetric_matrix_rank_1_update`
* `hermitian_matrix_rank_1_update`

# Acknowledgments

Many thanks (with permission) to Raffaele Solcà (CSCS Swiss National Supercomputing Centre, `raffaele.solca@cscs.ch`) for pointing out some of the issues fixed by this paper, as well as the issues leading to LWG4137.

# Wording

> Text in blockquotes is not proposed wording,
> but rather instructions for generating proposed wording.
> The � character is used to denote a placeholder section number
> which the editor shall determine.
>
> In **[version.syn]**, for the following definition,

```c++
#define __cpp_lib_linalg YYYYMML // also in <linalg>
```

> adjust the placeholder value `YYYYMML` as needed
> so as to denote this proposal's date of adoption.

## New exposition-only concepts for possibly-packed input and output matrices

> Add the following lines to the header `<linalg>` synopsis **[linalg.syn]**, just after the declaration of the exposition-only concept _`inout-matrix`_ and before the declaration of the exposition-only concept _`possibly-packed-inout-matrix`_.
> <i>[Editorial Note:</i> This addition is not shown in green, becuase the authors could not convince Markdown to format the code correctly. <i>-- end note]</i>

```c++
  template<class T>
    concept @_possibly-packed-in-matrix_@ = @_see below_@;     // @_exposition only_@

  template<class T>
    concept @_possibly-packed-out-matrix_@ = @_see below_@;     // @_exposition only_@
```

> Then, remove the declaration of the exposition-only concept _`possibly-packed-inout-matrix`_ from the header `<linalg>` synopsis **[linalg.syn]**.

> Then, add the following lines to **[linalg.helpers.concepts]**, just after the definition of the exposition-only variable template _`is-layout_blas_packed`_ and just before the definition of the exposition-only concept _`possibly-packed-inout-matrix`_.
> <i>[Editorial Note:</i> This addition is not shown in green, becuase the authors could not convince Markdown to format the code correctly. <i>-- end note]</i>

```c++
  template<class T>
    concept @_possibly-packed-in-matrix_@ =
      @_is-mdspan_@<T> && T::rank() == 2 &&
      (T::is_always_unique() || @_is-layout-blas-packed_@<typename T::layout_type>);

  template<class T>
    concept @_possibly-packed-out-matrix_@ =
      @_is-mdspan_@<T> && T::rank() == 2 &&
      is_assignable_v<typename T::reference, typename T::element_type> &&
      (T::is_always_unique() || @_is-layout-blas-packed_@<typename T::layout_type>);
```

> Then, remove the definition of the exposition-only concept _`possibly-packed-inout-matrix`_ from **[linalg.helpers.concepts]**.

> Then, in [linalg.helpers.concepts], change paragraph 3 to read as follows (new content "or _`possibly-packed-out-matrix`_" in green; removed content "or _`possibly-packed-inout-matrix`_" in red).

Unless explicitly permitted, any _`inout-vector`_, _`inout-matrix`_, _`inout-object`_, _`out-vector`_, _`out-matrix`_, _`out-object`_, <span style="color: green;">or _`possibly-packed-out-matrix`_</span>~~<span style="color: red;">, or _`possibly-packed-inout-matrix`_</span>~~ parameter of a function in [linalg] shall not overlap any other `mdspan` parameter of the function.

## Rank-1 update functions in synopsis

> In the header `<linalg>` synopsis **[linalg.syn]**, replace all the declarations of all the `matrix_rank_1_update`, `matrix_rank_1_update_c`, `symmetric_matrix_rank_1_update`, and `hermitian_matrix_rank_1_update` overloads to read as follows.
> <i>[Editorial Note:</i> 
> There are two changes here.
> First, the existing overloads become "overwriting" overloads.
> Second, new "updating" overloads are added.
>
> Changes do not use red or green highlighting, becuase the authors could not convince Markdown to format the code correctly.
> <i>-- end note]</i>

```c++
  // [linalg.algs.blas2.rank1], nonsymmetric rank-1 matrix update

  // overwriting nonsymmetric rank-1 matrix update
  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
    void matrix_rank_1_update(InVec1 x, InVec2 y, OutMat A);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
    void matrix_rank_1_update(ExecutionPolicy&& exec,
                              InVec1 x, InVec2 y, OutMat A);

  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
    void matrix_rank_1_update_c(InVec1 x, InVec2 y, OutMat A);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
    void matrix_rank_1_update_c(ExecutionPolicy&& exec,
                                InVec1 x, InVec2 y, OutMat A);

  // updating nonsymmetric rank-1 matrix update
  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2, @_in-matrix_@ InMat, @_out-matrix_@ OutMat>
    void matrix_rank_1_update(InVec1 x, InVec2 y, InMat E, OutMat A);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-matrix_@ InMat, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
    void matrix_rank_1_update(ExecutionPolicy&& exec,
                              InVec1 x, InVec2 y, InMat E, OutMat A);

  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2, @_in-matrix_@ InMat, @_out-matrix_@ OutMat>
    void matrix_rank_1_update_c(InVec1 x, InVec2 y, InMat E, OutMat A);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2, @_in-matrix_@ InMat, @_out-matrix_@ OutMat>
    void matrix_rank_1_update_c(ExecutionPolicy&& exec,
                                InVec1 x, InVec2 y, InMat E, OutMat A);

  // [linalg.algs.blas2.symherrank1], symmetric or Hermitian rank-1 matrix update

  // overwriting symmetric rank-1 matrix update 
  template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(Scalar alpha, InVec x, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        Scalar alpha, InVec x, OutMat A, Triangle t);
  template<@_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(InVec x, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        InVec x, OutMat A, Triangle t);

  // updating symmetric rank-1 matrix update 
  template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
  template<@_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(InVec x, InMat E, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        InVec x, InMat E, OutMat A, Triangle t);

  // overwriting Hermitian rank-1 matrix update 
  template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(Scalar alpha, InVec x, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        Scalar alpha, InVec x, OutMat A, Triangle t);
  template<@_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(InVec x, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        InVec x, OutMat A, Triangle t);

  // updating Hermitian rank-1 matrix update 
  template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
  template<@_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(InVec x, InMat E, OutMat A, Triangle t);
  template<class ExecutionPolicy,
           @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec,
                                        InVec x, InMat E, OutMat A, Triangle t);
```

## Rank-2 update functions in synopsis

> In the header `<linalg>` synopsis **[linalg.syn]**, replace all the declarations of all the `symmetric_matrix_rank_2_update` and `hermitian_matrix_rank_2_update` overloads to read as follows.
> <i>[Editorial Note:</i> These additions are not shown in green, becuase the authors could not convince Markdown to format the code correctly. <i>-- end note]</i>

```c++
  // [linalg.algs.blas2.rank2], symmetric and Hermitian rank-2 matrix updates

  // overwriting symmetric rank-2 matrix update
  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2_update(InVec1 x, InVec2 y, OutMat A, Triangle t);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2_update(ExecutionPolicy&& exec,
                                        InVec1 x, InVec2 y, OutMat A, Triangle t);

  // updating symmetric rank-2 matrix update
  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2_update(InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2_update(ExecutionPolicy&& exec,
                                        InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);

  // overwriting Hermitian rank-2 matrix update
  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2_update(InVec1 x, InVec2 y, OutMat A, Triangle t);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2_update(ExecutionPolicy&& exec,
                                        InVec1 x, InVec2 y, OutMat A, Triangle t);

  // updating Hermitian rank-2 matrix update
  template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2_update(InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);
  template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
           @_possibly-packed-in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2_update(ExecutionPolicy&& exec,
                                        InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);
```

## Rank-k update functions in synopsis

> In the header `<linalg>` synopsis **[linalg.syn]**, replace all the declarations of all the `symmetric_matrix_rank_k_update` and `hermitian_matrix_rank_k_update` overloads to read as follows.
> <i>[Editorial Note:</i> These additions are not shown in green, becuase the authors could not convince Markdown to format the code correctly. <i>-- end note]</i>

```c++
  // [linalg.algs.blas3.rankk], rank-k update of a symmetric or Hermitian matrix

  // overwriting symmetric rank-k matrix update
  template<class Scalar,
           @_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      Scalar alpha, InMat A, OutMat C, Triangle t);
  template<class ExecutionPolicy, class Scalar,
           @_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      ExecutionPolicy&& exec, Scalar alpha, InMat A, OutMat C, Triangle t);
  template<@_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      InMat A, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      ExecutionPolicy&& exec, InMat A, OutMat C, Triangle t);

  // updating symmetric rank-k matrix update
  template<class Scalar,
           @_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy, class Scalar,
           @_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      ExecutionPolicy&& exec, Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<@_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      ExecutionPolicy&& exec,
      InMat1 A, InMat2 E, OutMat C, Triangle t);

  // overwriting rank-k Hermitian matrix update
  template<class Scalar,
           @_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      Scalar alpha, InMat A, OutMat C, Triangle t);
  template<class ExecutionPolicy, class Scalar,
           @_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      ExecutionPolicy&& exec, Scalar alpha, InMat A, OutMat C, Triangle t);
  template<@_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      InMat A, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      ExecutionPolicy&& exec, InMat A, OutMat C, Triangle t);

  // updating rank-k Hermitian matrix update
  template<class Scalar,
           @_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy, class Scalar,
           @_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      ExecutionPolicy&& exec, Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<@_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1,
           @_possibly-packed-in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      ExecutionPolicy&& exec,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
```

## Rank-2k update functions in synopsis

> In the header `<linalg>` synopsis **[linalg.syn]**, replace all the declarations of all the `symmetric_matrix_rank_2k_update` and `hermitian_matrix_rank_2k_update` overloads to read as follows.
> <i>[Editorial Note:</i> These additions are not shown in green, becuase the authors could not convince Markdown to format the code correctly. <i>-- end note]</i>

```c++
  // [linalg.algs.blas3.rank2k], rank-2k update of a symmetric or Hermitian matrix

  // overwriting symmetric rank-2k matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(InMat1 A, InMat2 B, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, OutMat C, Triangle t);

  // updating symmetric rank-2k matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);

  // overwriting Hermitian rank-2k matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(InMat1 A, InMat2 B, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, OutMat C, Triangle t);

  // updating Hermitian rank-2k matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);
```

## Specification of nonsymmetric rank-1 update functions

> Replace the entire contents of [linalg.algs.blas2.rank1] with the following.

[1]{.pnum} The following elements apply to all functions in [linalg.algs.blas2.rank1].

[2]{.pnum} *Mandates*:

[2.1]{.pnum} _`possibly-multipliable`_`<OutMat, InVec2, InVec1>()` is `true`, and

[2.2]{.pnum} _`possibly-addable`_`(A, E, A)` is `true` for those overloads that take an `E` parameter.

[3]{.pnum} *Preconditions*:

[3.1]{.pnum} `multipliable(A, y, x)` is `true`, and

[3.2]{.pnum} _`addable`_`(A, E, A)` is `true` for those overloads that take an `E` parameter.

[4]{.pnum} *Complexity*: $O($ `x.extent(0)` × `y.extent(0)` $)$.

```c++
template<@_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
  void matrix_rank_1_update(InVec1 x, InVec2 y, OutMat A);
template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
  void matrix_rank_1_update(ExecutionPolicy&& exec, InVec1 x, InVec2 y, OutMat A);
```

[5]{.pnum} These functions perform a overwriting nonsymmetric nonconjugated rank-1 update.

<i>[Note:</i>
These functions correspond to the BLAS functions `xGER` (for real element types) and `xGERU` (for complex element types)[bib].
<i>-- end note]</i>

[6]{.pnum} *Effects*: Computes $A = x y^T$.

```c++
template<@_in-vector_@ InVec1, @_in-vector_@ InVec2, @_in-matrix_@ InMat, @_out-matrix_@ OutMat>
  void matrix_rank_1_update(InVec1 x, InVec2 y, InMat E, OutMat A);
template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2, @_in-matrix_@ InMat, @_out-matrix_@ OutMat>
  void matrix_rank_1_update(ExecutionPolicy&& exec, InVec1 x, InVec2 y, InMat E, OutMat A);
```

[7]{.pnum} These functions perform an updating nonsymmetric nonconjugated rank-1 update.

<i>[Note:</i>
These functions correspond to the BLAS functions `xGER` (for real element types) and `xGERU` (for complex element types)[bib].
<i>-- end note]</i>

[8]{.pnum} *Effects*: Computes $A = E + x y^T$.

[9]{.pnum} *Remarks*: `A` may alias `E`.

```c++
template<@_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
  void matrix_rank_1_update_c(InVec1 x, InVec2 y, OutMat A);
template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2, @_out-matrix_@ OutMat>
  void matrix_rank_1_update_c(ExecutionPolicy&& exec, InVec1 x, InVec2 y, OutMat A);
```

[10]{.pnum} These functions perform a overwriting nonsymmetric conjugated rank-1 update.

<i>[Note:</i>
These functions correspond to the BLAS functions `xGER` (for real element types) and `xGERC` (for complex element types)[bib].
<i>-- end note]</i>

[11]{.pnum} *Effects*:

[11.1]{.pnum} For the overloads without an `ExecutionPolicy` argument, equivalent to: 
```c++
matrix_rank_1_update(x, conjugated(y), A);
```

[11.2]{.pnum} otherwise, equivalent to:
```c++
matrix_rank_1_update(std::forward<ExecutionPolicy>(exec), x, conjugated(y), A);
```

## Specification of symmetric and Hermitian rank-1 update functions

> Replace the entire contents of [linalg.algs.blas2.symherrank1] with the following.

[1]{.pnum} <i>[Note:</i>
These functions correspond to the BLAS functions `xSYR`, `xSPR`, `xHER`, and `xHPR`[bib].
They have overloads taking a scaling factor `alpha`, because it would be impossible to express the update $A = A x x^T$ otherwise.
<i>-- end note]</i>

[2]{.pnum} The following elements apply to all functions in [linalg.algs.blas2.symherrank1].

[3]{.pnum} For any function `F` in this section that takes a parameter named `t`, an `InMat` template parameter, and a function parameter `InMat E`, `t` applies to accesses done through the parameter `E`.  `F` will only access the triangle of `E` specified by `t`.  For accesses of diagonal elements `E[i, i]`, `F` will use the value _`real-if-needed`_`(E[i, i])` if the name of `F` starts with `hermitian`.  For accesses `E[i, j]` outside the triangle specified by `t`, `F` will use the value

[3.1]{.pnum} _`conj-if-needed`_`(E[j, i])` if the name of `F` starts with `hermitian`, or

[3.2]{.pnum} `E[j, i]` if the name of `F` starts with `symmetric`.

[4]{.pnum} *Mandates*:

[4.1]{.pnum} If `OutMat` has `layout_blas_packed` layout, then the layout's `Triangle` template argument has the same type as the function's `Triangle` template argument;

[4.2]{.pnum} If the function has an `InMat` template parameter and `InMat` has `layout_blas_packed` layout, then the layout's `Triangle` template argument has the same type as the function's `Triangle` template argument;

[4.3]{.pnum} _`compatible-static-extents`_`<decltype(A), decltype(A)>(0, 1)` is `true`;

[4.4]{.pnum} _`compatible-static-extents`_`<decltype(A), decltype(x)>(0, 0)` is `true`; and

[4.5]{.pnum} _`possibly-addable`_`<decltype(A), decltype(E), decltype(A)>` is `true` for those overloads that take an `E` parameter.

[5]{.pnum} *Preconditions*:

[5.1]{.pnum} `A.extent(0)` equals `A.extent(1)`,

[5.2]{.pnum} `A.extent(0)` equals `x.extent(0)`, and

[5.3]{.pnum} _`addable`_`(A, E, A)` is `true` for those overloads that take an `E` parameter.

[6]{.pnum} *Complexity*: $O($ `x.extent(0)` × `x.extent(0)` $)$.

```c++
template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(Scalar alpha, InVec x, OutMat A, Triangle t);
template<class ExecutionPolicy,
         class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec,
                                      Scalar alpha, InVec x, OutMat A, Triangle t);
```

[7]{.pnum} These functions perform an overwriting symmetric rank-1 update of the symmetric matrix `A`, taking into account the `Triangle` parameter that applies to `A` ([linalg.general]).

[8]{.pnum} *Effects*: Computes $A = \alpha x x^T$, where the scalar $\alpha$ is `alpha`.

```c++
template<@_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(InVec x, OutMat A, Triangle t);
template<class ExecutionPolicy,
         @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec, InVec x, OutMat A, Triangle t);
```

[9]{.pnum} These functions perform an overwriting symmetric rank-1 update of the symmetric matrix `A`, taking into account the `Triangle` parameter that applies to `A` ([linalg.general]).

[10]{.pnum} *Effects*: Computes $A = x x^T$.

```c++
template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
template<class ExecutionPolicy,
         class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec,
                                      Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
```

[11]{.pnum} These functions perform an updating symmetric rank-1 update of the symmetric matrix `A` using the symmetric matrix `E`, taking into account the `Triangle` parameter that applies to `A` and `E` ([linalg.general]).

[12]{.pnum} *Effects*: Computes $A = E + \alpha x x^T$, where the scalar $\alpha$ is _`real-if-needed`_`(alpha)`.

```c++
template<@_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(InVec x, InMat E, OutMat A, Triangle t);
template<class ExecutionPolicy,
         @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_1_update(ExecutionPolicy&& exec,
                                      InVec x, InMat E, OutMat A, Triangle t);
```

[13]{.pnum} These functions perform an updating symmetric rank-1 update of the symmetric matrix `A` using the symmetric matrix `E`, taking into account the `Triangle` parameter that applies to `A` and `E` ([linalg.general]).

[14]{.pnum} *Effects*: Computes $A = E + x x^T$.

```c++
template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(Scalar alpha, InVec x, OutMat A, Triangle t);
template<class ExecutionPolicy,
         class Scalar, @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec,
                                      Scalar alpha, InVec x, OutMat A, Triangle t);
```

[15]{.pnum} These functions perform an overwriting Hermitian rank-1 update of the Hermitian matrix `A`, taking into account the `Triangle` parameter that applies to `A` ([linalg.general]).

[16]{.pnum} *Effects*: Computes $A = \alpha x x^H$, where the scalar $\alpha$ is _`real-if-needed`_`(alpha)`.

```c++
template<@_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(InVec x, OutMat A, Triangle t);
template<class ExecutionPolicy,
         @_in-vector_@ InVec, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec, InVec x, OutMat A, Triangle t);
```

[17]{.pnum} These functions perform an overwriting Hermitian rank-1 update of the Hermitian matrix `A`, taking into account the `Triangle` parameter that applies to `A` ([linalg.general]).

[18]{.pnum} *Effects*: Computes $A = x x^T$.

```c++
template<class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
template<class ExecutionPolicy,
         class Scalar, @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec,
                                      Scalar alpha, InVec x, InMat E, OutMat A, Triangle t);
```

[19]{.pnum} These functions perform an updating Hermitian rank-1 update of the Hermitian matrix `A` using the Hermitian matrix `E`, taking into account the `Triangle` parameter that applies to `A` and `E` ([linalg.general]).

[20]{.pnum} *Effects*: Computes $A = E + \alpha x x^H$, where the scalar $\alpha$ is _`real-if-needed`_`(alpha)`.

```c++
template<@_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(InVec x, InMat E, OutMat A, Triangle t);
template<class ExecutionPolicy,
         @_in-vector_@ InVec, @_possibly-packed-in-matrix_@ InMat, @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_1_update(ExecutionPolicy&& exec,
                                      InVec x, InMat E, OutMat A, Triangle t);
```

[21]{.pnum} These functions perform an updating Hermitian rank-1 update of the Hermitian matrix `A` using the Hermitian matrix `E`, taking into account the `Triangle` parameter that applies to `A` and `E` ([linalg.general]).

[22]{.pnum} *Effects*: Computes $A = E + x x^T$.

## Specification of symmetric and Hermitian rank-2 update functions

> Replace the entire contents of [linalg.algs.blas2.rank2] with the following.

[1]{.pnum} <i>[Note:</i> These functions correspond to the BLAS functions
`xSYR2`, `xSPR2`, `xHER2`, and `xHPR2` [bib]. <i>-- end note]</i>

[2]{.pnum} The following elements apply to all functions in [linalg.algs.blas2.rank2].

[3]{.pnum} For any function `F` in this section that takes a parameter named `t`, an `InMat` template parameter, and a function parameter `InMat E`, `t` applies to accesses done through the parameter `E`.  `F` will only access the triangle of `E` specified by `t`.  For accesses of diagonal elements `E[i, i]`, `F` will use the value _`real-if-needed`_`(E[i, i])` if the name of `F` starts with `hermitian`.  For accesses `E[i, j]` outside the triangle specified by `t`, `F` will use the value

[3.1]{.pnum} _`conj-if-needed`_`(E[j, i])` if the name of `F` starts with `hermitian`, or

[3.2]{.pnum} `E[j, i]` if the name of `F` starts with `symmetric`.

[4]{.pnum} *Mandates*:

[4.1]{.pnum} If `OutMat` has `layout_blas_packed` layout, then the layout's `Triangle` template argument has the same type as the function's `Triangle` template argument;

[4.2]{.pnum} If the function has an `InMat` template parameter and `InMat` has `layout_blas_packed` layout, then the layout's `Triangle` template argument has the same type as the function's `Triangle` template argument;

[4.3]{.pnum} _`compatible-static-extents`_`<decltype(A), decltype(A)>(0, 1)` is `true`;

[4.4]{.pnum} _`possibly-multipliable`_`<decltype(A), decltype(x), decltype(y)>()` is `true`; and

[4.5]{.pnum} _`possibly-addable`_`<decltype(A), decltype(E), decltype(A)>` is `true` for those overloads that take an `E` parameter.

[5]{.pnum} *Preconditions*:

[5.1]{.pnum} `A.extent(0)` equals `A.extent(1)`,

[5.2]{.pnum} _`multipliable`_`(A, x, y)` is `true`, and

[5.3]{.pnum} _`addable`_`(A, E, A)` is `true` for those overloads that take an `E` parameter.

[6]{.pnum} *Complexity*: $O($ `x.extent(0)` × `y.extent(0)` $)$.

```c++
template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_2_update(InVec1 x, InVec2 y, OutMat A, Triangle t);
template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_2_update(ExecutionPolicy&& exec,
                                      InVec1 x, InVec2 y, OutMat A, Triangle t);
```

[7]{.pnum} These functions perform an overwriting symmetric rank-2 update of the symmetric matrix `A`, taking into account the `Triangle` parameter that applies to `A` ([linalg.general]).

[8]{.pnum} Effects: Computes $A = x y^T + y x^T$.

```c++
template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_2_update(InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);
template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void symmetric_matrix_rank_2_update(ExecutionPolicy&& exec,
                                      InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);
```

[9]{.pnum} These functions perform an updating symmetric rank-2 update of the symmetric matrix `A` using the symmetric matrix `E`, taking into account the `Triangle` parameter that applies to `A` and `E` ([linalg.general]).

[10]{.pnum} Effects: Computes $A = E + x y^T + y x^T$.

```c++
template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_2_update(InVec1 x, InVec2 y, OutMat A, Triangle t);
template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_2_update(ExecutionPolicy&& exec,
                                      InVec1 x, InVec2 y, OutMat A, Triangle t);
```

[11]{.pnum} These functions perform an overwriting Hermitian rank-2 update of the Hermitian matrix `A`, taking into account the `Triangle` parameter that applies to `A` ([linalg.general]).

[12]{.pnum} Effects: Computes $A = x y^H + y x^H$.

```c++
template<@_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_2_update(InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);
template<class ExecutionPolicy, @_in-vector_@ InVec1, @_in-vector_@ InVec2,
         @_possibly-packed-in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat, class Triangle>
  void hermitian_matrix_rank_2_update(ExecutionPolicy&& exec,
                                      InVec1 x, InVec2 y, InMat E, OutMat A, Triangle t);
```

[13]{.pnum} These functions perform an updating Hermitian rank-2 update of the Hermitian matrix `A` using the Hermitian matrix `E`, taking into account the `Triangle` parameter that applies to `A` and `E` ([linalg.general]).

[14]{.pnum} Effects: Computes $A = E + x y^H + y x^H$.

## Specification of rank-k update functions

> Replace the entire contents of [linalg.algs.blas3.rankk] with the following.

<i>[Note:</i> These functions correspond to the BLAS functions
`xSYRK` and `xHERK`. <i>-- end note]</i>

[1]{.pnum} The following elements apply to all functions in [linalg.algs.blas3.rankk].

[2]{.pnum} For any function `F` in this section that takes a parameter named `t`, an `InMat2` template parameter, and a function parameter `InMat2 E`, `t` applies to accesses done through the parameter `E`.  `F` will only access the triangle of `E` specified by `t`.  For accesses of diagonal elements `E[i, i]`, `F` will use the value _`real-if-needed`_`(E[i, i])` if the name of `F` starts with `hermitian`.  For accesses `E[i, j]` outside the triangle specified by `t`, `F` will use the value

[2.1]{.pnum} _`conj-if-needed`_`(E[j, i])` if the name of `F` starts with `hermitian`, or

[2.2]{.pnum} `E[j, i]` if the name of `F` starts with `symmetric`.

[3]{.pnum} *Mandates:*

  * [3.1]{.pnum} If `OutMat` has `layout_blas_packed` layout, then the
      layout's `Triangle` template argument has the same type as
      the function's `Triangle` template argument.

  * [3.2]{.pnum} If the function takes an `InMat2` template parameter and
      if `InMat2` has `layout_blas_packed` layout, then the
      layout's `Triangle` template argument has the same type as
      the function's `Triangle` template argument.

  * [3.3]{.pnum} _`possibly-multipliable`_`<decltype(A), decltype(transposed(A)), decltype(C)>` is `true`.

  * [3.4]{.pnum} _`possibly-addable`_`<decltype(C), decltype(E), decltype(C)>` is `true` for those overloads that take an `E` parameter.

[4]{.pnum} *Preconditions:*

  * [4.1]{.pnum} _`multipliable`_`(A, transposed(A), C)` is `true`.  <i>[Note:</i> This implies that `C` is square. <i>-- end note]</i>

  * [4.2]{.pnum} _`addable`_`(C, E, C)` is `true` for those overloads that take an `E` parameter.

[5]{.pnum} *Complexity:* $O($ `A.extent(0)` $\cdot$ `A.extent(1)` $\cdot$ `A.extent(0)` $)$.

[6]{.pnum} *Remarks:* `C` may alias `E` for those overloads that take an `E` parameter.

```c++
template<class Scalar,
         @_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  Scalar alpha,
  InMat A,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         class Scalar,
         @_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  Scalar alpha,
  InMat A,
  OutMat C,
  Triangle t);
```

[5]{.pnum} *Effects:*
Computes $C = \alpha A A^T$,
where the scalar $\alpha$ is `alpha`.

```c++
template<@_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  InMat A,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  InMat A,
  OutMat C,
  Triangle t);
```

[6]{.pnum} *Effects:*
Computes $C = A A^T$.

```c++
template<class Scalar,
         @_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  Scalar alpha,
  InMat A,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         class Scalar,
         @_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  Scalar alpha,
  InMat A,
  OutMat C,
  Triangle t);
```

[7]{.pnum} *Effects:*
Computes $C = \alpha A A^H$,
where the scalar $\alpha$ is _`real-if-needed`_`(alpha)`.

```c++
template<@_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  InMat A,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  InMat A,
  OutMat C,
  Triangle t);
```

[8]{.pnum} *Effects:*
Computes $C = A A^H$.

```c++
template<class Scalar,
         @_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  Scalar alpha,
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         class Scalar,
         @_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  Scalar alpha,
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
```

[9]{.pnum} *Effects:*
Computes $C = E + \alpha A A^T$,
where the scalar $\alpha$ is _`real-if-needed`_`(alpha)`.

```c++
template<@_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
```

[10]{.pnum} *Effects:*
Computes $C = E + A A^T$.

```c++
template<class Scalar,
         @_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  Scalar alpha,
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         class Scalar,
         @_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  Scalar alpha,
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
```

[11]{.pnum} *Effects:*
Computes $C = E + \alpha A A^H$,
where the scalar $\alpha$ is _`real-if-needed`_`(alpha)`.

```c++
template<@_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_possibly-packed-in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  ExecutionPolicy&& exec,
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
```

[12]{.pnum} *Effects:*
Computes $C = E + A A^H$.

## Specification of rank-2k update functions

> Replace the entire contents of [linalg.algs.blas3.rank2k] with the following.

<i>[Note:</i> These functions correspond to the BLAS functions
`xSYR2K` and `xHER2K`. <i>-- end note]</i>

[1]{.pnum} The following elements apply to all functions in [linalg.algs.blas3.rank2k].

[2]{.pnum} For any function `F` in this section that takes a parameter named `t`, an `InMat3` template parameter, and a function parameter `InMat3 E`, `t` applies to accesses done through the parameter `E`.  `F` will only access the triangle of `E` specified by `t`.  For accesses of diagonal elements `E[i, i]`, `F` will use the value _`real-if-needed`_`(E[i, i])` if the name of `F` starts with `hermitian`.  For accesses `E[i, j]` outside the triangle specified by `t`, `F` will use the value

[2.1]{.pnum} _`conj-if-needed`_`(E[j, i])` if the name of `F` starts with `hermitian`, or

[2.2]{.pnum} `E[j, i]` if the name of `F` starts with `symmetric`.

[3]{.pnum} *Mandates:*

  * [3.1]{.pnum} If `OutMat` has `layout_blas_packed` layout, then the
      layout's `Triangle` template argument has the same type as
      the function's `Triangle` template argument;

  * [3.2]{.pnum} If the function takes an `InMat3` template parameter and
      if `InMat3` has `layout_blas_packed` layout, then the
      layout's `Triangle` template argument has the same type as
      the function's `Triangle` template argument.

  * [3.3]{.pnum} _`possibly-multipliable`_`<decltype(A), decltype(transposed(B)), decltype(C)>` is `true`.

  * [3.4]{.pnum} _`possibly-multipliable`_`<decltype(B), decltype(transposed(A)), decltype(C)>` is `true`.

  * [3.5]{.pnum} _`possibly-addable`_`<decltype(C), decltype(E), decltype(C)>` is `true` for those overloads that take an `E` parameter.

[4]{.pnum} *Preconditions:*

  * [4.1]{.pnum} _`multipliable`_`(A, transposed(B), C)` is `true`.

  * [4.2]{.pnum} _`multipliable`_`(B, transposed(A), C)` is `true`.  <i>[Note:</i> This and the previous imply that `C` is square. <i>-- end note]</i>

  * [4.3]{.pnum} _`addable`_`(C, E, C)` is `true` for those overloads that take an `E` parameter.

[4]{.pnum} *Complexity:* $O($ `A.extent(0)` $\cdot$ `A.extent(1)` $\cdot$ `B.extent(0)` $)$

[5]{.pnum} *Remarks:* `C` may alias `E` for those overloads that take an `E` parameter.

```c++
template<@_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  InMat1 A,
  InMat2 B,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  ExecutionPolicy&& exec,
  InMat1 A,
  InMat2 B,
  OutMat C,
  Triangle t);
```

[5]{.pnum} *Effects:* Computes $C = A B^T + B A^T$. 

```c++
template<@_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  InMat1 A,
  InMat2 B,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  ExecutionPolicy&& exec,
  InMat1 A,
  InMat2 B,
  OutMat C,
  Triangle t);
```

[6]{.pnum} *Effects:* Computes $C = A B^H + B A^H$.

```c++
template<@_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_in-matrix_@ InMat3,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  InMat1 A,
  InMat2 B,
  InMat3 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_in-matrix_@ InMat3,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  ExecutionPolicy&& exec,
  InMat1 A,
  InMat2 B,
  InMat3 E,
  OutMat C,
  Triangle t);
```

[7]{.pnum} *Effects:* Computes $C = E + A B^T + B A^T$. 

```c++
template<@_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_in-matrix_@ InMat3,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  InMat1 A,
  InMat2 B,
  InMat3 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_in-matrix_@ InMat3,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  ExecutionPolicy&& exec,
  InMat1 A,
  InMat2 B,
  InMat3 E,
  OutMat C,
  Triangle t);
```

[8]{.pnum} *Effects:* Computes $C = E + A B^H + B A^H$.

# Appendix A: Example of a generic numerical algorithm

The following example shows how to implement a generic numerical algorithm, `two_norm_abs`.  This algorithm computes the absolute value of a variety of number types.  For complex numbers, it returns the magnitude, which is the same as the two-norm of the two-element vector composed of the real and imaginary parts of the complex number.  <a href="https://godbolt.org/z/KcGPdjPvP">This Compiler Explorer link</a> demonstrates the implementation.  This is not meant to show an ideal implementation.  (A better one would use rescaling, in the manner of `std::hypot`,  to avoid undue overflow or underflow.)  Instead, it illustrates generic numerical algorithm development.  Commenting out the `#define DEFINE_CONJ_REAL_IMAG_FOR_REAL 1` line shows that without ADL-findable `conj`, `real`, and `imag`, users' generic numerical algorithms would need more special cases and more assumptions on number types.

```c++
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <type_traits>

#define DEFINE_CONJ_REAL_IMAG_FOR_REAL 1

template<class T>
constexpr bool is_std_complex = false;
template<class R>
constexpr bool is_std_complex<std::complex<R>> = true;

template<class T>
auto two_norm_abs(T t) {
  if constexpr (std::is_unsigned_v<T>) {
    return t;
  }
  else if constexpr (std::is_arithmetic_v<T> || is_std_complex<T>) {
    return std::abs(t);
  }
#if ! defined(DEFINE_CONJ_REAL_IMAG_FOR_REAL)
  else if constexpr (requires(T x) {
      {abs(x)} -> std::convertible_to<T>;
    }) {
    return T{abs(t)};
  }
#endif
  else if constexpr (requires(T x) {
      {sqrt(real(x * conj(x)))} -> std::same_as<decltype(real(x))>;
    }) {
    return sqrt(real(t * conj(t)));
  }
  else {
    static_assert(false, "No reasonable way to implement abs(t)");
  }
}

struct MyRealNumber {
  MyRealNumber() = default;
  MyRealNumber(double value) : value_(value) {}

  double value() const {
    return value_;
  }

  friend bool operator==(MyRealNumber, MyRealNumber) = default;
  friend MyRealNumber operator-(MyRealNumber x) {
    return {-x.value_};
  }
  friend MyRealNumber operator+(MyRealNumber x, MyRealNumber y) {
    return {x.value_ + y.value_};
  }
  friend MyRealNumber operator-(MyRealNumber x, MyRealNumber y) {
    return {x.value_ - y.value_};
  }
  friend MyRealNumber operator*(MyRealNumber x, MyRealNumber y) {
    return x.value_ * y.value_;
  }

#if defined(DEFINE_CONJ_REAL_IMAG_FOR_REAL)
  friend MyRealNumber conj(MyRealNumber x) { return x; }
  friend MyRealNumber real(MyRealNumber x) { return x; }
  friend MyRealNumber imag(MyRealNumber x) { return {}; }
#else
  friend MyRealNumber abs(MyRealNumber x) { return std::abs(x.value_); }
#endif
  friend MyRealNumber sqrt(MyRealNumber x) { return std::sqrt(x.value_); }

private:
  double value_{};
};

class MyComplexNumber {
public:
  MyComplexNumber(MyRealNumber re, MyRealNumber im = {}) : re_(re), im_(im) {}
  MyComplexNumber() = default;

  std::complex<double> value() const {
    return {re_.value(), im_.value()};
  }

  friend bool operator==(MyComplexNumber, MyComplexNumber) = default;
  friend MyComplexNumber operator*(MyComplexNumber z, MyComplexNumber w) {
    return {real(z) * real(w) - imag(z) * imag(w),
      real(z) * imag(w) + imag(z) * real(w)};
  }
  friend MyComplexNumber conj(MyComplexNumber z) { return {real(z), -imag(z)}; }
  friend MyRealNumber real(MyComplexNumber z) { return z.re_; }
  friend MyRealNumber imag(MyComplexNumber z) { return z.im_; }

private:
  MyRealNumber re_{};
  MyRealNumber im_{};
};

int main() {
  [[maybe_unused]] double x1 = two_norm_abs(-4.2);
  assert(x1 == 4.2);

  [[maybe_unused]] float y0 = two_norm_abs(std::complex<float>{-3.0f, 4.0f});
  assert(y0 == 5.0f);

  [[maybe_unused]] MyRealNumber r = two_norm_abs(MyRealNumber{-6.7});
  assert(r == MyRealNumber{6.7});

  [[maybe_unused]] MyRealNumber z = two_norm_abs(MyComplexNumber{-3, 4});
  assert(z.value() == 5.0);
 
  return 0;
}
```
