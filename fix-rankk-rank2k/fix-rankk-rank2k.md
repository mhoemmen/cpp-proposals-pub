
---
title: "Fix C++26 by making the symmetric and Hermitian rank-k and rank-2k updates consistent with the BLAS"
document: P3371R1
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

    * Add Ilya Burylov as coauthor

    * Change title from "Fix C++26 by making the symmetric and Hermitian rank-k and rank-2k updates consistent with the BLAS," to "Fix C++26 by making the symmetric and Hermitian rank-1, rank-k, and rank-2k updates consistent with the BLAS."  Change abstract to list the newly proposed rank-1 update change (constrain `Scalar` to be noncomplex).

    * Reorganize and expand nonwording sections

    * Add "`C` may alias `E`" to all the new updating overloads of the symmetric and Hermitian rank-k and rank-2k functions

# Abstract

The [linalg] functions `hermitian_rank_1_update`, `symmetric_matrix_rank_k_update`,  `hermitian_matrix_rank_k_update`, `symmetric_matrix_rank_2k_update`, and  `hermitian_matrix_rank_2k_update` currently have behavior inconsistent with their corresponding BLAS (Basic Linear Algebra Subroutines) routines.  (`symmetric_rank_1_update`, `hermitian_rank_2_update`, and `symmetric_rank_2_update` are fine.)  Also, the behavior of the rank-k and rank-2k updates is inconsistent with that of `matrix_product`, even though in mathematical terms they are special cases of a matrix-matrix product.  We propose three fixes.

1. Add "updating" overloads to the symmetric and Hermitian rank-k and rank-2k update functions.  The new overloads are analogous to the updating overloads of `matrix_product`.  For example, `symmetric_matrix_rank_k_update(A, scaled(beta, C), C, upper_triangle)` will perform $C := \beta C + A A^T$.

2. Change the behavior of the existing symmetric and Hermitian rank-k and rank-2k update functions to be "overwriting."  For example, `symmetric_matrix_rank_k_update(A, C, upper_triangle)` will perform $C := A A^T$ instead of $C := C + A A^T$.

3. For `hermitian_rank_1_update` and `hermitian_rank_k_update`, we constrain the `Scalar` template parameter (if any) to be noncomplex.  This ensures that the update will be mathematically Hermitian.

Items (2) and (3) are breaking changes to the current Working Draft.  Thus, we must finish this before finalization of C++26.

# Discussion and proposed changes

## Support both overwriting and updating rank-k and rank-2k updates

### BLAS supports scaling factor beta; std::linalg currently does not

Each function in std::linalg generally corresponds to one or more routines or functions in the original BLAS (Basic Linear Algebra Subroutines).  Every computation that the BLAS can do, a function in std::linalg should be able to do.

One `std::linalg` user <a href="https://github.com/kokkos/stdBLAS/issues/272#issuecomment-2248273146">reported</a> an exception to this rule.  The BLAS routine `DSYRK` (Double-precision SYmmetric Rank-K update) computes $C := \beta C + \alpha A A^T$, but the corresponding `std::linalg` function `symmetric_matrix_rank_k_update` only computes $C := C + \alpha A A^T$.  That is, `std::linalg` currently has no way to express this BLAS operation with a general $\beta$ scaling factor.  This issue applies to all of the symmetric and Hermitian rank-k and rank-2k update functions.

* `symmetric_matrix_rank_k_update`: computes $C := C + \alpha A A^T$ 
* `hermitian_matrix_rank_k_update`: computes $C := C + \alpha A A^H$
* `symmetric_matrix_rank_2k_update`: computes $C := C + \alpha A B^H + \alpha B A^H$
* `hermitian_matrix_rank_2k_update`: computes $C := C + \alpha A B^H + \bar{\alpha} B A^H$, where $\bar{\alpha}$ denotes the complex conjugate of $\alpha$

### Inconsistency with general matrix product

These functions implement special cases of matrix-matrix products.  The `matrix_product` function in `std::linalg` implements the general case of matrix-matrix products.  This function corresponds to the BLAS's `SGEMM`, `DGEMM`, `CGEMM`, and `ZGEMM`, which compute $C := \beta C + \alpha A B$, where $\alpha$ and $\beta$ are scaling factors.  The `matrix_product` function has two kinds of overloads:

1. *overwriting* ($C = A B$) and

2. *updating* ($C = E + A B$).

The updating overloads handle the general $\alpha$ and $\beta$ case by `matrix_product(scaled(alpha, A), B, scaled(beta, C), C)`.  The specification explicitly permits the input `scaled(beta, C)` to alias the output `C` (**[linalg.algs.blas3.gemm]** 10: "*Remarks*: `C` may alias `E`").  The `std::linalg` library provides overwriting and updating overloads so that it can do everything that the BLAS does, just in a more idiomatically C++ way.  Please see <a href="https://isocpp.org/files/papers/P1673R13.html#function-argument-aliasing-and-zero-scalar-multipliers">P1673R13 Section 10.3</a> ("Function argument aliasing and zero scalar multipliers") for a more detailed explanation.

### Fix requires changing behavior of existing overloads

The problem with the current symmetric and Hermitian rank-k and rank-2k functions is that they have the same _calling syntax_ as the overwriting version of `matrix_product`, but _semantics_ that differ from both the overwriting and the updating versions of `matrix_product`.  For example,
```c++
hermitian_matrix_rank_k_update(alpha, A, C);
```
updates $C$ with $C - \alpha A A^H$, but
```c++
matrix_product(scaled(alpha, A), conjugate_transposed(A), C);
```
overwrites $C$ with $\alpha A A^H$.  The current rank-k and rank-2k overloads are not overwriting, so we can't just fix this problem by introducing an "updating" overload for each function.  

Incidentally, the fact that these functions have "update" in their name is not relevant, because that naming choice is original to the BLAS.  The BLAS calls its corresponding `xSYRK`, `xHERK`, `xSYR2K`, and `xHER2K` routines "{Symmetric, Hermitian} rank {one, two} update," even though setting $\beta = 0$ makes these routines "overwriting" in the sense of `std::linalg`.

### Add new updating overloads; make existing ones overwriting

We propose to fix this by making the four functions work just like `matrix_vector_product` or `matrix_product`.  This entails three changes.

1. Add a new exposition-only concept _`possibly-packed-out-matrix`_ for constraining the output-only parameter of the new updating overloads (see (2)).

2. Add "updating" overloads of the symmetric and Hermitian rank-k and rank-2k update functions.

    a. The updating overloads take a new input matrix parameter `E`, analogous to the updating overloads of `matrix_product`, and make `C` an output parameter instead of an in/out parameter.  For example, `symmetric_matrix_rank_k_update(A, E, C, upper_triangle)` computes $C = E + A A^T$.
    
    b. Explicitly permit `C` and `E` to alias, thus permitting the desired case where `E` is `scaled(beta, C)`.
    
    c. The updating overloads take `E` as an _`in-matrix`_, and take `C` as a _`possibly-packed-out-matrix`_ (instead of a _`possibly-packed-inout-matrix`_).
    
    d. `E` must be accessed as a symmetric or Hermitian matrix (depending on the function name) and using the same triangle as `C`.  (The existing [linalg.general] 4 wording for symmetric and Hermitian behavior does not cover `E`.)

3. Change the behavior of the existing symmetric and Hermitian rank-k and rank-2k overloads to be overwriting instead of updating.

    a. For example, `symmetric_matrix_rank_k_update(A, C, upper_triangle)` will compute $C = A A^T$ instead of $C := C + A A^T$.
    
    b. Change `C` from a _`possibly-packed-inout-matrix`_ to a _`possibly-packed-out-matrix`_.

Items (2) and (3) are breaking changes to the current Working Draft.  This needs to be so that we can provide the overwriting behavior $C := \alpha A A^T$ or $C := \alpha A A^H$ that the corresponding BLAS routines already provide.  Thus, we must finish this before finalization of C++26.

Both sets of overloads still only write to the specified triangle (lower or upper) of the output matrix `C`.  As a result, the new updating overloads only read from that triangle of the input matrix `E`.  Therefore, even though `E` may be a different matrix than `C`, the updating overloads do not need an additional `Triangle t_E` parameter for `E`.  The `symmetric_*` functions interpret `E` as symmetric in the same way that they interpret `C` as symmetric, and the `hermitian_*` functions interpret `E` as Hermitian in the same way that they interpret `C` as Hermitian.  Nevertheless, we do need new wording to explain how the functions may interpret and access `E`.

### Do not apply this fix to rank-1 or rank-2 update functions

The rank-k and rank-2k update functions have the following rank-1 and rank-2 analogs, where $A$ denotes a symmetric or Hermitian matrix (depending on the function's name) and $x$ and $y$ denote vectors.

* `symmetric_matrix_rank_1_update`: computes $A := A + \alpha x x^T$
* `hermetian_matrix_rank_1_update`: computes $A := A + \alpha x x^H$
* `symmetric_matrix_rank_2_update`: computes $A := A + \alpha x y^T + \alpha y x^T$
* `hermitian_matrix_rank_2_update`: computes $A := A + \alpha x y^H + \bar{\alpha} x y^H$

We do NOT propose to change these functions analogously to the rank-k and rank-2k update functions.  This is because the BLAS routines corresponding to the rank-1 and rank-2 functions -- `xSYR`, `xHER`, `xSYR2`, and `xHER2` -- do not have a way to supply a $\beta$ scaling factor.  That is, these `std::linalg` functions can already do everything that their corresponding BLAS routines can do.  This is consistent with our design intent in <a href="https://isocpp.org/files/papers/P1673R13.html#function-argument-aliasing-and-zero-scalar-multipliers">Section 10.3 of P1673R3</a> for translating Fortran `INTENT(INOUT)` arguments into a C++ idiom.

> b. Else, if the BLAS function unconditionally updates (like `xGER`), we retain read-and-write behavior for that argument.
>
> c. Else, if the BLAS function uses a scalar `beta` argument to decide whether to read the output argument as well as write to it (like `xGEMM`), we provide two versions: a write-only [that is, "overwriting"] version (as if `beta` is zero), and a read-and-write [that is, "updating"] version (as if `beta` is nonzero).

The rank-1 and rank-2 update functions "unconditionally update," in the same way that the BLAS's general rank-1 update routine `xGER` does.  However, the BLAS's rank-k and rank-2k update functions "use a scalar `beta` argument...," so for consistency, it makes sense for `std::linalg` to provide both overwriting and updating versions.  Users who want overwriting behavior in a rank-1 or rank-2 update can call the corresponding rank-k or rank-2k updating function with a matrix with one column ($k = 1$) instead of a vector.

Since we do not propose changing the symmetric and Hermitian rank-1 and rank-2 functions, we retain the exposition-only concept _`possibly-packed-inout-matrix`_, which they use to constrain their parameter `A`.

## Constrain alpha in Hermitian rank-1 and rank-k updates to be noncomplex

### Scaling factor alpha needs to be noncomplex, else update may be non-Hermitian

The C++ Working Draft already has `Scalar alpha` overloads of `hermitian_rank_k_update`.  The `Scalar` type currently can be complex.  However, if `alpha` has nonzero imaginary part, then $\alpha A A^H$ may no longer be a Hermitian matrix, even though $A A^H$ is mathematically always Hermitian.  For example, if $A$ is the identity matrix (with ones on the diagonal and zeros elsewhere) and $\alpha = i$, then $\alpha A A^H$ is the diagonal matrix whose diagonal elements are all $i$.  While that matrix is symmetric, it is not Hermitian, because all elements on the diagonal of a Hermitian matrix must have nonzero imaginary part.  The rank-1 update function `hermitian_rank_1_update` has the analogous issue.

The BLAS solves this problem by having the Hermitian rank-1 update routines `xHER` and rank-k update routines `xHERK` take the scaling factor $\alpha$ as a noncomplex number.  This suggests a fix: For all `hermitian_rank_1_update` and `hermitian_rank_k_update` overloads that take `Scalar alpha`, constrain `Scalar` so that it is noncomplex.  We can avoid introducing new undefined behavior (or "valid but unspecified" elements of the output matrix) by making "noncomplex" a constraint on the `Scalar` type of `alpha`.  "Noncomplex" should follow the definition of "noncomplex" used by _`conj-if-needed`_: either an arithmetic type, or `conj(E)` is not ADL-findable for an expression `E` of type `Scalar`.

### Nothing wrong with rank-2 or rank-2k updates

This issue does *not* arise with the rank-2 or rank-2k updates.  In the BLAS, the rank-2 updates `xHER2` and the rank-2k updates `xHER2K` all take `alpha` as a complex number.  There's no need to impose a precondition on the value of `alpha`, because $\alpha A B^H + \bar{alpha} B A^H$ is Hermitian by construction.

### Nothing wrong with scaling factor beta

The scaling factor `beta` only arises with the rank-k update function `hermitian_rank_k_update`.  The current wording behaves correctly with respect to `beta`.  For the new updating overloads of `hermitian_rank_k_update`, [linalg] expresses a `beta` scaling factor by letting users supply `scaled(beta, C)` as the argument for `E`.  The current wording merely requires that `scaled(beta, C)` be Hermitian.  It is actually incorrect to constrain `beta` or `C` separately.  For example, if $\beta = -i$ and $C$ is the matrix whose elements are all $i$, then $C$ is not Hermitian but $\beta C$ (and therefore `scaled(beta, C)`) is Hermitian.

The current wording for `hermitian_rank_1_update` is correct.  As explained elsewhere in this proposal, `hermitian_rank_1_update` only needs to support what `xHER` supports, and `xHER` does not support updates with a `beta` scaling factor.

This issue does *not* arise with the rank-2k updates.  In the BLAS, `xHER2K` takes `beta` as a real number.  The previous paragraph's reasoning for `beta` applies here as well.

This issue also does not arise with the rank-2 updates.  In the Reference BLAS, the rank-2 update routines `xHER2` do not have a way to supply `beta`.  As explained in P1673, [linalg] prefers to limit itself to the functionality in the Reference BLAS, as this is the most widely available implementation.  Thus, [linalg] retains the no-`beta` interface.  Interestingly, in the BLAS Standard, `xHER2` *does* take `beta`.  The BLAS Standard says that "$\alpha$ is a complex scalar and and [sic] $\beta$ is a real scalar."  The Fortran 77 and C bindings specify the type of `beta` as real (`<rtype>` resp. `RSCALAR_IN`), but the Fortran 95 binding lists both `alpha` and `beta` as `COMPLEX(<wp>)`.  The type of `beta` in the Fortran 95 is likely a typo, considering the wording.

### Nothing wrong with Hermitian matrix-vector and matrix-matrix products

In our view, the current behavior of `hermitian_matrix_vector_product` and `hermitian_matrix_product` is correct with respect to the `alpha` scaling factor.  These functions do not need extra overloads that take `Scalar alpha`.  Users who want to supply `alpha` with nonzero imaginary part should *not* scale the matrix `A` (as in `scaled(alpha, A)`).  Instead, they should scale the input vector `x`, as in the following.
```c++
auto alpha = std::complex{0.0, 1.0};
hermitian_matrix_vector_product(A, upper_triangle, scaled(alpha, x), y);
```

#### In BLAS, matrix is Hermitian, but scaled matrix need not be

In Chapter 2 of the BLAS Standard, both `xHEMV` and `xHEMM` take the scaling factors $\alpha$ and $\beta$ as complex numbers (`COMPLEX<wp>`, where `<wp>` represents the current working precision).  The BLAS permits `xHEMV` or `xHEMM` to be called with `alpha` whose imaginary part is nonzero.  The matrix that the BLAS assumes to be Hermitian is $A$, not $\alpha A$.  Even if $A$ is Hermitian, $\alpha A$ might not necessarily be Hermitian.  For example, if $A$ is the identity matrix (diagonal all ones) and $\alpha$ is $i$, then $\alpha A$ is not Hermitian but skew-Hermitian.

The current [linalg] wording requires that the input matrix be Hermitian.  This excludes using `scaled(alpha, A)` as the input matrix, where `alpha` has nonzero imaginary part.  For example, the following gives mathematically incorrect results.
```c++
auto alpha = std::complex{0.0, 1.0};
hermitian_matrix_vector_product(scaled(alpha, A), upper_triangle, x, y);
```
Note that the behavior of this is still well defined, at least after applying the fix proposed in LWG4136 for diagonal elements with nonzero imaginary part.  It does not violate a precondition.  Therefore, the Standard has no way to tell the user that they did something wrong.

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

# Ordering with respect to other proposals and LWG issues

We currently have two other `std::linalg` fix papers in review.

* P3222: Fix C++26 by adding `transposed` special cases for P2642 layouts (forwarded by LEWG to LWG on 2024-08-27 pending electronic poll results)

* P3050: "Fix C++26 by optimizing `linalg::conjugated` for noncomplex value types" (forwarded by LEWG to LWG on 2024-09-03 pending electronic poll results)

LEWG was aware of these two papers and this pending paper P3371 in its 2024-09-03 review of P3050R2.  All three of these papers increment the value of the `__cpp_lib_linalg` macro.  While this technically causes a conflict between the papers, advice from LEWG on 2024-09-03 was not to introduce special wording changes to avoid this conflict.

We also have two outstanding LWG issues.

* <a href="https://cplusplus.github.io/LWG/lwg-active.html#4136">LWG4136</a> specifies the behavior of Hermitian algorithms on diagonal matrix elements with nonzero imaginary part.  (As the BLAS Standard specifies and the Reference BLAS implements, the Hermitian algorithms do not access the imaginary parts of diagonal elements, and assume they are zero.)  In our view, P3371 does not conflict with LWG4136.

* <a href="https://cplusplus.github.io/LWG/lwg-active.html#4137">LWG4137</a>, "Fix Mandates, Preconditions, and Complexity elements of [linalg] algorithms," affects several sections touched by this proposal, including [linalg.algs.blas3.rankk] and [linalg.algs.blas3.rank2k].  We consider P3371 rebased atop the wording changes proposed by LWG4137.  While the wording changes may conflict in a formal ("diff") sense, it is our view that they do not conflict in a mathematical or specification sense.

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

## New exposition-only concept

> In the header `<linalg>` synopsis **[linalg.syn]**,
> immediately before the following,

```c++
  template<class T>
    concept @_possibly-packed-inout-matrix_@ = @_see below_@;   // exposition only
```

> add the following.

```c++
  template<class T>
    concept @_possibly-packed-out-matrix_@ = @_see below_@;   // exposition only
```

Then, to **[linalg.helpers.concepts]**, immediately before the following,

```c++
template<class T>
  concept @_possibly-packed-inout-matrix_@ =
    is-mdspan<T> && T::rank() == 2 &&
    is_assignable_v<typename T::reference, typename T::element_type> &&
    (T::is_always_unique() || is-layout-blas-packed<typename T::layout_type>);
```

> add the following definition of the exposition-only concept _`possibly-packed-out-matrix`_.

```c++
template<class T>
  concept @_possibly-packed-out-matrix_@ =
    is-mdspan<T> && T::rank() == 2 &&
    is_assignable_v<typename T::reference, typename T::element_type> &&
    (T::is_always_unique() || is-layout-blas-packed<typename T::layout_type>);
```

## Rank-k update functions in synopsis

> In the header `<linalg>` synopsis **[linalg.syn]**,
> change the following

```c++
  // rank-k symmetric matrix update
  template<class Scalar, @_in-matrix_@ InMat, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void symmetric_matrix_rank_k_update(Scalar alpha, InMat A, InOutMat C, Triangle t);
  template<class ExecutionPolicy, class Scalar,
           @_in-matrix_@ InMat, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void symmetric_matrix_rank_k_update(ExecutionPolicy&& exec,
                                        Scalar alpha, InMat A, InOutMat C, Triangle t);

  template<@_in-matrix_@ InMat, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void symmetric_matrix_rank_k_update(InMat A, InOutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix InMat_@, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void symmetric_matrix_rank_k_update(ExecutionPolicy&& exec,
                                        InMat A, InOutMat C, Triangle t);

  // rank-k Hermitian matrix update
  template<class Scalar, @_in-matrix_@ InMat, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void hermitian_matrix_rank_k_update(Scalar alpha, InMat A, InOutMat C, Triangle t);
  template<class ExecutionPolicy,
           class Scalar, @_in-matrix_@ InMat, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void hermitian_matrix_rank_k_update(ExecutionPolicy&& exec,
                                        Scalar alpha, InMat A, InOutMat C, Triangle t);

  template<@_in-matrix_@ InMat, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void hermitian_matrix_rank_k_update(InMat A, InOutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat, @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void hermitian_matrix_rank_k_update(ExecutionPolicy&& exec,
                                        InMat A, InOutMat C, Triangle t);
```

> to read as follows.

```c++
  // overwriting rank-k symmetric matrix update
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

  // updating rank-k symmetric matrix update
  template<class Scalar,
           @_in-matrix_@ InMat1,
           @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy, class Scalar,
           @_in-matrix_@ InMat1,
           @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      ExecutionPolicy&& exec, Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<@_in-matrix_@ InMat1,
           @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void symmetric_matrix_rank_k_update(
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1,
           @_in-matrix_@ InMat2,
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
           @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy, class Scalar,
           @_in-matrix_@ InMat1,
           @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      ExecutionPolicy&& exec, Scalar alpha,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<@_in-matrix_@ InMat1,
           @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      InMat1 A, InMat2 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1,
           @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat,
           class Triangle>
    void hermitian_matrix_rank_k_update(
      ExecutionPolicy&& exec,
      InMat1 A, InMat2 E, OutMat C, Triangle t);
```

## Rank-2k update functions in synopsis

> In the header `<linalg>` synopsis **[linalg.syn]**,
> change the following

```c++
  // rank-2k symmetric matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(InMat1 A, InMat2 B, InOutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, InOutMat C, Triangle t);

  // rank-2k Hermitian matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(InMat1 A, InMat2 B, InOutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-inout-matrix_@ InOutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, InOutMat C, Triangle t);
```

> to read as follows.

```c++
  // overwriting rank-2k symmetric matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(InMat1 A, InMat2 B, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, OutMat C, Triangle t);

  // updating rank-2k symmetric matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2, @_in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2, @_in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void symmetric_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);

  // overwriting rank-2k Hermitian matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(InMat1 A, InMat2 B, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, OutMat C, Triangle t);

  // updating rank-2k Hermitian matrix update
  template<@_in-matrix_@ InMat1, @_in-matrix_@ InMat2, @_in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);
  template<class ExecutionPolicy,
           @_in-matrix_@ InMat1, @_in-matrix_@ InMat2, @_in-matrix_@ InMat3,
           @_possibly-packed-out-matrix_@ OutMat, class Triangle>
    void hermitian_matrix_rank_2k_update(ExecutionPolicy&& exec,
                                         InMat1 A, InMat2 B, InMat3 E, OutMat C, Triangle t);
```

## Specification of rank-k update functions

> Replace the entire contents of [linalg.algs.blas3.rankk] with the following.

<i>[Note:</i> These functions correspond to the BLAS functions
`xSYRK` and `xHERK`. <i>-- end note]</i>

[1]{.pnum} The following elements apply to all functions in [linalg.algs.blas3.rankk].

[2]{.pnum} *Mandates:*

  * [2.1]{.pnum} If `OutMat` has `layout_blas_packed` layout, then the
      layout's `Triangle` template argument has the same type as
      the function's `Triangle` template argument.

  * [2.2]{.pnum} _`possibly-multipliable`_`<decltype(A), decltype(transposed(A)), decltype(C)>` is `true`.

  * [2.3]{.pnum} _`possibly-addable`_`<decltype(C), decltype(E), decltype(C)>` is `true` for those overloads that take an `E` parameter.

[3]{.pnum} *Preconditions:*

  * [3.1]{.pnum} _`multipliable`_`(A, transposed(A), C)` is `true`.  <i>[Note:</i> This implies that `C` is square. <i>-- end note]</i>

  * [3.2]{.pnum} _`addable`_`(C, E, C)` is `true` for those overloads that take an `E` parameter.

[4]{.pnum} *Complexity:* $O($ `A.extent(0)` $\cdot$ `A.extent(1)` $\cdot$ `A.extent(0)` $)$.

[5]{.pnum} *Remarks:* `C` may alias `E` for those overloads that take an `E` parameter.

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
where the scalar $\alpha$ is `alpha`.

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
         @_in-matrix_@ InMat2,
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
         @_in-matrix_@ InMat2,
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
where the scalar $\alpha$ is `alpha`.

```c++
template<@_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void symmetric_matrix_rank_k_update(
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
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
         @_in-matrix_@ InMat2,
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
         @_in-matrix_@ InMat2,
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
where the scalar $\alpha$ is `alpha`.

```c++
template<@_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
         @_possibly-packed-out-matrix_@ OutMat,
         class Triangle>
void hermitian_matrix_rank_k_update(
  InMat1 A,
  InMat2 E,
  OutMat C,
  Triangle t);
template<class ExecutionPolicy,
         @_in-matrix_@ InMat1,
         @_in-matrix_@ InMat2,
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

[2]{.pnum} *Mandates:*

  * [2.1]{.pnum} If `OutMat` has `layout_blas_packed` layout, then the
      layout's `Triangle` template argument has the same type as
      the function's `Triangle` template argument;

  * [2.2]{.pnum} _`possibly-multipliable`_`<decltype(A), decltype(transposed(B)), decltype(C)>` is `true`.

  * [2.3]{.pnum} _`possibly-multipliable`_`<decltype(B), decltype(transposed(A)), decltype(C)>` is `true`.

  * [2.4]{.pnum} _`possibly-addable`_`<decltype(C), decltype(E), decltype(C)>` is `true` for those overloads that take an `E` parameter.

[3]{.pnum} *Preconditions:*

  * [3.1]{.pnum} _`multipliable`_`(A, transposed(B), C)` is `true`.

  * [3.2]{.pnum} _`multipliable`_`(B, transposed(A), C)` is `true`.  <i>[Note:</i> This and the previous imply that `C` is square. <i>-- end note]</i>

  * [3.3]{.pnum} _`addable`_`(C, E, C)` is `true` for those overloads that take an `E` parameter.

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
