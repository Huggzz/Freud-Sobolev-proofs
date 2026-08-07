# Sparse matrix multiplication overloads
Base.:*(A::SparseMatrixCSC{Interval{Float64}, Int64},x::Vector{Interval{Float64}})=Vector((A*sparse(x[:,:]))[:])
Base.:*(A::SparseMatrixCSC{Complex{Interval{Float64}}, Int64},x::Vector{Complex{Interval{Float64}}})=Vector((A*sparse(x[:,:]))[:])

# Enlarge interval with guaranteed bounds
function enlarge(x::Interval{BigFloat})
    isguaranteed(x) || error("interval is not guaranteed")
    return interval(BigFloat(inf(x), RoundDown), BigFloat(sup(x), RoundUp))
end

LinearAlgebra.norm(v::Vector) = sqrt(sum(abs2.(v)))

# Compute operator norm (spectral norm)
function op_norm(A)
    if size(A) == (2,2)
        # 2×2 case: use analytical formula
        # ||A||² = (Frob(A)² + √(Frob(A)⁴ - 4det(A)²)) / 2
        frob2 = sum(abs2.(A))
        det2 = abs2(A[1,1]*A[2,2]-A[2,1]*A[1,2])
        Z = sqrt((frob2 + sqrt(max((frob2^2-I"4"*det2), interval(0))))/I"2")
        return Z
    elseif size(A)[1] == 1 || size(A)[2] == 1
        # Vector case: norm is just the Euclidean norm
        return norm(vec(A))
    elseif size(A)[1] == 2
        # 2×n matrix: ||A|| = √(||AA'||)
        return sqrt(op_norm(A*A'))
    elseif size(A)[2] == 2
        # n×2 matrix: ||A|| = √(||A'A||)
        return sqrt(op_norm(A'*A))
    else
        # General case: eigenvalue decomposition approach
        # Compute A'A and eigendecompose its midpoint
        B = A'A
        Λ̄, V̄ = eigen(Hermitian(mid.(B)))
        # Convert to interval and verify bounds are guaranteed
        Λ = inv(interval.(V̄))*B*interval.(V̄)
        all(isguaranteed.(Λ)) || error("matrix not guaranteed")
        # Upper bound: add error estimate from off-diagonal perturbations
        σ̄ = sup(sqrt(maximum(abs.(diag(Λ) + interval(-1,1)*[sum(abs.(Λ[i,1:i-1]))+sum(abs.(Λ[i,i+1:end])) for i=1:size(B)[1]]))))
        # Lower bound: use power iteration on eigenvector
        σ̲ = inf(norm(A*interval.(V̄)[:,end])/norm(interval.(V̄)[:,end]))
        return interval(σ̲, σ̄)
    end
end

# Fast bound on operator norm for larger matrices via ℓ₁ and ℓ∞ norms
function lazy_op_norm(A)
    if size(A)[1] == 1 || size(A)[2] == 1 || size(A)[1] == 2 || size(A)[2] == 2
        return op_norm(A)
    else
        all(isguaranteed.(A)) || error("matrix not guaranteed")
        return interval(0,sqrt(interval(maximum(sup.(sum(abs.(A), dims = 1))))*interval(maximum(sup.(sum(abs.(A), dims = 2))))))
    end
end

# Bessel function extensions for interval arguments
SpecialFunctions.besseli(ν::Interval{BigFloat},z::Interval{BigFloat}) = interval(besseli(Arb(ν), Arb(z)))
SpecialFunctions.besseli(ν::Rational,z::Interval{BigFloat}) = interval(besseli(interval(BigFloat, ν), z))

# Compute first Painlevé coefficient b₁
function compute_b1(κ̃)
    κ̂ = κ̃^2/interval(8)
    num = κ̃^2*(besseli(-1//4, κ̂)+besseli(3//4, κ̂)+besseli(5//4, κ̂))+(interval(4)+κ̃^2)*besseli(1//4, κ̂)
    denom = interval(2)*κ̃*(besseli(-1//4, κ̂)+besseli(1//4, κ̂))
    return num/denom
end

# Multiply lower banded interval matrix A with dense complex interval matrix B
function Base.:*(A::BandedMatrix{Interval{Float64}, Matrix{Interval{Float64}}, Base.OneTo{Int64}}, B::Matrix{Complex{Interval{Float64}}})
    (A.u ≤ 0) || error("only implemented for lower banded matrices")
    m = size(A)[1]
    n = size(B)[2]
    C = zeros(Complex{Interval{Float64}}, (m,n))
    # Extract each diagonal of A and multiply with corresponding block of B
    # A has A.u upper diagonals and A.l lower diagonals
    for k in max(0,-A.u):1:A.l
        # Extract diagonal k (offset by -k) and multiply with B rows
        C[k+1:end,:] += diag(A,-k).*B[1:end-k,:]
    end
    return C
end

# Multiply dense complex interval matrix A with lower banded interval matrix B
function Base.:*(A::Matrix{Complex{Interval{Float64}}}, B::BandedMatrix{Interval{Float64}, Matrix{Interval{Float64}}, Base.OneTo{Int64}})
    (B.u ≤ 0) || error("only implemented for lower banded matrices")
    m = size(A)[1]
    n = size(B)[2]
    C = zeros(Complex{Interval{Float64}}, (m,n))
    # Extract each diagonal of B and multiply with corresponding columns of A
    for k in max(0,-B.u):1:B.l
        # Extract diagonal k (offset by -k) and multiply with A columns
        C[:,1:end-k] += A[:,k+1:end].*transpose(diag(B, -k))
    end
    return C
end

# Multiply dense complex interval matrix A with symmetric tridiagonal complex interval matrix B
function Base.:*(A::Matrix{Complex{Interval{Float64}}}, B::SymTridiagonal{Complex{Interval{Float64}}, Vector{Complex{Interval{Float64}}}})
    # B has diagonal B.dv and superdiagonal B.ev (symmetric structure)
    # Initialize result with diagonal part: A .* B.dv (row-wise broadcast)
    C = A .* transpose(B.dv)
    # Add contribution from superdiagonal B.ev (upper off-diagonal)
    # Multiply A[:,2:end] by B.ev, place in columns 1:end-1
    C[:,1:end-1] += A[:,2:end].*transpose(B.ev)
    # Add contribution from subdiagonal (equal to superdiagonal by symmetry)
    # Multiply A[:,1:end-1] by B.ev, place in columns 2:end (every other column)
    C[:,2:end] += A[:,1:end-1].*transpose(B.ev)
    return C
end

# Invert complex interval bidiagonal matrix using forward/back substitution
function Base.inv(P::Bidiagonal{Complex{Interval{Float64}}, Vector{Complex{Interval{Float64}}}})
    if P.uplo == 'U'
        # Upper bidiagonal: use back substitution
        C = -P.ev./P.dv[1:end-1]
        invC = UpperTriangular(zeros(eltype(P), size(P)))
        for i in 1:size(P)[1]
            invC[i,i:end] = cumprod([one(eltype(P)); C[i:end]])
        end
        return UpperTriangular(invC ./ transpose(P.dv))
    else
        # Lower bidiagonal: use forward substitution (via transpose)
        return LowerTriangular(transpose(inv(transpose(P))))
    end
end


function compute_δ(c⁺, c⁻, η̃, κ̃, ω̃, cert, M, N, Gv̄)
    """
        Compute the final error bound δ = Y/(1-Z) for ||v-v̄||
    """
    # enlarge parameter intervals
    η̃ = interval(Float64, η̃)
    κ̃ = enlarge(κ̃)
    ω̃ = interval(Float64, ω̃)

    # constants in Appendix E.2
    C_α = interval(3*((N₀+1)//N₀)^3)^I"1//4"/sqrt(c⁻)
    C_β = sqrt(c⁺)^3*interval((N₀+1)*(N₀+2)*(N₀+3)//(3*N₀)^3)^I"1//4"
    C_λ = sqrt(I"3")/c⁺ + sqrt(interval((N₀-1)*(N₀-2)//N₀^2))*c⁻^3/sqrt(I"3")^3
    C_μ = c⁺/sqrt(I"3")*interval((N₀+1)*(N₀+2)//N₀^2)^I"1//4"

    K = 500 # truncation level for ξ_N

    # compute Painlevé coefficients
    b₀ = interval(BigFloat, 0.0)
    b₁ = compute_b1(κ̃)
    b = zeros(Interval{BigFloat},2*N+2*K+3)
    b[1:2] = [b₀, b₁]
    for k = 2:2*N+2*K+2
        b[k+1] = κ̃+(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1]
    end
    b = interval.(Float64, b[2:end]);
    a = sqrt.(b)

    # sequences defined in Appendix E.1
    α = interval.(collect(1:2*N+2*K+2))./a[1:2*N+2*K+2]
    β = a[1:2*N+2*K].*a[2:2*N+2*K+1].*a[3:2*N+2*K+2]
    λ = zeros(Interval{Float64},2*N+2*K+2)
    λ[3:2*N+2*K+2] = α[3:2*N+2*K+2].^2 + β[1:2*N+2*K].^2
    λ[1:2] = α[1:2].^2
    μ = α[1:2*N+2*K].*β[1:2*N+2*K]
    μ² = μ.^2
 
    function compute_d(m)
        # computes the sequence (dₖ) defined Appendix E.1 for a given m
        d² = zeros(Complex{Interval{Float64}}, 2*N+2*K+2)
        d²[1:2] = interval(im*m)*ω̃ .+ λ[1:2]
        for k=3:2*N+2*K+2
            d²[k] = interval(im*m)*ω̃ .+ λ[k]-μ²[k-2]/d²[k-2]
        end
        return sqrt.(d²)
    end
    ds = [compute_d(m) for m in -M:M];
    
    # χ and γ constants in Appendix E.2
    χ = C_μ/C_λ
    if sup(χ) ≥ 1//2
        cert = false
    end
    γ = interval(1//2) + sqrt(interval(1//4)-χ^2)
    C_d = sqrt(γ*C_λ)
    ϑ = C_μ/C_d^2
    C_αd = C_α/C_d
    C_βd = C_β/C_d

    # checks that the base case of the induction in Appendix E.2 holds
    if !all([isstrictless(γ, abs(ds[m+M+1][2*N-mod1(m,2)]^2/(interval(im*m)*ω̃+λ[2*N-mod1(m,2)])))  for m in -M:M])
        cert = false
    end

    # constants in calculations of Zⁱʲₖₗ (parts which do not depend on j)
    Z¹²_fact = (η̃/I"2")/(C_d^2*(interval(2*N)*(I"1"-ϑ^2))^I"3//2")
    Z²¹_fact = η̃*(C_αd +C_βd)/(I"2"*(I"1"-ϑ))
    Z²² = η̃*(C_αd +C_βd)/(I"2"*(I"1"-ϑ)^2*C_d*interval(2*N-2)^I"3//4")

    function compute_J(m::Int64, N)
        # computes the operators J₁ and J₂ defined in Appendix E.1
        if mod1(m, 2) == 1
            return -BandedMatrix(0 => α[2:2:2*N], -1 => β[2:2:2*N-2])
        else
            return -BandedMatrix(-1 => α[3:2:2*N-1], -2 => β[3:2:2*N-3])
        end
    end

    Js = [compute_J(1, N), compute_J(2, N)];

    function compute_U(m::Int64, d, N)
        # computes the upper triangular matrix U defined in Appendix E.1
         if mod1(m, 2) == 1
            return Bidiagonal(d[1:2:2*N-1], μ[1:2:2*N-3]./d[1:2:2*N-3], :U)
        else
            return Bidiagonal(d[2:2:2*N], μ[2:2:2*N-2]./d[2:2:2*N-2], :U)
        end
    end


    function compute_L(m::Int64)
        # computes the operators Lₘ defined in Appendix E.1
        if mod1(m, 2) == 1
            return SymTridiagonal(interval(im*m)*ω̃ .+ λ[1:2:2*N-1], μ[1:2:2*N-3])
        else
            return SymTridiagonal(interval(im*m)*ω̃ .+ λ[2:2:2*N], μ[2:2:2*N-2])
        end
    end
    function compute_inv_L(U, ξ)
        # computes the finite dimensional projection of the inverse of the operator Lₘ
        A = inv(U)
        return A*transpose(A)+ξ*(A[:,end]*transpose(A[:,end]))
    end

    function compute_L̃(m::Int64, U, ξ)
        # computes the matrix L̃ₘ¹¹ defined in Appendix E.3
        if mod1(m, 2) == 1
            v = interval(im*m)*ω̃ .+ λ[1:2:2*N-1]
            v[end] -= (ξ/(interval(1)+ξ))*U[end,end]^2
            return SymTridiagonal(v, μ[1:2:2*N-3])
        else
            v = interval(im*m)*ω̃ .+ λ[2:2:2*N]
            v[end] -= (ξ/(interval(1)+ξ))*U[end,end]^2
            return SymTridiagonal(v, μ[2:2:2*N-2])
        end
    end

    function compute_ξ(m, N, d)
        # computes a certified enclosure of the constant ξ_N defined at the end of Appendix E.3
        ϵ = (μ[2*N+mod1(m,2)-2]/abs(d[2*N+mod1(m,2)-2])/C_d)^2/interval(2*N+2*K)^I"3//2"*ϑ^interval(2*K)/(I"1"- ϑ^2)
        if mod1(m, 2) == 1
            ξ̄ = (μ[2*N-1]/d[2*N-1])^2*sum([complex(interval(1.0)); cumprod(μ[2*N+1:2:2*N+2*K-3].^2 ./d[2*N+1:2:2*N+2*K-3].^4)]./d[2*N+1:2:2*N+2*K-1].^2)
        else
            ξ̄ = (μ[2*N]/d[2*N])^2*sum([complex(interval(1.0)); cumprod(μ[2*N+2:2:2*N+2*K-2].^2 ./d[2*N+2:2:2*N+2*K-2].^4)]./d[2*N+2:2:2*N+2*K].^2)
        end
        return ξ̄ + interval(-1,1)*ϵ + interval(-im,im)*ϵ
    end

    function compute_I(m, M, N)
        # computes a block column of the identify matrix
        rows = ones(Int64, 2*M+1)*N
        cols = [N]
        A = BandedBlockBandedMatrix(Zeros(Float64, sum(rows), sum(cols)), rows, cols, ((m+M), -(m+M)), (0, 0))
        A[Block(m+M+1, 1)] = BandedMatrix(I(N))
        return A
    end

    function bound_B(m, cert)
        # computes an ℓ₂ bound on the operator Bₘ defined in Section 5.3 using bounds from Appendix E.4
        d = compute_d(m)
        if !isstrictless(γ, abs(d[2*N-mod1(m,2)]^2/(interval(im*m)*ω̃+λ[2*N-mod1(m,2)])))
            cert = false
        end
        U = compute_U(m, d, N)
        U⁻¹ = inv(U)
        ξ_N = compute_ξ(m, N, d)
        ξ_Nm = compute_ξ(m, N-mod1(m,2), d)
        L⁻¹ = compute_inv_L(U, ξ_N)
        B = ((η̃/interval(2))*Js[mod1(m, 2)])*L⁻¹
        normB¹¹ = op_norm(B[:,1:end - mod1(m,2)])
        # println(normB¹¹)
        normB¹²₁ = op_norm(B[:,end - mod1(m,2)+1:end])
        normB¹²₂ = Z¹²_fact*abs(μ[2*N+mod1(m,2)-2]/d[2*N+mod1(m,2)-2])*norm(Js[mod1(m, 2)]*U⁻¹[:,end])
        # normB¹² = norm([normB¹²₁, normB¹²₂])
        normB¹² = normB¹²₁ + normB¹²₂
        normB²¹ = Z²¹_fact*sqrt(abs(ξ_Nm))*norm(U⁻¹[1:N-mod1(m,2),N-mod1(m,2)])
        normB = op_norm([normB¹¹ normB¹²;
            normB²¹ Z²²])
        # println(normB¹¹)
        return normB, cert
    end

    # bound first the ℓ₂ norm of Bₘ for m = -M-2 and m = -M-3 which bounds the ℓ₂ norm of
    # all the Bₘ for |m| ≥ M+1 by monotonicity by a Lemma in Section 5.4
    Z_B1, cert = bound_B(-M-2, cert)
    Z_B2, cert = bound_B(-M-3, cert)
    Z_B1 *= I"2"
    Z_B2 *= I"2"
    # println("Z_B1 = ", Z_B1)
    # println("Z_B2 = ", Z_B2)

    # compute all finite dimensional projections of all the operators
    Us = [compute_U(m, ds[m+M+1], N) for m in -M:M];
    inv_Us = [inv(U) for U in Us];
    ξ_Ns = [compute_ξ(m, N, ds[m+M+1]) for m in -M:M]
    ξ_Nms = [compute_ξ(m, N - mod1(m,2), ds[m+M+1]) for m in -M:M]
    L̃s = [compute_L̃(m, Us[m+M+1], ξ_Ns[m+M+1]) for m in -M:M];
    inv_Ls = [compute_inv_L(Us[m+M+1], ξ_Ns[m+M+1]) for m in -M:M];

    # computes operator norms involved in the faster estimaton of id-A¹¹F̄¹¹ (see Appendix E.3)
    L̃_norms = [op_norm(L) for L in L̃s];
    inv_L_norms = [op_norm(inv_Ls[m+M+1][:, 1:end - mod1(m,2)]) for m in -M:M];
    inv_L_norms_m = [op_norm(inv_Ls[m+M+1][:, end - mod1(m,2)+1:end]) for m in -M:M];

    rows = cols = ones(Int64, 2*M+1)*N

    # Constructs the matrix F̃¹¹ defined in Appendix E.3 (banded block tridiagonal matrix)
    F̃¹¹ = BandedBlockBandedMatrix(Zeros(Complex{Interval{Float64}}, sum(rows), sum(cols)), rows, cols, (1, 1), (2, 1))
    for m in -M:M
        # println(m)
        F̃¹¹[Block(m+M+1, m+M+1)] = L̃s[m+M+1]
        if m > -M
            F̃¹¹[Block(m+M, m+M+1)] = η̃/I"2"*Js[mod1(m,2)]
        end
        if m < M
            F̃¹¹[Block(m+M+2, m+M+1)] =  η̃/I"2"*Js[mod1(m,2)]
        end
    end
    col = [N]

    # compute LU factorization of F̃¹¹ to solve F̃¹¹x = y for multiple right hand sides efficiently
    fF̃¹¹ = lu(mid.(sparse(F̃¹¹)))

    # compute the block columns Ãₘ of the matrix Ã¹¹ defined in Appendix E.3 and the products ÃₘJ
    # (We drop the ¹¹ superscript in the sequel)
    # Here m = -M (but this will be reused with different m in the loop below) 
    Ãₘ₋₁J = zeros(Complex{Interval{Float64}}, sum(rows), sum(col))
    Ãₘ = interval.(fF̃¹¹\Matrix(compute_I(-M,M,N)))
    ÃₘJ = Ãₘ*Js[mod1(-M-1,2)]
    Ãₘ₊₁ = interval.(fF̃¹¹\Matrix(compute_I(-M+1,M,N)))
    Ãₘ₊₁J = Ãₘ₊₁*Js[mod1(-M,2)]

    # snippet allowing to implement the symmetry relation Ã₋ₘ = conj.(Ãₘ) (see Remark at the end of Appendix E.3)
    # This is only used for computing the Y bound, not the Z bound. We still have m = -M here
    Ãₘ_block = BlockArray(Ãₘ, rows, col)
    Ã₋ₘ = zeros(Complex{Interval{Float64}}, sum(rows), sum(col))
    Ã₋ₘ_block = BlockArray(Ã₋ₘ, rows, col)
    for i=-M:M
        Ã₋ₘ_block[Block(i+M+1, 1)] = conj.(Ãₘ_block[Block(M+1-i, 1)]) 
    end
    Ã₋ₘ = Matrix(Ã₋ₘ_block)

    # compute norm of block column of ĀF̄ - id for m = -M-1 (using appendix E.3)
    d = compute_d(-M-1)
    U = compute_U(-M-1, d, N)
    U⁻¹ = inv(U)
    ξ_N = compute_ξ(-M-1, N, d)
    ξ_Nm = compute_ξ(-M-1, N-mod1(-M-1,2), d)
    L⁻¹ = compute_inv_L(U, ξ_N)
    ÃₘBₘ₋₁ = BlockArray(η̃/I"2"*ÃₘJ*L⁻¹, rows, col)

    Z_left, cert = bound_B(-M-1, cert)

    for i=1:2*M+1
        AᵢₘBₘ₋₁ = L̃s[i]*ÃₘBₘ₋₁[Block(i,1)]
        Zᵢₘ¹¹ = lazy_op_norm(AᵢₘBₘ₋₁[:,1:end - mod1(-M-1,2)])
        if sup.(Zᵢₘ¹¹) > 0.000001
            Zᵢₘ¹¹ = op_norm(AᵢₘBₘ₋₁[:,1:end - mod1(-M-1,2)])

        end
        temp₁ = op_norm(AᵢₘBₘ₋₁[:,end - mod1(-M-1,2)+1:end])
        ÃₘJU⁻¹ = BlockVector(ÃₘJ*(U⁻¹[:,end]), rows)
        temp₂  = abs(μ[2*N+mod1(-M-1,2)-2]/d[2*N+mod1(-M-1,2)-2])*norm(L̃s[i]*ÃₘJU⁻¹[Block(i)])/(C_d^2*(interval(2*N)*(I"1"-ϑ^2))^I"3//2")
        # Zᵢₘ¹² = norm([temp₁, temp₂])
        Zᵢₘ¹² = temp₁ + temp₂
        if i==1
            # the Zᵢₘ²¹ formula given in the paper can be rewritten in terms of ξ
            Zᵢₘ²¹ = Z²¹_fact*sqrt(abs(ξ_Nm))*norm(U⁻¹[1:N-mod1(-M-1,2),N-mod1(-M-1,2)])
            Zᵢₘ = op_norm([Zᵢₘ¹¹ Zᵢₘ¹²;
                    Zᵢₘ²¹ Z²²])
        else
            Zᵢₘ = norm([Zᵢₘ¹¹, Zᵢₘ¹²])
        end
        Z_left += Zᵢₘ

    end

    # intialise norms of block columns of ĀF̄ - id
    # each entry Zₘs[m+M+1] will be an upper bound for ∑ᵢ Zᵢₘ 
    Zₘs = zeros(Interval{Float64}, M+1)

    # initialise ÃGv̄ (which will be used to compute the Y bound)
    ÃGv̄ = BlockVector(zeros(Complex{Interval{Float64}}, sum(rows)), rows)
    # ÃJ = zeros(Complex{Interval{Float64}}, size(Ãₘ₋₁J))
    # ÃJU⁻¹ = BlockVector(zeros(Complex{Interval{Float64}}, sum(rows)), rows)
    # Rₘ = BlockArray(zeros(Complex{Interval{Float64}}, sum(rows), sum(col)), rows, col)
    # now iterate on m (only need to do half the columns thanks to symmetry)
    for m in -M:0
        # apply Ãₘ to the residue
        ÃGv̄ += Ãₘ*(Gv̄[Block(M+1+m)])
        if m ≠ 0
            # take into account the symmetry
            ÃGv̄ += Ã₋ₘ*(Gv̄[Block(M+1-m)])
        end

        # intialise ℓ² norm of blocks. Zₘ will contain all the Zᵢₘ
        Zₘ = zeros(Interval{Float64}, 2*M+1)

        ÃJ = Ãₘ₋₁J + Ãₘ₊₁J
        ÃJU⁻¹ = BlockVector(ÃJ*(inv_Us[m+M+1][:,end]), rows)

        # Residue defined in Appendix E.3
        Rₘ = interval.(compute_I(m,M,N))-BlockArray(η̃/I"2"*ÃJ+Ãₘ*L̃s[m+M+1], rows, col)
        # we now compute each Zᵢₘ
        for i=1:2*M+1
            norm_Rᵢₘ = lazy_op_norm(Rₘ[Block(i,1)])
            Zᵢₘ¹¹ = norm_Rᵢₘ*L̃_norms[i]*inv_L_norms[m+M+1]
            temp₁ = norm_Rᵢₘ*L̃_norms[i]*inv_L_norms_m[m+M+1]

            temp₂ = Z¹²_fact*abs(μ[2*N+mod1(m,2)-2]/ds[m+M+1][2*N+mod1(m,2)-2])*norm(L̃s[i]*ÃJU⁻¹[Block(i)])
            # Zᵢₘ¹² = norm([temp₁, temp₂])
            Zᵢₘ¹² = temp₁ + temp₂
            if (i == m+M) || (i == m+M+2)
                # the Zᵢₘ²¹ formula given in the paper can be rewritten in terms of ξ
                Zᵢₘ²¹ = Z²¹_fact*sqrt(abs(ξ_Nms[m+M+1]))*norm(inv_Us[m+M+1][1:N-mod1(m,2),N-mod1(m,2)])
                Zᵢₘ = op_norm([Zᵢₘ¹¹ Zᵢₘ¹²;
                    Zᵢₘ²¹ Z²²])
            else
                Zᵢₘ = norm([Zᵢₘ¹¹, Zᵢₘ¹²])
            end

            Zₘ[i] = Zᵢₘ
        end
        Zₘs[m+M+1] = sum(Zₘ)
       
        if abs(m) == M
            # in this case need to add the bound on Bₘ (left above ĀF̄ - id)
            temp, cert = bound_B(m, cert)
            Zₘs[m+M+1] += temp
        end

        # update the block columns Ãₘ₋₁, Ãₘ, Ãₘ₊₁ etc. for the next iteration
        if m ≤ -1
            # println("testing m = ", m)
            # Ãₘ₋₁ = copy(Ãₘ)
            Ãₘ = copy(Ãₘ₊₁)
            Ãₘ₋₁J = copy(ÃₘJ)
            ÃₘJ = copy(Ãₘ₊₁J)

            if m+1 == 0
                # in this case, need to impose the symmetry
                Ãₘ₊₁ = copy(Ã₋ₘ)
            else
                # otherwise, solve for the next block column
                Ãₘ₊₁ = interval.(fF̃¹¹\Matrix(compute_I(m+2,M,N)))
            end
            Ãₘ₊₁J = Ãₘ₊₁*Js[mod1(m+1,2)]

            # define the opposite block column via the symmetry
            Ãₘ_block = BlockArray(Ãₘ, rows, col)
            for i=-M:M
                Ã₋ₘ_block[Block(i+M+1, 1)] = conj.(Ãₘ_block[Block(M+1-i, 1)])
            end
            Ã₋ₘ = Matrix(Ã₋ₘ_block)

        end
        GC.gc()
    end

    # final Z
    Z = max(Z_B1, Z_B2, Z_left, maximum(Zₘs))
    println("Z = ", Z)

    # initialise Y = ||AFv̄||
    Y = interval(0)
    for m = -M:M
        Y += sqrt(sum(abs2.(L̃s[m+M+1]*ÃGv̄[Block(m+M+1)])))
    end

    # final error bound δ as in Section 5.2
    cert = cert && (sup(Z) < 1)
    δ = Y/(I"1"-Z)
    return δ, cert
end