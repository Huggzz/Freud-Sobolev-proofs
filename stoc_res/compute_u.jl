Base.:*(A::SparseMatrixCSC{Interval{Float64}, Int64},x::Vector{Interval{Float64}})=Vector((A*sparse(x[:,:]))[:])
Base.:*(A::SparseMatrixCSC{Complex{Interval{Float64}}, Int64},x::Vector{Complex{Interval{Float64}}})=Vector((A*sparse(x[:,:]))[:])

LinearAlgebra.norm(v::Vector) = sqrt(sum(abs2.(v)))

function compute_ū_Gv̄(η̃,κ̃,ω̃, M, N)

    # Computation of the Painlevé coefficients bₖ (and of aₖ = \sqrt{bₖ}) 
    b₀ = interval(BigFloat, 0.0)
    b₁ = compute_b1(κ̃)
    b = zeros(Interval{BigFloat},2*N+3)
    b[1:2] = [b₀, b₁]
    for k = 2:2*N+2
        b[k+1] = κ̃+(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1] # k/bₖ = bₖ₊₁ + bₖ + bₖ₋₁ - κ̃       
    end

    setprecision(128)
    b = enlarge.(b[2:end])
    a = sqrt.(b)

    # Computation of the coefficients α, β, λ and μ that go into the operators Lₘ and Jₘ
    α = interval.(BigFloat, collect(1:2*N+2))./a[1:2*N+2] # αₖ = k/aₖ
    β = a[1:2*N].*a[2:2*N+1].*a[3:2*N+2] # βₖ = aₖ*aₖ₊₁*aₖ₊₂
    λ = zeros(Interval{BigFloat},2*N+2) # λₖ = αₖ² + βₖ₋₂² (note that β₁=β₂=0)
    λ[1:2] = α[1:2].^2
    λ[3:2*N+2] = α[3:2*N+2].^2 + β[1:2*N].^2
    μ = α[1:2*N].*β[1:2*N] # μₖ = αₖ*βₖ 

    function compute_J(m::Int64, N)
        # computes the operators J₁ and J₂ defined in Appendix E.1
        if mod1(m, 2) == 1
            return -BandedMatrix(0 => α[2:2:2*N], -1 => β[2:2:2*N-2])
        else
            return -BandedMatrix(-1 => α[3:2:2*N-1], -2 => β[3:2:2*N-3])
        end
    end

    Js = [compute_J(1, N), compute_J(2, N)]

    function compute_L(m::Int64)
        # computes the operators Lₘ defined in Appendix E.1
        if mod1(m, 2) == 1
            return SymTridiagonal(interval(im*m)*ω̃ .+ λ[1:2:2*N-1], μ[1:2:2*N-3])
        else
            return SymTridiagonal(interval(im*m)*ω̃ .+ λ[2:2:2*N], μ[2:2:2*N-2])
        end
    end

    Ls = [compute_L(m) for m in -M:M]

    rows = ones(Int64, 2*M+3)*N
    cols = [N-mod1(m, 2) for m=-M:M]

    # Construction of a finite dimensional projection of F̃ = F*(L+ω∂ₛ) which is block tridiagonal. 
    # The diagonal blocks themselves are tridiagonal, whereas the upper and lower diagonal blocks are lower triangular, with bandwith 1 or 2 depending on the parity of m)
    F̃ = BandedBlockBandedMatrix(Zeros(Complex{BigFloat}, sum(rows), sum(cols)), rows, cols, (2, 0), (2, 1))
    for m in -M:M
        F̃[Block(m+M+2, m+M+1)] = mid.(Ls[m+M+1][:,1:cols[m+M+1]])
        F̃[Block(m+M+1, m+M+1)] = mid.(η̃/I"2"*Js[mod1(m,2)][:,1:cols[m+M+1]])
        F̃[Block(m+M+3, m+M+1)] = mid.(η̃/I"2"*Js[mod1(m,2)][:,1:cols[m+M+1]])
    end
    F̃ = sparse(F̃)
    # The right-hand side g of the equation F̃u = F*v = g
    g = BlockVector(zeros(Interval{BigFloat}, sum(rows)), rows)
    g[Block(M+1)[1:2]] = [α[1],β[1]]/I"2"*η̃ 
    g[Block(M+3)[1:2]] = [α[1],β[1]]/I"2"*η̃

    fF̃ = ComplexF64.(sparse(F̃))
    fg = collect(mid.(g))

    # initial guess for ū
    ū = big.(fF̃\Float64.(fg))
    # refine ū with iterative refinement (still nonrigorous but improves the residual)
    for i=1:3
        r = fg - F̃*ū
        y = fF̃\ComplexF64.(r)
        ū += big.(y)
    end

    ū = BlockVector(interval.(ū), cols)

    # impose that ū is real valued
    ū[Block(M+1)] = real.(ū[Block(M+1)])
    for m = 1:M
        ū[Block(m+M+1)] = conj.(ū[Block(-m+M+1)])
    end


    # initialise residual vector G(v̄) = F(v̄) - g (now rigorously)
    Gv̄ = BlockVector(zeros(Complex{Interval{BigFloat}}, sum(rows)), rows)
    Gv̄[:] = - g

    for m = -M:M
        Gv̄[Block(m+M+1)] += η̃/I"2"*Js[mod1(m,2)][:,1:N-mod1(m, 2)]*ū[Block(m+M+1)]
        Gv̄[Block(m+M+2)] += Ls[m+M+1][:,1:N-mod1(m, 2)]*ū[Block(m+M+1)]
        Gv̄[Block(m+M+3)] += η̃/I"2"*Js[mod1(m,2)][:,1:N-mod1(m, 2)]*ū[Block(m+M+1)]
    end

    Gv̄ = interval.(Float64, Gv̄)
    setprecision(512)
    return mid.(ū), Gv̄
end