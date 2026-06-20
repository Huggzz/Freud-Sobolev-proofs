# Invert complex interval bidiagonal matrix using forward/back substitution
function Base.inv(P::Bidiagonal{Interval{Float64}, Vector{Interval{Float64}}})
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

function compute_CP(κ)
    N = 200
    N₀ = N
    cert = true
    setprecision(8192*16)
    c⁺, c⁻, cert = get_cs(κ, cert, N₀)

    setprecision(4096)
    b₁ = compute_b1(κ)
    C_α = interval(3)^interval(1//4)/sqrt(c⁺)
    θ = c⁺^2*interval((1+1//N₀)*(1+2//N₀))^interval(1//4)/interval(3)
    setprecision(precision(mid.(b₁)))
    b₀ = interval(BigFloat, 0.0)
    b = zeros(Interval{BigFloat},N+4)
    b[1:2] = [b₀, b₁]
    for k = 2:N+3
        b[k+1] = interval(κ)+(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1]
    end
    setprecision(256)
    b = b[2:end]
    a = enlarge.(sqrt.(b))
    b = enlarge.(b)

    # we first compute the bound on odd indices
    α = interval.(Float64, interval.(collect(1:2:N+1))./a[1:2:N+1])
    β = interval.(Float64, a[1:2:N+1].*a[2:2:N+2].*a[3:2:N+3])
    A = Bidiagonal(α,  β[1:end-1], :U)
    A⁻¹ = Matrix(interval.(Float64, inv(A)))
    C₁₂ = interval(1)/(C_α*sqrt(interval(1)-θ^2))*β[end]*norm(A⁻¹[:,end])
    C₂₂ = interval(1)/(C_α*(interval(1)-θ))
    C̄ₚ₁ = op_norm(A⁻¹)^2
    supCₚ₁ = op_norm([interval(sqrt(sup(C̄ₚ₁))) C₁₂/interval(N+1)^interval(3//4);
            interval(0) C₂₂/interval(N+1)^interval(3//4)])^2;
    Cₚ₁ = interval(inf(C̄ₚ₁), sup(supCₚ₁))

    # we now compute the bound on even indices
    α = interval.(Float64, interval.(collect(2:2:N))./a[2:2:N])
    β = interval.(Float64, a[2:2:N].*a[3:2:N+1].*a[4:2:N+2])
    A = Bidiagonal(α,  β[1:end-1], :U)
    A⁻¹ = Matrix(interval.(Float64, inv(A)))
    C₁₂ = interval(1)/(C_α*sqrt(interval(1)-θ^2))*β[end]*norm(A⁻¹[:,end])
    C₂₂ = interval(1)/(C_α*(interval(1)-θ))
    C̄ₚ₀ = op_norm(A⁻¹)^2
    supCₚ₀ = op_norm([interval(sqrt(sup(C̄ₚ₀))) C₁₂/interval(N)^interval(3//4);
            interval(0) C₂₂/interval(N)^interval(3//4)])^2;
    Cₚ₀ = max(interval(inf(C̄ₚ₀), sup(supCₚ₀)));

    # we take the maximum of the bounds on odd and even indices
    Cₚ = max(Cₚ₁, Cₚ₀)
    return Cₚ
end