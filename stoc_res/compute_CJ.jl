# Invert complex interval bidiagonal matrix using forward/back substitution

function Base.:*(D::Diagonal, v::Vector)
    return diag(D).*v
end

function Base.:*(D::Diagonal, A::Matrix)
    return diag(D).*A
end

function Base.:*(A::Matrix, D::Diagonal)
    return A.*diag(D)'
end

function compute_CJ(κ)
    N = 200
    N₀ = N
    cert = true
    setprecision(8192*16)
    c⁺, c⁻, cert = get_cs(κ, cert, N₀)

    setprecision(4096)
    b₁ = compute_b1(κ)
    C_α = interval(3)^interval(1//4)/sqrt(c⁺)
    θ = c⁺^2*interval((1+1//N₀)*(1+2//N₀))^interval(1//4)/interval(3)
    CD = interval(1)/C_α/sqrt(interval(1)-θ^2);
    setprecision(precision(mid.(b₁)))
    b₀ = interval(BigFloat, 0.0)
    b = zeros(Interval{BigFloat},N+6)
    b[1:2] = [b₀, b₁]
    for k = 2:N+5
        b[k+1] = interval(κ)+(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1]
    end
    setprecision(256)
    b = b[2:end]
    a = enlarge.(sqrt.(b))
    b = enlarge.(b);

    # we first compute the bound on even indices
    α = interval.(Float64, interval.(collect(2:2:N+2))./a[2:2:N+2])
    β = interval.(Float64, a[2:2:N+2].*a[3:2:N+3].*a[4:2:N+4])
    P̄ = Bidiagonal([interval(1); α[1:end-1]],  [interval(0); β[1:end-2]], :U)
    P̄⁻¹ = Matrix(interval.(Float64, inv(P̄)))
    α = interval.(Float64, interval.(collect(1:2:N+2))./a[1:2:N+1])
    β = interval.(Float64, a[1:2:N+2].*a[2:2:N+2].*a[3:2:N+3])
    A = α.*P̄⁻¹
    B = β.*P̄⁻¹;
    c_α = op_norm([op_norm(A) β[end-1]*norm(A[:,end])*CD/interval(N)^interval(3//4) ;
        interval(0) sqrt(c⁺/c⁻)/(interval(1)-θ)*interval((N+3)//(N+2))^interval(3//4)])
    c_β = op_norm([op_norm(B) β[end-1]*norm(B[:,end])*CD/interval(N)^interval(3//4) ;
        interval(0) c⁺^2/(interval(1)-θ)*interval((N+3)//(N+2))^interval(3//4)/interval(3)])
    CJ₀ = c_α + c_β

    # We now compute the bound on even indices
    α = interval.(Float64, interval.(collect(1:2:N+3))./a[1:2:N+3])
    β = interval.(Float64, a[1:2:N+3].*a[2:2:N+4].*a[3:2:N+5])
    P̄ = Bidiagonal(α[1:end-1],  β[1:end-2], :U)
    P̄⁻¹ = Matrix(interval.(Float64, inv(P̄)))
    α = interval.(Float64, interval.(collect(2:2:N+2))./a[2:2:N+2])
    β = interval.(Float64, a[2:2:N+2].*a[3:2:N+3].*a[4:2:N+4])
    A = α.*P̄⁻¹
    B = β.*P̄⁻¹;
    c_α = op_norm([op_norm(A) β[end-1]*norm(A[:,end])*CD/interval(N+1)^interval(3//4) ;
        interval(0) sqrt(c⁺/c⁻)/(interval(1)-θ)*interval((N+4)//(N+3))^interval(3//4)])
    c_β = op_norm([op_norm(B) β[end-1]*norm(B[:,end])*CD/interval(N+1)^interval(3//4) ;
        interval(0) c⁺^2/(interval(1)-θ)*interval((N+4)//(N+3))^interval(3//4)/interval(3)])

    CJ₁ = c_α + c_β
    # we take the maximum of the bounds on odd and even indices
    CJ = max(CJ₀, CJ₁)
    return CJ
end