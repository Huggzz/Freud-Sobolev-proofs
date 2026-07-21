using IntervalArithmetic, Polynomials, PolynomialRoots, Serialization, LinearAlgebra, Arblib, SpecialFunctions

function enlarge(x::Interval{BigFloat})
    isguaranteed(x) || error("interval is not guaranteed")
    return interval(BigFloat(inf(x), RoundDown), BigFloat(sup(x), RoundUp))
end


SpecialFunctions.besseli(ν::Interval{BigFloat},z::Interval{BigFloat}) = interval(besseli(Arb(ν), Arb(z)))
SpecialFunctions.besseli(ν::Rational,z::Interval{BigFloat}) = interval(besseli(interval(BigFloat, ν), z))

function compute_M0(a, b)
    # computes ∫exp(-ax^4/4+bx^2/2)dx
    κ̂ = b^2/(interval(8)*a)
    return interval(BigFloat, π)*sqrt(b/a)*exp(κ̂)*(besseli(-1//4, κ̂)+besseli(1//4, κ̂))/I"2"
end

function compute_M2(a, b)
    # computes ∫x^2exp(-ax^4/4+bx^2/2)dx
    κ̂ = b^2/(interval(8)*a)
    return (b^2*(besseli(-1//4, κ̂)+besseli(3//4, κ̂)+besseli(5//4, κ̂))+(I"4"*a+b^2)*besseli(1//4, κ̂))*exp(κ̂)*interval(BigFloat, π)/(I"4"*sqrt(a^3*b))
end


function eval_poly(q, Y)
    n = length(Y)
    M = Threads.nthreads()
    K = length(Y)÷M+1
    ind = [1+K*m for m=0:M]
    X = [Y[ind[m]:min(ind[m+1]-1,length(Y))] for m=1:M]
    Threads.@threads for i=1:Threads.nthreads()
        x = X[i]
        X[i] = q.(x)
    end
    return reduce(vcat,X)
end

println("four-product quadrature rule:")

m = 4
κ = interval(BigFloat, 4)
n = 2500
N = 2*n+2
setprecision(8192*16)
a = interval(BigFloat, m//2)
b = κ*a
b₁ = compute_M2(a, b)/compute_M0(a, b)
Z = compute_M0(a, b)
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},N+1)
b[1:2] = [b₀, b₁]
for n = 2:N
    b[n+1] = κ+(interval(BigFloat, n-1)/(interval(BigFloat, m//2)*b[n]))-b[n]-b[n-1]
end

prec = 8192*2
setprecision(prec)
a = enlarge.(sqrt.(b)[2:end])

x = Polynomial(interval.(big.([0,1])))
Pprev = Polynomial(interval.(big.([1]))./enlarge(sqrt(Z)))
P = Polynomial([interval(BigFloat, 0), interval(1)/a[1]]./enlarge(sqrt(Z)))

for i=2:N
    global P, Pprev = (x*P - a[i-1]*Pprev)/a[i], P
end

p = Polynomial(P.coeffs[1:2:end])
dp = Polynomial((interval.(big.(collect(1:N))).*((P.coeffs)[2:end]))[2:2:end])
pprev = Polynomial(Pprev.coeffs[2:2:end])

Pf = Polynomial(mid.(p.coeffs));
dPf = derivative(Pf);

J = SymTridiagonal(zeros(N), Float64.(mid.(a))[1:end])
X = eigvals(J)
X = (big.(X[X .>= 0.0])).^2

# setprecision(8192)
M = Threads.nthreads()
K = (N÷2)÷M+1
ind = [1+K*m for m=0:M]
Y = [BigFloat.(X[ind[m]:min(ind[m+1]-1,N÷2)]) for m=1:M]
Threads.@threads for i=1:Threads.nthreads()
    x = copy(Y[i])
    dx = ones(BigFloat, length(x))
    while maximum(abs.(dx))> big(2)^(-precision(BigFloat)//2)
        # println((i,Float64(log2(maximum(abs.(dx))))))
        indices = abs.(dx) .> big(2)^(-precision(BigFloat)//2)
        dx[indices] .= Pf.(x[indices])./dPf.(x[indices])
        x[indices] .-= dx[indices]
    end
    Y[i] = x
end

GC.gc()

X = reduce(vcat,Y)
ϵ = big(2)^(-prec//2)
Xl = interval.(BigFloat, X .- ϵ)
Xu = interval.(BigFloat, X .+ ϵ)


# check enclosure via intermediate value Thm and fundamental Thm of Algebra
if all(sign.(eval_poly(p, Xl)).*sign.(eval_poly(p, Xu)) .==-1) && all(sup.(Xu[1:end-1]).< inf.(Xl[2:end])) && length(Xl) == N÷2
    println("enclosure of roots checked")
else
    println("enclosure of roots failed")
end

GC.gc()

Xrig = interval.(inf.(Xl), sup.(Xu));

# check a index below
# quadrature wrt to the normalised measure
W = (interval(2)/a[end])./(eval_poly(dp, Xrig).*eval_poly(pprev, Xrig).*Xrig)/compute_M0(interval(BigFloat, 1), κ)

b₁ = compute_M2(interval(BigFloat, 1), κ)/compute_M0(interval(BigFloat, 1), κ)
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},n+3)
b[1:2] = [b₀, b₁]
for k = 2:n+2
    b[k+1] = κ+(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1]
end

setprecision(8192)
a = enlarge.(sqrt.(b)[2:end])
Yrig = enlarge.(Xrig)

qprev = enlarge.(W.^interval(1//m))
q = (Yrig .- a[1].^2).*qprev/(a[1]*a[2])
V = zeros(Interval{BigFloat}, (length(Xrig), n÷2+1))
V[:,1] .= copy(qprev)

# Only even polynomials are computed
for i=2:2:n
    # println(i)
    global q , qprev = ((Yrig .-(a[i+1]^2+a[i]^2)).*q - (a[i]*a[i-1])*qprev)/(a[i+1]*a[i+2]), q
    V[:,i÷2+1] .= copy(qprev)
end

setprecision(1024)
serialize("GP_V4", enlarge.(V))

V = 0
GC.gc()

println("six-product quadrature rule:")

m = 6
n = 2500
N = 3*n+2
setprecision(8192*32)
a = interval(BigFloat, m//2)
b = κ*a
b₁ = compute_M2(a, b)/compute_M0(a, b)
Z = compute_M0(a, b)
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},N+1)
b[1:2] = [b₀, b₁]
for n = 2:N
    b[n+1] = κ+(interval(BigFloat, n-1)/(interval(BigFloat, m//2)*b[n]))-b[n]-b[n-1]
end


prec = 8192*4
setprecision(prec)
a = enlarge.(sqrt.(b)[2:end])

x = Polynomial(interval.(big.([0,1])))
Pprev = Polynomial(interval.(big.([1]))./enlarge(sqrt(Z)))
P = Polynomial([interval(BigFloat, 0), interval(1)/a[1]]./enlarge(sqrt(Z)))

for i=2:N
    global P, Pprev = (x*P - a[i-1]*Pprev)/a[i], P
end

p = Polynomial(P.coeffs[1:2:end])
dp = Polynomial((interval.(big.(collect(1:N))).*((P.coeffs)[2:end]))[2:2:end])
pprev = Polynomial(Pprev.coeffs[2:2:end])

Pf = Polynomial(mid.(p.coeffs));
dPf = derivative(Pf);

J = SymTridiagonal(zeros(N), Float64.(mid.(a))[1:end])
X = eigvals(J)
X = (big.(X[X .>= 0.0])).^2

# setprecision(8192)
M = Threads.nthreads()
K = (N÷2)÷M+1
ind = [1+K*m for m=0:M]
Y = [BigFloat.(X[ind[m]:min(ind[m+1]-1,N÷2)]) for m=1:M]
Threads.@threads for i=1:Threads.nthreads()
    x = copy(Y[i])
    dx = ones(BigFloat, length(x))
    while maximum(abs.(dx))> big(2)^(-precision(BigFloat)//2)
        # println((i,Float64(log2(maximum(abs.(dx))))))
        indices = abs.(dx) .> big(2)^(-precision(BigFloat)//2)
        dx[indices] .= Pf.(x[indices])./dPf.(x[indices])
        x[indices] .-= dx[indices]
    end
    Y[i] = x
end

GC.gc()

X = reduce(vcat,Y)
ϵ = big(2)^(-prec//2)
Xl = interval.(BigFloat, X .- ϵ)
Xu = interval.(BigFloat, X .+ ϵ)


# check enclosure via intermediate value Thm and fundamental Thm of Algebra
if all(sign.(eval_poly(p, Xl)).*sign.(eval_poly(p, Xu)) .==-1) && all(sup.(Xu[1:end-1]).< inf.(Xl[2:end])) && length(Xl) == N÷2
    println("enclosure of roots checked")
else
    println("enclosure of roots failed")
end

GC.gc()

Xrig = interval.(inf.(Xl), sup.(Xu));

# check a index below
W = (interval(2)/a[end])./(eval_poly(dp, Xrig).*eval_poly(pprev, Xrig).*Xrig)/compute_M0(interval(BigFloat, 1), κ)

b₁ = compute_M2(interval(BigFloat, 1), κ)/compute_M0(interval(BigFloat, 1), κ)
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},n+3)
b[1:2] = [b₀, b₁]
for k = 2:n+2
    b[k+1] = κ +(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1]
end

setprecision(8192*2)
a = enlarge.(sqrt.(b)[2:end])
Yrig = enlarge.(Xrig)

qprev = enlarge.(W.^interval(1//m))
q = (Yrig .- a[1].^2).*qprev/(a[1]*a[2])
setprecision(1024)
V = zeros(Interval{BigFloat}, (length(Xrig), n÷2+1))
V[:,1] .= enlarge.(qprev)
setprecision(8192*2)

for i=2:2:n
    # println(i)
    global q , qprev = ((Yrig .-(a[i+1]^2+a[i]^2)).*q - (a[i]*a[i-1])*qprev)/(a[i+1]*a[i+2]), q
    setprecision(1024)
    V[:,i÷2+1] .= enlarge.(qprev)
    setprecision(8192*2)
end

setprecision(1024)
serialize("GP_V6", enlarge.(V))

