using IntervalArithmetic, Polynomials, PolynomialRoots, Serialization, LinearAlgebra

function enlarge(x::Interval{BigFloat})
    isguaranteed(x) || error("interval is not guaranteed")
    return interval(BigFloat(inf(x), RoundDown), BigFloat(sup(x), RoundUp))
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

n = 2500
N = 2*n+2
setprecision(8192*16)
b₁ = deserialize("b1_quad4")
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},N+1)
b[1:2] = [b₀, b₁]
for n = 2:N
    b[n+1] = interval(BigFloat, 4)+(interval(BigFloat, n-1)/(interval(BigFloat, 2)*b[n]))-b[n]-b[n-1]
end

prec = 8192*2
setprecision(prec)
a = enlarge.(sqrt.(b)[2:end])

x = Polynomial(interval.(big.([0,1])))
Pprev = Polynomial(interval.(big.([1]))./enlarge(sqrt(deserialize("GP_quad_Z4"))))
P = Polynomial([interval(BigFloat, 0), interval(1)/a[1]]./enlarge(sqrt(deserialize("GP_quad_Z4"))))

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
if all(sign.(eval_poly(p, Xl)).*sign.(eval_poly(p, Xu)) .==-1) && all(sup.(Xl[1:end-1]).< inf.(Xu[2:end])) && length(Xl) == N÷2
    println("enclosure of roots checked")
else
    println("enclosure of roots failed")
end

GC.gc()

Xrig = interval.(inf.(Xl), sup.(Xu));

# check a index below
# quadrature wrt to the normalised measure
W = (interval(2)/a[end])./(eval_poly(dp, Xrig).*eval_poly(pprev, Xrig).*Xrig)/deserialize("GP_Z")

b₁ = deserialize("b1_k4")
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},n+3)
b[1:2] = [b₀, b₁]
for k = 2:n+2
    b[k+1] = interval(4)+(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1]
end

setprecision(8192)
a = enlarge.(sqrt.(b)[2:end])
Yrig = enlarge.(Xrig)

qprev = enlarge.(W.^interval(1//4))
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

n = 2500
N = 3*n+2
setprecision(8192*32)
b₁ = deserialize("b1_quad6")
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},N+1)
b[1:2] = [b₀, b₁]
for n = 2:N
    b[n+1] = interval(BigFloat, 4)+(interval(BigFloat, n-1)/(interval(BigFloat, 3)*b[n]))-b[n]-b[n-1]
end

prec = 8192*4
setprecision(prec)
a = enlarge.(sqrt.(b)[2:end])

x = Polynomial(interval.(big.([0,1])))
Pprev = Polynomial(interval.(big.([1]))./enlarge(sqrt(deserialize("GP_quad_Z6"))))
P = Polynomial([interval(BigFloat, 0), interval(1)/a[1]]./enlarge(sqrt(deserialize("GP_quad_Z6"))))

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
if all(sign.(eval_poly(p, Xl)).*sign.(eval_poly(p, Xu)) .==-1) && all(sup.(Xl[1:end-1]).< inf.(Xu[2:end])) && length(Xl) == N÷2
    println("enclosure of roots checked")
else
    println("enclosure of roots failed")
end

GC.gc()

Xrig = interval.(inf.(Xl), sup.(Xu));

# check a index below
W = (interval(2)/a[end])./(eval_poly(dp, Xrig).*eval_poly(pprev, Xrig).*Xrig)/deserialize("GP_Z")

b₁ = deserialize("b1_k4")
b₀ = interval(BigFloat, 0.0)
b = zeros(Interval{BigFloat},n+3)
b[1:2] = [b₀, b₁]
for k = 2:n+2
    b[k+1] = interval(4)+(interval(BigFloat, k-1)/b[k])-b[k]-b[k-1]
end

setprecision(8192*2)
a = enlarge.(sqrt.(b)[2:end])
Yrig = enlarge.(Xrig)

qprev = enlarge.(W.^interval(1//6))
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

