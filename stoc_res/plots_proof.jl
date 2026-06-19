using LinearAlgebra, BlockBandedMatrices, IntervalArithmetic, SpecialFunctions, Arblib, BandedMatrices, BlockArrays, SparseArrays, IterativeSolvers, Serialization, Base.Threads, Random
# nthreads() = 6
setprecision(8192*16)
include("get_enclosure.jl")
include("get_Painleve_bounds.jl");
include("compute_u.jl");
include("compute_CP.jl");
include("compute_CJ.jl");

function compute_Z(κ̃)
    κ̂ = κ̃^2/interval(8)
    return exp(κ̂)*sqrt(κ̃)interval(BigFloat,π)/interval(2)*(besseli(-1//4, κ̂)+besseli(1//4, κ̂))
end

# Fixed parameter values (before rescaling)
κ = interval(BigFloat, 1)
η = interval(BigFloat, 12//100)
ω = interval(BigFloat, 1//1000)

# the different value of σ (before rescaling) for which we want to compute the bounds
σs = interval.(BigFloat, [2//10, 287129152//1000000000, 8//10])
# σs = interval.(BigFloat, 2//10:5//1000:4//10)

# truncation levels for A
N = 100 # in space (we take N even modes and N odd modes, so 2N in total)
Ms = [1401, 1001, 1001]# in time
N₀ = 2*N-4 # threshold for the Painlevé bounds
# Note that 2N-2 ≥ N₀, hence all the conditions of the form [m]+2N-2 ≥ N₀ in the paper are automatically satisfied.

cols = [[N-mod1(m, 2) for m=-(M-1):M-1] for M in Ms] # size of ū (for each m)
rows = [ones(Int64, 2*M+1)*N for M in Ms] # size of G(v̄) := F(v̄) - g (for each m)

σN = length(σs)
# σN = 10 # for testing
certs = ones(Bool, σN) # flag that will be put to false if something goes wrong during the proof
setprecision(512)
ūs = [BlockVector(zeros(Complex{BigFloat}, sum(cols[i])), cols[i]) for i=1:σN]
# G(v̄) := F(v̄) - g
Gv̄s = [BlockVector(zeros(Complex{Interval{Float64}}, sum(rows[i])), rows[i]) for i=1:σN] 
# constants c⁺ and c⁻ for the Painlevé bounds
c⁺s = zeros(Interval{Float64}, σN)
c⁻s = zeros(Interval{Float64}, σN)
δs = zeros(Interval{Float64}, σN)

# compute the approximate solution ū, and the corresponding residual F(v̄) - g, for each value of σ
IntervalArithmetic.configure_matmul(:slow) # because IntervalArithmetic.jl does not like the fast option with BigFloats

# loop cannot be multithreaded due to global change in precision
for i in 1:σN
    GC.gc()
    σ = σs[i]
    M = Ms[i]
    # rescaling the parameters (and changing the precision via enlarge)
    κ̃ = κ/σ*sqrt(interval(BigFloat, 2))
    η̃ = enlarge(η/(σ^interval(3//2))*(interval(BigFloat, 2)^interval(3//4)))
    ω̃ = enlarge(ω/σ*sqrt(interval(BigFloat, 2)))
    # computation of ū (this is nonrigorous, we just want a good approximation) and of F(v̄) - g (this is done rigorously, we get an interval enclosure)
    ū, Gv̄ = compute_ū_Gv̄(η̃,κ̃,ω̃, M-1, N)
    ūs[i] = ū
    Gv̄s[i] = Gv̄ 
    GC.gc()
end
GC.gc()

# compute Painlevé bounds
setprecision(8192*16)
for i=1:σN
    σ = σs[i]
    # rescaling the parameters (and changing the precision via enlarge)
    κ̃ = κ/σ*sqrt(interval(BigFloat, 2))
    η̃ = enlarge(η/(σ^interval(3//2))*(interval(BigFloat, 2)^interval(3//4)))
    ω̃ = enlarge(ω/σ*sqrt(interval(BigFloat, 2)))
    cert = true
    # computation of the Painlevé bounds c⁺ and c⁻, valid for all n ≥ N₀
    c⁺, c⁻, cert = get_cs(κ̃, cert, N₀) 
    GC.gc()
    certs[i] = cert
    c⁺s[i] = c⁺
    c⁻s[i] = c⁻
end

GC.gc()

# compute the final error bound for ||v-v̄|| 
IntervalArithmetic.configure_matmul(:fast)
setprecision(8192)
for k = 1:σN
    σ = σs[k]
    M = Ms[k]
    # rescaling the parameters (and changing the precision via enlarge)
    κ̃ = κ/σ*sqrt(interval(BigFloat, 2))
    η̃ = enlarge(η/(σ^interval(3//2))*(interval(BigFloat, 2)^interval(3//4)))
    ω̃ = enlarge(ω/σ*sqrt(interval(BigFloat, 2)))
    cert = certs[k]
    Gv̄ = Gv̄s[k]
    c⁺ = c⁺s[k]
    c⁻ = c⁻s[k]
    # computation of the final error bound (the computation of Z₁, and the check that Z₁<1, happen here)
    δ, cert = compute_δ(c⁺, c⁻, η̃, κ̃, ω̃, cert, M, N, Gv̄)
    δs[k] = δ
    certs[k] = cert
    GC.gc()
end


# we now rigorously enclose the stochastic resonance indicator
setprecision(256)
sup_bounds = zeros(Interval{Float64}, σN)
for i in 1:σN
    σ = σs[i]
    setprecision(8192*16)
    κ̃ = κ/σ*sqrt(interval(BigFloat, 2))
    ū = ūs[i]
    M = Ms[i]
    Cₚ = compute_CP(κ̃)
    CJ = compute_CJ(κ̃)
    sup_bounds[i] = I"2"^I"1//4"*Cₚ^I"3//4"*sqrt((CJ+I"1")/(σ*compute_Z(κ̃)))*δs[i]
    println(I"2"^I"1//4"*Cₚ^I"3//4"*sqrt((CJ+I"1")/(σ*compute_Z(κ̃))))
    setprecision(256)
end

println("A uniform bound is given by: ", sup(maximum(sup_bounds)))