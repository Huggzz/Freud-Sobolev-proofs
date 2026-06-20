using LinearAlgebra, BlockBandedMatrices, IntervalArithmetic, SpecialFunctions, Arblib, BandedMatrices, BlockArrays, SparseArrays, IterativeSolvers, Serialization, Base.Threads, Random, Plots, LaTeXStrings
setprecision(8192*16)
include("get_enclosure.jl")
include("get_Painleve_bounds.jl")
include("compute_u.jl")
include("compute_CP.jl")

# Fixed parameter values (before rescaling)
κ = interval(BigFloat, 1)
η = interval(BigFloat, 12//100)
ω = interval(BigFloat, 1//1000)

# the different value of σ (before rescaling) for which we want to compute the bounds
σs = interval.(BigFloat, 2//10:5//1000:8//10)

# truncation levels for A
N = 75 # in space (we take N even modes and N odd modes, so 2N in total)
Ms = deserialize("Ms") # in time

N₀ = 2*N-4 # threshold for the Painlevé bounds
# Note that 2N-2 ≥ N₀, hence all the conditions of the form [m]+2N-2 ≥ N₀ in the paper are automatically satisfied.

cols = [[N-mod1(m, 2) for m=-(M-1):M-1] for M in Ms] # size of ū (for each m)
rows = [ones(Int64, 2*M+1)*N for M in Ms] # size of G(v̄) := F(v̄) - g (for each m)

σN = length(σs)
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
Threads.@threads for i=shuffle(collect(1:σN))
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
Threads.@threads for k = shuffle(collect(1:σN))
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
Rs = zeros(Interval{Float64}, σN)
for i in 1:σN
    σ = σs[i]
    setprecision(8192*16)
    κ̃ = κ/σ*sqrt(interval(BigFloat, 2))
    ū = ūs[i]
    M = Ms[i]
    # get Poincaré constant C_P.
    Cₚ = compute_CP(κ̃)
    setprecision(256)
    # initialise fourier coefficients of the mean of ρ̄(t, x)
    ūcos = zeros(Interval{BigFloat}, length(M+1:2:2*M-1))
    ūsin = zeros(Interval{BigFloat}, length(M+1:2:2*M-1))
    for (k,m) in enumerate(M+1:2:2*M-1)
        ūcos[k] = I"2"*interval(real(ū[Block(m)][1]))
        ūsin[k] = I"2"*interval(imag(ū[Block(m)][1]))
    end
    # scaling factor to get the mean of ρ̄(t, x) following Appendix E.5
    scaling = sqrt(compute_b1(κ̃))*sqrt(σ/sqrt(interval(BigFloat, 2)))

    function eval_mean(t)
        # evaluates the mean of ρ̄(t, x) given a time t (possibly an interval)
        return scaling*(sum(ūcos.*cos.(interval.(1:2:M-1)*t))+sum(ūsin.*sin.(interval.(1:2:M-1)*t)))
    end
    function eval_dmean(t)
        # evaluates the derivative of the mean of ρ̄(t, x) given a time t (possibly an interval)
        return scaling*(-sum(ūcos.*interval.(1:2:M-1).*sin.(interval.(1:2:M-1)*t))+sum(ūsin.*interval.(1:2:M-1).*cos.(interval.(1:2:M-1)*t)))
    end

    # intialise interval containing argmax of the absolute value of the mean of ρ̄(t, x)
    t̄ = interval(BigFloat, -π, π)/I"2"
    
    # enclose all the zeros of the derivative of the mean by successive n-section of the interval t̄
    # We expect the mean of ρ̄(t, x) is unimodal on [-π/2, π/2], which is why the easy procedure below does indeed yield a narrow interval in practice
    while diam(t̄) > 1e-20
        # n-section of the interval t̄
        ts = mince(t̄, 200)
        # evaluate the derivative of the mean on each subinterval of the n-section
        dfts = eval_dmean.(ts)
        # find the subintervals of t̄ where the derivative of the mean contains 0
        indmax = in_interval.(0, dfts)
        tmin = minimum(inf.(ts[indmax]))
        tmax = maximum(sup.(ts[indmax]))
        
        # refine interval t̄
        t̄ = interval(tmin, tmax)
    end

    # evaluate the mean of ρ̄  on the interval t̄ (which contains all the ts where the derivative could vanish)
    Rs[i] = interval(Float64, abs.(eval_mean.(t̄)) + interval(-1,1)*Cₚ*sqrt(σ*compute_b1(κ̃))/I"2"^I"1//4"*δs[i])
end

serialize("Rsigmas", Dict("sigmas" => σs, "Rs" => Rs))

println("maximum radius = ", nextfloat((maximum(diam.(Rs)))/2))
scatter(mid.(σs), mid.(Rs), xlabel = L"$\sigma$", ylabel = L"$\mathcal{R}(\sigma)$", label = false, dpi = 500, markersize = 2)
png("indicator")