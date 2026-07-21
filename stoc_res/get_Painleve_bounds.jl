using IntervalArithmetic, SpecialFunctions, Arblib, Serialization

function enlarge(x::Interval{BigFloat})
    isguaranteed(x) || error("interval is not guaranteed")
    return interval(BigFloat(inf(x), RoundDown), BigFloat(sup(x), RoundUp))
end


SpecialFunctions.besseli(ν::Interval{BigFloat},z::Interval{BigFloat}) = interval(besseli(Arb(ν), Arb(z)))
SpecialFunctions.besseli(ν::Rational,z::Interval{BigFloat}) = interval(besseli(interval(BigFloat, ν), z))

function compute_b1(κ)
    κ̂ = κ^2/interval(8)
    num = κ^2*(besseli(-1//4, κ̂)+besseli(3//4, κ̂)+besseli(5//4, κ̂))+(interval(4)+κ^2)*besseli(1//4, κ̂)
    denom = interval(2)*κ*(besseli(-1//4, κ̂)+besseli(1//4, κ̂))
    return num/denom
end

# functions f and g involved in the definition of the operator S
g(x, y, κ, n) = (x + y - κ)/sqrt(n)/interval(2)
f(x) = -x + sqrt(interval(1) + x^2)

function get_cs(κ, cert, N₀)
    # initailise some values for c⁺ and c⁻
    c⁺ = I"1.02"
    c⁻ = I"0.987"

    # functions to evaluate inital guess for b₊ and b₋
    B⁻(n) = sqrt(n/interval(3))*c⁻
    B⁺(n) = sqrt(n/interval(3))*c⁺

    # check asymptotic condition (Step 3)
    N₂ = 2000000
    cond1 = sqrt(interval(12)+(-sqrt(interval(3)/interval(N₂))*κ+c⁺*(sqrt(interval(1)-interval(1)/interval(N₂))+sqrt(interval(1)+interval(1)/interval(N₂))))^2)-interval(2)*c⁺-interval(2)*c⁻
    cond2 = interval(2)*c⁺-(sqrt(interval(12)+(interval(2)*c⁻)^2)+sqrt(interval(3)/interval(N₂))*κ-c⁻*(sqrt(interval(1)-interval(1)/interval(N₂))+sqrt(interval(1)+interval(1)/interval(N₂))))
    if !(inf(cond1)>0 && inf(cond2)>0)
        println("Asymptotic condition not satisfied for N₂ = ", N₂)
        cert = false
    end

    b⁻ = B⁻.(interval.(1:N₂))
    b⁺ = B⁺.(interval.(1:N₂))

    Sb⁻ = f.(g.(vcat(interval(0.0),b⁻[1:N₂-1]), vcat(b⁻[2:N₂], B⁻(interval(N₂+1))), interval(Float64, κ), interval.(1:N₂))).*sqrt.(interval.(1:N₂))
    Sb⁺ = f.(g.(vcat(interval(0.0),b⁺[1:N₂-1]), vcat(b⁺[2:N₂], B⁺(interval(N₂+1))), interval(Float64, κ), interval.(1:N₂))).*sqrt.(interval.(1:N₂))
    N₃ = argmax(cumsum((inf.(b⁺-Sb⁻).≤ 0) .||(inf.(Sb⁺-b⁻) .≤ 0)))

    # if N₃ = N₂, cannot conclude on self-mapping
    if N₃ == N₂
        println("N₃ = N₂")
        cert = false
    end

    # compute elements of the Painlevé recursion
    b₁ = compute_b1(κ)
    b₀ = interval(BigFloat, 0.0)

    b = zeros(Interval{BigFloat}, N₃+11)
    b[1:2] = [b₀, b₁]
    for n = 2:(N₃+10)
        b[n+1] = κ+(interval(BigFloat, n-1)/b[n])-b[n]-b[n-1]
    end

    b = interval.(Float64, b[2:end])
    
    # find minimum bound N₁ such that sqrt bound is satisfied for all n ≥ N₁ (Step 5)
    N₁ = argmax(cumsum((inf.(b⁺[1:N₃]-b[1:N₃]).<=0) .|| (inf.(b[1:N₃]-b⁻[1:N₃]).<=0)))+1

    # first rank where sqrt bound is not satisfied
    N₄ = argmax(cumsum((inf.(b⁺-Sb⁻).<=0) .||(inf.(Sb⁺-b⁻).<=0)))

    # construction of self-mapping compact K as by ε-inflation (Step 4)
    while sum((inf.(b⁺-Sb⁻).<=0) .||(inf.(Sb⁺-b⁻).<=0)) >0
  
        for i in N₄:-1:1
            while inf(b⁺[i]-Sb⁻[i]) <= 0.0
                b⁺[i] *= I"1.01"
            end
        end
        Sb⁺ = f.(g.(vcat(interval(0.0),b⁺[1:N₂-1]), vcat(b⁺[2:N₂], B⁺(interval(N₂+1))), interval(Float64, κ), interval.(collect(1:N₂)))).*sqrt.(interval.(1:N₂))
        N₄ = argmax(cumsum((inf.(b⁺-Sb⁻).<=0) .||(inf.(Sb⁺-b⁻).<=0)))
        for i in N₄:-1:1
            while inf(Sb⁺[i]-b⁻[i]) <= 0.0
                b⁻[i] *= I"0.999"
            end
        end
        Sb⁻ = f.(g.(vcat(interval(0.0),b⁻[1:N₂-1]), vcat(b⁻[2:N₂], B⁻(interval(N₂+1))), interval(Float64, κ), interval.(collect(1:N₂)))).*sqrt.(interval.(1:N₂))
        N₄ = argmax(cumsum((inf.(b⁺-Sb⁻).<=0) .||(inf.(Sb⁺-b⁻).<=0)))
    end

    # decrease c⁻ until bound is satisfied
    while true
        if any(inf.(b[N₀:N₁+5]) .< sup.(B⁻.(interval.(collect(N₀:N₁+5)))))
            c⁻ *= I"0.999"
        else
            break
        end
    end

    # increase c⁺ until bound is satisfied
    while true
        if any(inf.(B⁺.(interval.(collect(N₀:N₁+5)))) .< sup.(b[N₀:N₁+5]))
            c⁺ *= I"1.001"
        else
            break
        end
    end
    return c⁺, c⁻, cert
end
