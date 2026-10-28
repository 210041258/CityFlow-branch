 MATHEMATICAL DOCUMENTATION: TRAFFIC GROWTH MODEL COMPARISON
This documentation outlines the mathematical formulas, physical expressions, and boundary characteristics of the four traffic growth models .
1. Logistic (Mechanistic) Model
Mathematical Expression:
y^(t)=L1+e−k(t−x0) 
Parameters:
L : Carrying Capacity (limiting ceiling representing physical lane saturation:  L≈302.23  vehicles).
k : Logistic growth rate coefficient ( k≈0.0143 ).
x0 : Inflection point where the growth rate transitions from acceleration to deceleration ( x0≈101.54  frames).
Physical Significance: Perfectly models capacity-constrained, self-limiting biological or physical growth profiles. It features zero growth at  t=−∞ , max growth speed at  t=x0 , and asymptotically approaches carrying capacity  L  as  t→∞ .
2. Polynomial (Degree 2) Model
Mathematical Expression:
y^(t)=at2+bt+c 
Parameters:
a,b,c : Fitted empirical polynomial coefficients.
Physical Significance: An empirical statistical fit that captures quadratic acceleration. While highly flexible locally, it lacks physical constraints, ultimately leading to unbounded quadratic divergence ( ∥y(t)∥→∞  as  t→∞ ).
3. Polynomial (Degree 1) Model
Mathematical Expression:
y^(t)=mt+c 
Parameters:
m : Constant linear growth slope.
c : Linear intercept (baseline background vehicle count).
Physical Significance: A basic first-order regression baseline. It completely fails to capture the initial exponential acceleration phase as well as the late-stage capacity saturation profile.
4. Exponential Model
Mathematical Expression:
y^(t)=aebt+c 
Parameters:
a : Scale coefficient.
b : Exponential growth speed multiplier.
c : Baseline vertical offset.
Physical Significance: Models pure unconstrained multiplicative growth. Highly physically unrealistic for traffic systems as it neglects structural bottle-necks and physical queue limitations, leading to rapid mathematical overflow.
