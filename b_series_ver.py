###Code for the B series verification of the Extended Stability Runge-Kutta scheme used 
## In the paper, this corresponds to the mathematical justification of the coefficients section
import sympy as sp

# Number of stages
S = 15

a_np= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0243586417803786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0258303808904268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0667956303329210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0140960387721938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0412105997557866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0149469583607297, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.414086419082813, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00395908281378477, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.480561088337756, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.319660987317690, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0.00668808071535874, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.0374638233561973, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.439499983548480, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.0327614907498598, 0.367805790222090, 0]]

b_np=[0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.006612457947210495, 0.21674686949693006, 0.0, 0.42264597549826616, 0.03276149074985981, 0.0330623263939421, 0.0009799086295048407]


# 1) Symbolic b_i
b_syms = sp.symbols(f"b0:{S}")        # b0, b1, …, b14
b_main = sp.symbols("b")              # the repeated value for b0..b7
# enforce b0=…=b7 = b_main
subs_b = { b_syms[i]: b_main for i in range(8) }

# 2) Symbolic alpha_j  for the subdiagonal a_{j+1,j}
alpha = sp.symbols(f"alpha0:{S-1}")   # alpha0 = a_{1,0}, alpha1 = a_{2,1}, …

# 3) Build A as a list of lists
A = [ [0]*S for _ in range(S) ]
for i in range(1, S):
    for j in range(i):
        if i == j+1:
            A[i][j] = alpha[j]        # free subdiagonal
        else:
            A[i][j] = b_syms[j]       # all other strictly lower = b_j

# 4) Nodes c_i = sum_{j<i} A[i][j]
c = [ sum(A[i][j] for j in range(i)) for i in range(S) ]

# 5) Define the four elementary B-series sums up to order 3
B1 = sum(b_syms[i] for i in range(S))
B2 = sum(b_syms[i]*c[i] for i in range(S))
B3 = sum(b_syms[i]*c[i]**2 for i in range(S))
B4 = sum(b_syms[i] * sum(A[i][j]*c[j] for j in range(S)) for i in range(S))

# 6) Substitute b0..b7 → b_main, leave b8..b14 distinct
B1r = sp.simplify(B1.subs(subs_b))
B2r = sp.simplify(B2.subs(subs_b))
B3r = sp.simplify(B3.subs(subs_b))
B4r = sp.simplify(B4.subs(subs_b))

# 7) (Optionally) rename b8..b14 to nicer symbols
for i in range(8, S):
    B1r = B1r.subs(b_syms[i], sp.symbols(f"b{i}"))
    B2r = B2r.subs(b_syms[i], sp.symbols(f"b{i}"))
    B3r = B3r.subs(b_syms[i], sp.symbols(f"b{i}"))
    B4r = B4r.subs(b_syms[i], sp.symbols(f"b{i}"))

# 8) Display the four B-series coefficients
print("B1 =", B1r)   # should simplify to 1
print("B2 =", B2r)   # should simplify to 1/2
print("B3 =", B3r)   # should simplify to 1/3
print("B4 =", B4r)   # should simplify to 1/6
zero = sp.Integer(0)
checks = [
    sp.simplify(B1r - 1),
    sp.simplify(B2r - sp.Rational(1,2)),
    sp.simplify(B3r - sp.Rational(1,3)),
    sp.simplify(B4r - sp.Rational(1,6)),
]

for i, chk in enumerate(checks, start=1):
    print(f"B{i} check:", chk)


check1 = sp.simplify( B1r - 1 )
check2 = sp.simplify( B2r - sp.Rational(1,2) )
check3 = sp.simplify( B3r - sp.Rational(1,3) )
check4 = sp.simplify( B4r - sp.Rational(1,6) )

# --- after you've built B1r,B2r,B3r,B4r as before ---
# suppose your numeric arrays are b_np and a_np from before
subs_map = {}

# 1) b_main = your numeric b (all eight equal weights)
subs_map[b_main] = b_np[0]

# 2) the distinct b8..b14:
for i in range(8, 15):
    subs_map[sp.symbols(f"b{i}")] = b_np[i]

# 3) the alpha_j = a_{j+1,j} entries:
for j in range(0, 14):
    subs_map[alpha[j]] = a_np[j+1][j]

# now evaluate
print("B1 numeric:", float(B1r.subs(subs_map)))
print("B2 numeric:", float(B2r.subs(subs_map)))
print("B3 numeric:", float(B3r.subs(subs_map)))
print("B4 numeric:", float(B4r.subs(subs_map)))
