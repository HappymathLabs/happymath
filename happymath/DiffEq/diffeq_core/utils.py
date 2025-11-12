import sympy

sympy_assumptions = [
            "real",       # real
            "imaginary",  # imaginary
            "integer",    # integer
            "rational",   # rational
            "irrational", # irrational
            "positive",   # positive
            "negative",   # negative
            "nonpositive",# non-positive
            "nonnegative",# non-negative
            "even",       # even
            "odd",        # odd
            "prime",      # prime
            "finite",     # finite
            "infinite",   # infinite
            "complex",    # complex
        ]

def check_symbol_type(symbol):
    if symbol.is_real:
        return "real"
    elif symbol.is_imaginary:
        return "imaginary"
    elif symbol.is_integer:
        return "integer"
    elif symbol.is_rational:
        return "rational"
    elif symbol.is_irrational:
        return "irrational"
    elif symbol.is_positive:
        return "positive"
    elif symbol.is_negative:
        return "negative"
    elif symbol.is_nonpositive:
        return "nonpositive"
    elif symbol.is_nonnegative:
        return "nonnegative"
    elif symbol.is_even:
        return "even"
    elif symbol.is_odd:
        return "odd"
    elif symbol.is_prime:
        return "prime"
    elif symbol.is_finite:
        return "finite"
    elif symbol.is_infinite:
        return "infinite"
    elif symbol.is_complex:
        return "complex"
    else:
        return "unknown"

def forced_trans_type(org_symbol, new_symbol_type:str):
    if new_symbol_type == "real":
        return sympy.Symbol(str(org_symbol), real=True)
    elif new_symbol_type == "imaginary":
        return sympy.Symbol(str(org_symbol), imaginary=True)
    elif new_symbol_type == "integer":
        return sympy.Symbol(str(org_symbol), integer=True)
    elif new_symbol_type == "rational":
        return sympy.Symbol(str(org_symbol), rational=True)
    elif new_symbol_type == "irrational":
        return sympy.Symbol(str(org_symbol), irrational=True)
    elif new_symbol_type == "positive":
        return sympy.Symbol(str(org_symbol), positive=True)
    elif new_symbol_type == "negative":
        return sympy.Symbol(str(org_symbol), negative=True)
    elif new_symbol_type == "nonpositive":
        return sympy.Symbol(str(org_symbol), nonpositive=True)
    elif new_symbol_type == "nonnegative":
        return sympy.Symbol(str(org_symbol), nonnegative=True)
    elif new_symbol_type == "even":
        return sympy.Symbol(str(org_symbol), even=True)
    elif new_symbol_type == "odd":
        return sympy.Symbol(str(org_symbol), odd=True)
    elif new_symbol_type == "prime":
        return sympy.Symbol(str(org_symbol), prime=True)
    elif new_symbol_type == "finite":
        return sympy.Symbol(str(org_symbol), finite=True)
    elif new_symbol_type == "infinite":
        return sympy.Symbol(str(org_symbol), infinite=True)
    elif new_symbol_type == "complex":
        return sympy.Symbol(str(org_symbol), complex=True)
    else:
        raise ValueError(f"Unknown symbol type: {new_symbol_type}")



