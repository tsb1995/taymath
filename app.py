from flask import Flask, render_template, request, redirect
from sympy import *
from helpers import apology

app = Flask(__name__)

# Convert our commonly used variables into sympy symbols
x, y, z, t, X, Y, Z, T = symbols('x y z t X Y Z T')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/differentiation', methods=['GET', 'POST'])
def differentiation():
    if request.method == "POST":
        # Check if inputs were given
        if not request.form.get("function"):
            return apology("must provide a function", 400)
        f = request.form.get("function")

        # Setup our symbols for SymPy
        f = setup_symbols(f)

        # Differentiate and return latex expressions
        fprime = latex(f.diff(x))
        value = latex(f)
        return render_template("differentiated.html", value=value, fprime=fprime)
    else:
        return render_template("differentiation.html")

@app.route("/integration", methods=["GET", "POST"])
def integration():
    if request.method == "POST":
        # Check if inputs were given
        if not request.form.get("function"):
            return apology("must provide a function", 400)
        f = request.form.get("function")

        # Setup our symbols for SymPy
        f = setup_symbols(f)

        # Integrate and return latex expressions
        fintegral = latex(f.integrate(x))
        value = latex(f)
        return render_template("integrated.html", value=value, fintegral=fintegral)
    else:
        return render_template("integration.html")

@app.route("/riemann", methods = ["GET", "POST"])
def riemann():
    if request.method == "POST":
        # Check if inputs were given
        if not request.form.get("function"):
            return apology("must provide a function", 400)
        if not request.form.get("lowerbound"):
            return apology("must provide a lower bound", 400)
        if not request.form.get("upperbound"):
            return apology("must provide an upper bound", 400)
        if not request.form.get("subintervals"):
            return apology("must provide a number of subintervals", 400)
        if not request.form.get("sumtype"):
            return apology("must choose left or right", 400)

        # Get our info from form
        f = request.form.get("function")
        sumtype = request.form.get("sumtype")
        lb = int(request.form.get("lowerbound"))
        ub = int(request.form.get("upperbound"))
        si = int(request.form.get("subintervals"))

        # Setup our symbols for SymPy
        f = setup_symbols(f)

        # Run through Riemann Sum algorithm, creatings lists for display
        #           of inputs, outputs, and areas (their products)

        value = latex(f)
        dx = round((ub - lb) / si, 3)
        inputs = list()
        if sumtype == "1":
            for i in range(0, si):
                inputs.append(round(dx * (i), 3))
        if sumtype == "2":
            for i in range(0, si):
                inputs.append(round(dx * (i + 1), 3))
        outputs = list()
        for input in inputs:
            temp = f.subs(x, input)
            outputs.append(round(temp, 3))
        rectangles = list()
        for output in outputs:
            temp = output * dx
            rectangles.append(round(temp, 3))
        result = round(sum(rectangles), 3)

        # Choose template based on left or right sum
        if sumtype == "1":
            return render_template("summed.html", value=value, sumtype=sumtype, lb=lb, ub=ub, si=si, dx=dx,
                                   inputs=inputs, outputs=outputs, rectangles=rectangles, result=result)
        else:
            return render_template("rightSummed.html", value=value, sumtype=sumtype, lb=lb, ub=ub, si=si, dx=dx,
                                   inputs=inputs, outputs=outputs, rectangles=rectangles, result=result)
    else:
        return render_template("riemann.html")

@app.route("/maxmin", methods=["GET", "POST"])
def maxmin():
    if request.method == "POST":
        # Check if inputs were given
        if not request.form.get("function"):
            return apology("must provide a function", 400)
        if not request.form.get("lowerbound"):
            return apology("must provide a lower bound", 400)
        if not request.form.get("upperbound"):
            return apology("must provide an upper bound", 400)

        # Get input from form
        f = request.form.get("function")
        lb = round(sympify(request.form.get("lowerbound")), 3)
        ub = round(sympify(request.form.get("upperbound")), 3)

        # Setup our symbols for SymPy
        f = setup_symbols(f)

        # Get Derivative, solve for real solutions, update candidates list
        fprime = f.diff(x)
        solutions = list()
        solutions.append(round(f.subs(x,0), 3))
        candidates = list()
        for solution in solutions:
            candidates.append(solution)
        candidates.append(lb)
        candidates.append(ub)

        # Fill values list with solutions
        values = list()
        for candidate in candidates:
            temp = round(f.subs(x, candidate), 3)
            values.append(temp)

        # Find max/min of values
        maximum = max(values)
        newvar = min(values)

        # Turn all into latex
        value = latex(f)
        fprime = latex(fprime)
        for i, solution in enumerate(solutions):
            solutions[i] = latex(solution)
        return render_template("optimized.html", value=value, fprime=fprime, solutions=solutions, lb=lb, ub=ub,
                               candidates=candidates, newvar=newvar, values=values, maximum=maximum)
    else:
        return render_template("maxmin.html")

@app.route("/aprox", methods=["GET", "POST"])
def aprox():
    if request.method == "POST":
        # Check if inputs were given
        if not request.form.get("function"):
            return apology("must provide a function", 400)
        if not request.form.get("easy"):
            return apology("must provide an easy value", 400)
        if not request.form.get("hard"):
            return apology("must provide a difficult value", 400)

        # Get inputs, sympify them, and check to see if valid
        f = request.form.get("function")
        a = request.form.get("easy")
        h = request.form.get("hard")
        # Setup our symbols for SymPy
        f = setup_symbols(f)

        # Make sure a and h are numbers
        a = sympify(a)
        h = sympify(h)
        if not a.is_number:
            return apology("easy value must be a number", 400)
        if not h.is_number:
            return apology("difficult value must be a number", 400)

        a = round(a, 3)
        h = round(h, 3)
        # Run through Linearization algorithm
        fprime = f.diff(x)
        fa = round(f.subs(x, a), 3)
        fprimea = round(fprime.subs(x, a), 3)
        lh = round(fa + fprimea*(float(h)-float(a)), 3)

        # Convert to latex for MathJax reading
        value = latex(f)
        fprime = latex(fprime)
        fa = latex(fa)
        lh = latex(lh)

        return render_template("aproxd.html", value=value, fprime=fprime, a=a, h=h, fa=fa, fprimea=fprimea, lh=lh)
    else:
        return render_template("aprox.html")

def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return apology(e.name, e.code)

def setup_symbols(f):
    f = sympify(f)

    # replace commonly used variables with x
    for letter in [x, y, z, t, X, Y, Z, T]:
        f = f.subs(letter, x)
    return f



if __name__ == '__main__':
    app.run()
