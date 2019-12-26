from flask import Flask, render_template, request, redirect, Response
from sympy import *
from helpers import apology
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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
            return gif_apology("must provide a function", 400)
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
        lam_f = lambdify(x, f, 'numpy')

        # Calculate max/min and store values
        fprime = f.diff(x)
        extremaX = solve(fprime, x)
        extremaY = []
        candidatesX = [lb, ub]
        candidatesY = [lam_f(lb), lam_f(ub)]
        for extrema in extremaX:
            extremaY.append(lam_f(extrema))
            if (lb < extrema < ub):
                candidatesX.append(extrema)
                candidatesY.append(lam_f(extrema))

        max_input = candidatesX[np.argmax(candidatesY)]
        min_input = candidatesX[np.argmin(candidatesY)]

        max_output = max(candidatesY)
        min_output = min(candidatesY)

        # Plot and save figure
        dist = ub-lb
        X = np.linspace(lb, ub, (dist)*100)
        plt.style.use('seaborn-whitegrid')
        plt.plot(X, lam_f(X))
        for i in range(2, len(candidatesX)):
            print(candidatesX[i])
            epsilon = dist/6
            templb = int(candidatesX[i] - epsilon)
            tempub = int(candidatesX[i] + epsilon)
            tempX = np.linspace(templb, tempub, epsilon*100)
            line = np.ones(tempX.shape)
            line = line * candidatesY[i]
            plt.plot(tempX, line)
        plt.plot(max_input, max_output, c='r', marker='o', label='Maximum')
        plt.plot(min_input, min_output, c='b', marker='o', label='Minimum')
        plt.legend()
        plt.savefig('static/img/maxmin_plot.png')
        plt.close()

        # Turn all into latex
        f = latex(f)
        fprime = latex(fprime)
        return render_template("optimized.html", ub=ub, lb=lb, max_input=max_input,
                                max_output=max_output, min_input=min_input, min_output=min_output,
                               url='static/img/maxmin_plot.png', candidatesX=candidatesX,
                               candidatesY=candidatesY, f=f, fprime=fprime, extremaX=extremaX)
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

        # Create and Save Plot
        dist = abs(a - h)
        X = np.linspace(a-dist-100, a + dist+100, (dist)*1000)

        # Lambdify so we can apply the function to a linspace
        lam_f = lambdify(x, f, 'numpy')
        lam_fprime = lambdify(x, fprime, 'numpy')

        X = np.linspace(a-dist-15, a + dist+15, (dist)*1000)

        tan_line = fprimea * (X - a) + fa
        lam_tan_line = lambdify(x, tan_line, "numpy")

        plt.style.use('seaborn-whitegrid')
        plt.plot(X, lam_f(X), label='Original f(x)')
        plt.plot(X, lam_tan_line(X), label='Tangent Line')
        plt.plot(a, fa, c='r', marker='o', label='Easy Point a')
        plt.plot(h, lh, c='b', marker='o', label='Approximation of f(h)')
        plt.legend()
        plt.savefig('static/img/aprox_plot.png')
        plt.close()

        # Convert to latex for MathJax reading
        value = latex(f)
        fprime = latex(fprime)
        fa = latex(fa)
        lh = latex(lh)

        return render_template("aproxd.html", value=value, fprime=fprime, a=a, h=h, fa=fa, fprimea=fprimea,
        lh=lh, url='static/img/aprox_plot.png')

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

def gif_apology(message="oopsy", code=400):
    "Render gif failure"
    PATH = 'static/img/gifs/reaction' + str(2) + '.gif'
    return render_template("gif_apology.html", PATH=PATH)


if __name__ == '__main__':
    app.run()
