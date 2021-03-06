import requests
from flask import redirect, render_template, request, session
from functools import wraps
from sympy import *
import random

x, y, z, t, X, Y, Z, T = symbols('x y z t X Y Z T')


def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message)), code


def login_required(f):
    """
    Decorate routes to require login.

    http://flask.pocoo.org/docs/1.0/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function


def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"

def setup_symbols(f):
    f = sympify(f)

    # replace commonly used variables with x
    for letter in [x, y, z, t, X, Y, Z, T]:
        f = f.subs(letter, x)
    return f

def gif_apology(message="oopsy", code=400):
    "Render gif failure"
    PATH = 'static/img/gifs/reaction' + str(random.randint(1,5)) + '.gif'
    return render_template("gif_apology.html", PATH=PATH)
