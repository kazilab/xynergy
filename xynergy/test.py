def calc_explained_var(x, y):
    """Calculate the variance of x explained by y"""

    rss = sum(sum((x - y) ** 2))
    tss = sum(sum(x**2))

    return 1 - rss / tss
