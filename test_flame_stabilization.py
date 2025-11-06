def initial_horizontal_line(Y, y_flame):
    """
    Determines the horizontal line based on input Y and y_flame.

    G > 0: unburnt region (below the line)
    G < 0: burnt region (above the line)
    """
    G = -(Y - y_flame)  # Updated line
    return G
