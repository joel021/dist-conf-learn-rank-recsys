
def ratings_of_user(b_u:float, b_c_i:float) -> float:

    """
    b_u: (overall_user_mean) over all item mean of the target user
    b_c_i: (group_mean) item mean of the group of target user
    """
    y = 0.5
    return y * b_u + (1 - y) * b_c_i
