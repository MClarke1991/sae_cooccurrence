from math import log

def expected_time(n_features, n_acts, batch_size=4096):
    # https://math.stackexchange.com/questions/151267/coupon-collector-problem-for-collecting-set-k-times
    # https://mathoverflow.net/questions/229060/batched-coupon-collector-problem 
    n = n_features
    m = n_acts
    if n <= 0:
        raise ValueError("n must be a positive number")
    
    term1 = n * log(n)
    term2 = (m - 1) * n * log(log(n))
    
    return (term1 + term2) / (7 * batch_size)