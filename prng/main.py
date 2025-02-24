from utils import *

digits_pi = read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.pi')
digits_e = read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.e')
digits_sqrt2 = read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.sqrt2')

a, b = run_test_for_generator("LCG", seed=1)
print(a)
print(b)

a, b = run_test_for_generator("GLCG", seed=[1, 1, 1])
print(a)
print(b)

a, b = run_test_for_generator("RC4", seed=1)
print(a)
print(b)

a, b = run_test_for_generator("MT", seed=1)
print(a)
print(b)


datasets = {
        "pi": digits_pi,
        "e": digits_e,
        "sqrt2": digits_sqrt2
    }


results_df, second_level_df = perform_tests(datasets, subset_count=100)
print(results_df)
print(second_level_df)
