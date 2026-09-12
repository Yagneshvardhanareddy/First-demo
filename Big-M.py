import numpy as np

EPS = 1e-9
M = 1e6

def print_tableau(tableau, column_names, basis, iteration):
    print("\n" + "=" * 90)
    print(f"ITERATION {iteration}")
    print("=" * 90)
    print(f"{'BV':>8}", end="")
    for name in column_names:
        print(f"{name:>12}", end="")
    print()
    for i in range(len(basis)):
        print(f"{basis[i]:>8}", end="")
        for value in tableau[i]:
            print(f"{value:12.3f}", end="")
        print()
    print(f"{'Z':>8}", end="")
    for value in tableau[-1]:
        print(f"{value:12.3f}", end="")
    print()

def convert_objective(objective, problem_type):
    if problem_type.lower() == "min":
        return [-c for c in objective]
    return objective.copy()

def create_initial_tableau(objective, constraints, problem_type):
    num_variables = len(objective)
    num_constraints = len(constraints)
    processed_constraints = []

    for coefficients, sign, rhs in constraints:
        coefficients = np.array(coefficients, dtype=float)
        if rhs < 0:
            coefficients = -coefficients
            rhs = -rhs
            if sign == "<=":
                sign = ">="
            elif sign == ">=":
                sign = "<="
        processed_constraints.append((coefficients, sign, rhs))
        
    constraints = processed_constraints
    variable_names = []

    for i in range(num_variables):
        variable_names.append(f"x{i + 1}")

    slack_count = 0
    surplus_count = 0
    artificial_count = 0

    for _, sign, _ in constraints:
        if sign == "<=":
            slack_count += 1
        elif sign == ">=":
            surplus_count += 1
            artificial_count += 1
        elif sign == "=":
            artificial_count += 1
        else:
            raise ValueError("Constraint sign must be <=, >=, or =")

    for i in range(slack_count):
        variable_names.append(f"s{i + 1}")
    for i in range(surplus_count):
        variable_names.append(f"e{i + 1}")
    for i in range(artificial_count):
        variable_names.append(f"A{i + 1}")

    total_variables = len(variable_names)
    tableau = np.zeros((num_constraints + 1, total_variables + 1))
    basis = []
    artificial_columns = []

    slack_index = num_variables
    surplus_index = num_variables + slack_count
    artificial_index = num_variables + slack_count + surplus_count

    slack_used = 0
    surplus_used = 0
    artificial_used = 0

    for i, (coefficients, sign, rhs) in enumerate(constraints):
        tableau[i, :num_variables] = coefficients

        if sign == "<=":
            col = slack_index + slack_used
            tableau[i, col] = 1
            basis.append(variable_names[col])
            slack_used += 1
        elif sign == ">=":
            surplus_col = surplus_index + surplus_used
            artificial_col = artificial_index + artificial_used
            tableau[i, surplus_col] = -1
            tableau[i, artificial_col] = 1
            basis.append(variable_names[artificial_col])
            artificial_columns.append(artificial_col)
            surplus_used += 1
            artificial_used += 1
        elif sign == "=":
            artificial_col = artificial_index + artificial_used
            tableau[i, artificial_col] = 1
            basis.append(variable_names[artificial_col])
            artificial_columns.append(artificial_col)
            artificial_used += 1

        tableau[i, -1] = rhs

    converted_objective = convert_objective(np.array(objective, dtype=float), problem_type)
    tableau[-1, :num_variables] = -converted_objective

    for col in artificial_columns:
        tableau[-1, col] = M

    for i, basis_variable in enumerate(basis):
        if basis_variable.startswith("A"):
            tableau[-1] -= M * tableau[i]

    return tableau, variable_names, basis, artificial_columns

def find_entering_variable(tableau):
    objective_row = tableau[-1, :-1]
    pivot_col = np.argmin(objective_row)
    if objective_row[pivot_col] >= -EPS:
        return None
    return pivot_col

def find_leaving_variable(tableau, pivot_col):
    ratios = []
    num_constraints = tableau.shape[0] - 1
    
    for i in range(num_constraints):
        element = tableau[i, pivot_col]
        if element > EPS:
            ratio = tableau[i, -1] / element
            if ratio >= -EPS:
                ratios.append((ratio, i))
                
    if not ratios:
        return None
        
    ratios.sort(key=lambda x: x[0])
    return ratios[0][1]

def pivot(tableau, pivot_row, pivot_col):
    pivot_element = tableau[pivot_row, pivot_col]
    if abs(pivot_element) < EPS:
        raise ValueError("Pivot element is zero.")
        
    tableau[pivot_row] /= pivot_element
    
    for i in range(tableau.shape[0]):
        if i != pivot_row:
            factor = tableau[i, pivot_col]
            tableau[i] -= factor * tableau[pivot_row]

def check_infeasible(tableau, variable_names, artificial_columns):
    for col in artificial_columns:
        value = tableau[:, col]
        for row in range(len(tableau) - 1):
            if abs(value[row] - 1) < 1e-7:
                other_values = np.delete(value[:-1], row)
                if np.all(np.abs(other_values) < 1e-7):
                    artificial_value = tableau[row, -1]
                    if artificial_value > EPS:
                        return True
    return False

def extract_solution(tableau, variable_names, basis):
    solution = {}
    for variable in variable_names:
        solution[variable] = 0.0
    for row, basic_variable in enumerate(basis):
        solution[basic_variable] = tableau[row, -1]
    return solution

def solve_big_m(objective, constraints, problem_type="max"):
    print("\n")
    print("=" * 90)
    print("                 BIG-M SIMPLEX METHOD")
    print("=" * 90)

    tableau, variable_names, basis, artificial_columns = create_initial_tableau(
        objective, constraints, problem_type
    )

    iteration = 0
    print("\nInitial Tableau:")
    print_tableau(tableau, variable_names + ["RHS"], basis, iteration)

    while True:
        pivot_col = find_entering_variable(tableau)
        if pivot_col is None:
            print("\nOptimality condition satisfied.")
            break

        pivot_row = find_leaving_variable(tableau, pivot_col)
        if pivot_row is None:
            print("\n" + "=" * 90)
            print("The problem is UNBOUNDED.")
            print("=" * 90)
            return

        entering_variable = variable_names[pivot_col]
        leaving_variable = basis[pivot_row]

        print(f"\nEntering variable : {entering_variable}")
        print(f"Leaving variable  : {leaving_variable}")
        print(f"Pivot element     : {tableau[pivot_row, pivot_col]:.3f}")

        pivot(tableau, pivot_row, pivot_col)
        basis[pivot_row] = entering_variable
        iteration += 1
        
        print_tableau(tableau, variable_names + ["RHS"], basis, iteration)

    if check_infeasible(tableau, variable_names, artificial_columns):
        print("\n" + "=" * 90)
        print("The problem is INFEASIBLE.")
        print("=" * 90)
        return

    solution = extract_solution(tableau, variable_names, basis)
    objective_value = tableau[-1, -1]

    if problem_type.lower() == "min":
        objective_value = -objective_value

    print("\n" + "=" * 90)
    print("                     FINAL SOLUTION")
    print("=" * 90)

    for variable in variable_names:
        print(f"{variable:>5} = {solution[variable]:.6f}")

    print("-" * 90)
    print(f"Optimal Objective Value = {objective_value:.6f}")
    print("=" * 90)

    return solution, objective_value

if __name__ == "__main__":
    objective = [3, 5]
    constraints = [
        ([1, 0], "<=", 4),
        ([0, 2], "<=", 12),
        ([3, 2], ">=", 18)
    ]
    
    solve_big_m(objective, constraints, problem_type="max")