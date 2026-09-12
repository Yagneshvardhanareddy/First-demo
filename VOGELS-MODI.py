import numpy as np

def print_vam_table(costs, alloc, supply, demand, active_rows, active_cols, row_pen_dict, col_pen_dict, step):
    rows, cols = costs.shape
    print(f"\n{'='*80}\n VAM ITERATION {step}\n{'='*80}")
    
    header = f"| {'':^6} |"
    for j in range(cols):
        header += f" {'D'+str(j+1):^10} |"
    header += f" {'Supply':^8} | {'Penalty':^8} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for i in range(rows):
        row_str = f"| {'S'+str(i+1):^6} |"
        for j in range(cols):
            if i in active_rows and j in active_cols:
                row_str += f" {costs[i, j]:^10} |"
            else:
                if alloc[i, j] > 0:
                    row_str += f" {'['+str(int(alloc[i,j]))+']':^10} |"
                else:
                    row_str += f" {'-':^10} |"
        
        sup_val = str(int(supply[i])) if i in active_rows else "-"
        pen_val = str(row_pen_dict.get(i, "-"))
        row_str += f" {sup_val:^8} | {pen_val:^8} |"
        print(row_str)
    
    print("-" * len(header))
    dem_str = f"| {'Demand':^6} |"
    for j in range(cols):
        dem_val = str(int(demand[j])) if j in active_cols else "-"
        dem_str += f" {dem_val:^10} |"
    dem_str += f" {'':^8} | {'':^8} |"
    print(dem_str)
    
    pen_str = f"| {'Penalty':^6} |"
    for j in range(cols):
        pen_val = str(col_pen_dict.get(j, "-"))
        pen_str += f" {pen_val:^10} |"
    pen_str += f" {'':^8} | {'':^8} |"
    print(pen_str)
    print("-" * len(header))

def vam(costs, supply, demand):
    costs = np.array(costs, dtype=float)
    supply = np.array(supply, dtype=float)
    demand = np.array(demand, dtype=float)
    rows, cols = costs.shape
    alloc = np.zeros((rows, cols))
    
    active_rows = list(range(rows))
    active_cols = list(range(cols))
    step = 1
    
    while active_rows and active_cols:
        row_pen_dict = {}
        for r in active_rows:
            r_costs = sorted([costs[r, c] for c in active_cols])
            pen = r_costs[1] - r_costs[0] if len(r_costs) > 1 else r_costs[0]
            row_pen_dict[r] = pen
            
        col_pen_dict = {}
        for c in active_cols:
            c_costs = sorted([costs[r, c] for r in active_rows])
            pen = c_costs[1] - c_costs[0] if len(c_costs) > 1 else c_costs[0]
            col_pen_dict[c] = pen
            
        print_vam_table(costs, alloc, supply, demand, active_rows, active_cols, row_pen_dict, col_pen_dict, step)
            
        max_row_pen = max(row_pen_dict.values()) if row_pen_dict else -1
        max_col_pen = max(col_pen_dict.values()) if col_pen_dict else -1
        
        if max_row_pen >= max_col_pen:
            best_r = max(row_pen_dict, key=row_pen_dict.get)
            best_c = min(active_cols, key=lambda c: costs[best_r, c])
        else:
            best_c = max(col_pen_dict, key=col_pen_dict.get)
            best_r = min(active_rows, key=lambda r: costs[r, best_c])
            
        qty = min(supply[best_r], demand[best_c])
        alloc[best_r, best_c] = qty
        supply[best_r] -= qty
        demand[best_c] -= qty
        
        print(f"\n-> Allocated {qty} units to (S{best_r+1}, D{best_c+1}) at cost {costs[best_r, best_c]}")
        
        if supply[best_r] == 0 and demand[best_c] == 0:
            if len(active_rows) > 1:
                active_rows.remove(best_r)
            elif active_cols:
                active_cols.remove(best_c)
        elif supply[best_r] == 0:
            active_rows.remove(best_r)
        else:
            active_cols.remove(best_c)
            
        step += 1
        
    return alloc

def get_uv(costs, basic_cells, rows, cols):
    u = {i: None for i in range(rows)}
    v = {j: None for j in range(cols)}
    u[basic_cells[0][0]] = 0
    
    changed = True
    while changed and (None in u.values() or None in v.values()):
        changed = False
        for i, j in basic_cells:
            if u[i] is not None and v[j] is None:
                v[j] = costs[i, j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = costs[i, j] - v[j]
                changed = True
                
    for i in range(rows):
        if u[i] is None: u[i] = 0
    for j in range(cols):
        if v[j] is None: v[j] = 0
        
    return u, v

def find_loop(start, basic_cells):
    nodes = basic_cells + [start]
    
    def dfs(curr, is_horizontal, path):
        if len(path) >= 4 and path[-1] == start:
            return path[:-1]
            
        for r, c in nodes:
            if (r, c) not in path or (r, c) == start:
                if (r, c) == start and len(path) < 4:
                    continue
                if is_horizontal and r == curr[0] and c != curr[1]:
                    res = dfs((r, c), not is_horizontal, path + [(r, c)])
                    if res: return res
                elif not is_horizontal and c == curr[1] and r != curr[0]:
                    res = dfs((r, c), not is_horizontal, path + [(r, c)])
                    if res: return res
        return None

    loop = dfs(start, True, [start])
    if loop is None:
        loop = dfs(start, False, [start])
    return loop

def print_modi_table(costs, alloc, u, v, deltas, basic_cells, step):
    rows, cols = costs.shape
    print(f"\n{'='*80}\n MODI ITERATION {step}\n{'='*80}")
    
    header = f"| {'':^6} |"
    for j in range(cols):
        header += f" {'D'+str(j+1):^14} |"
    header += f" {'u(i)':^8} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for i in range(rows):
        row_str = f"| {'S'+str(i+1):^6} |"
        for j in range(cols):
            if (i, j) in basic_cells:
                cell_str = f"[{int(alloc[i,j])}]"
            else:
                cell_str = f"Δ={deltas.get((i,j), 0):.0f}"
            
            combined = f"c:{int(costs[i,j])} {cell_str}"
            row_str += f" {combined:^14} |"
            
        row_str += f" {u[i]:^8.0f} |"
        print(row_str)
    
    print("-" * len(header))
    v_str = f"| {'v(j)':^6} |"
    for j in range(cols):
        v_str += f" {v[j]:^14.0f} |"
    v_str += f" {'':^8} |"
    print(v_str)
    print("-" * len(header))

def modi(costs, alloc):
    costs = np.array(costs, dtype=float)
    rows, cols = alloc.shape
    step = 1
    
    while True:
        basic_cells = [(i, j) for i in range(rows) for j in range(cols) if alloc[i, j] > 0]
        
        while len(basic_cells) < rows + cols - 1:
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in basic_cells:
                        basic_cells.append((i, j))
                        alloc[i, j] = 1e-9
                        break
                if len(basic_cells) == rows + cols - 1:
                    break
                    
        u, v = get_uv(costs, basic_cells, rows, cols)
        
        min_delta = 0
        enter_cell = None
        deltas = {}
        
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in basic_cells:
                    delta = costs[i, j] - (u[i] + v[j])
                    deltas[(i, j)] = delta
                    if delta < min_delta - 1e-9:
                        min_delta = delta
                        enter_cell = (i, j)
                        
        print_modi_table(costs, alloc, u, v, deltas, basic_cells, step)
                        
        if enter_cell is None:
            print("\n-> All Δ >= 0. Optimal solution reached.")
            break
            
        print(f"\n-> Most negative Δ is {min_delta:.0f} at cell (S{enter_cell[0]+1}, D{enter_cell[1]+1}).")
            
        loop = find_loop(enter_cell, basic_cells)
        minus_cells = loop[1::2]
        
        theta = min(alloc[r, c] for r, c in minus_cells)
        
        loop_str = " -> ".join([f"(S{r+1}, D{c+1})" for r, c in loop])
        print(f"-> Found loop: {loop_str}")
        print(f"-> Theta value to shift: {theta:.0f}")
        
        for idx, (r, c) in enumerate(loop):
            if idx % 2 == 0:
                alloc[r, c] += theta
            else:
                alloc[r, c] -= theta
                
        for i in range(rows):
            for j in range(cols):
                if alloc[i, j] < 1e-8:
                    alloc[i, j] = 0
                    
        step += 1
                    
    return alloc

if __name__ == "__main__":
    costs = [
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40, 8,  70, 20]
    ]
    supply = [7, 9, 18]
    demand = [5, 8, 7, 14]

    print("\nCalculating Initial Basic Feasible Solution using VAM...")
    initial_alloc = vam(costs, supply.copy(), demand.copy())
    
    print("\nOptimizing Initial Solution using MODI (u-v) Method...")
    optimal_alloc = modi(costs, initial_alloc)
    
    rows, cols = optimal_alloc.shape
    total_cost = sum(optimal_alloc[i, j] * costs[i][j] for i in range(rows) for j in range(cols))
    
    print(f"\n{'='*80}")
    print(f"FINAL MINIMUM TRANSPORTATION COST: {total_cost:.2f}")
    print(f"{'='*80}")