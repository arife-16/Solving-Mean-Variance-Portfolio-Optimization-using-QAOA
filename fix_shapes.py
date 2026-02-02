"""
Fix script for returns shape issues in formulations
Run this once to fix all files
"""

import os

# Fix 1: formulations.py - energy_mad_bitstring
print("Fixing formulations.py...")
file_path = "quant_portfolio/formulations.py"

with open(file_path, 'r') as f:
    content = f.read()

# Replace returns.T @ x with returns @ x
content = content.replace("rp = (returns.T @ x) / k", "rp = (returns @ x) / k")

with open(file_path, 'w') as f:
    f.write(content)

print("✓ Fixed formulations.py")

# Fix 2: formulations_extended.py - energy_cvar_bitstring
print("\nFixing formulations_extended.py...")
file_path = "quant_portfolio/formulations_extended.py"

with open(file_path, 'r') as f:
    content = f.read()

# Replace returns.T @ x with returns @ x
content = content.replace("portfolio_returns = (returns.T @ x) / k", 
                          "portfolio_returns = (returns @ x) / k")

with open(file_path, 'w') as f:
    f.write(content)

print("✓ Fixed formulations_extended.py")

# Fix 3: data.py - transpose synthetic returns
print("\nFixing data.py...")
file_path = "quant_portfolio/data.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the return statement in generate_synthetic_returns
new_lines = []
for i, line in enumerate(lines):
    if 'def generate_synthetic_returns' in line:
        # Mark that we're in this function
        in_function = True
    
    if 'return returns' in line and 'generate_synthetic_returns' in ''.join(lines[max(0,i-20):i]):
        # Add .T if not already there
        if '.T' not in line:
            line = line.replace('return returns', 'return returns.T')
            print(f"  Fixed line {i+1}: {line.strip()}")
    
    new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("✓ Fixed data.py")

print("\n" + "="*60)
print("All fixes applied!")
print("="*60)
print("\nNow:")
print("1. Restart your Jupyter kernel")
print("2. Re-run the import cells")
print("3. Try the examples again")
print("\nAll formulations should now work with both synthetic and real data!")