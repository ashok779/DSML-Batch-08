# Calculate tax based on income in Nepal's tax bracket
def calculate_tax(income):
    if income <= 500000:
        tax = income * 0.001
    elif income <= 700000:
        tax = (100000 * 0.01) + (income - 500000) * 0.10
    elif income <= 2000000:
        tax = (100000 * 0.01) + (200000 * 0.10) + (income - 700000) * 0.20
    else:
        tax = (100000 * 0.01) + (200000 * 0.10) + (1300000 * 0.20) + (income - 2000000) * 0.30
    return tax