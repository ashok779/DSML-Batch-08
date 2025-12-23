# Calculate monthly emi given principal, annual interest rate, and tenure in years
def calculate_emi(principal, annual_rate, tenure_years = 10):
    monthly_rate = annual_rate / (12 * 100)  # Convert annual rate to monthly and percentage to decimal
    tenure_months = tenure_years * 12

    if monthly_rate == 0:  # If interest rate is 0%
        emi = principal / tenure_months
    else:
        emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
    
    return emi
