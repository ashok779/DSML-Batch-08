## 🟢 Beginner Level (Basics of try–except)

### 1️⃣ Handle ZeroDivisionError

Write a program that takes two numbers from the user and prints their division.
Handle the error when the denominator is zero.

**Expected concepts:**
`try`, `except ZeroDivisionError`

---

### 2️⃣ Handle ValueError

Ask the user to enter an integer.
If the user enters a non-integer value, print a friendly error message.

**Expected concepts:**
`ValueError`

---

### 3️⃣ Multiple except blocks

Write a program that:

* Takes two inputs
* Converts them to integers
* Divides them

Handle:

* `ValueError`
* `ZeroDivisionError`

---

### 4️⃣ Using else block

Write a program that divides two numbers.
Print `"Division successful"` only if **no exception occurs**.

**Expected concepts:**
`try–except–else`

---

## 🟡 Intermediate Level (finally, custom messages)

### 5️⃣ Using finally

Write a program that opens a file named `data.txt` and reads its content.
Ensure a message `"Program execution completed"` is printed **no matter what**.

**Expected concepts:**
`finally`

---

### 6️⃣ Custom error messages

Modify a division program so that:

* Zero division prints `"Cannot divide by zero"`
* Invalid input prints `"Please enter valid numbers"`

---

### 7️⃣ Function-level exception handling

Create a function `safe_divide(a, b)` that:

* Returns division result if valid
* Returns `None` if an exception occurs
* Prints the error message

---

### 8️⃣ Catching multiple exceptions in one block

Write a program that catches both `ValueError` and `TypeError` in a single `except` block.

---

## 🔵 Intermediate–Advanced (raise statement)

### 9️⃣ Manual validation using raise

Write a function `check_age(age)` that:

* Raises `ValueError` if age is negative
* Prints `"Valid age"` otherwise

---

### 🔟 Raising exceptions in functions

Create a function `withdraw(balance, amount)`:

* Raise `ValueError` if `amount` is greater than `balance`
* Return remaining balance otherwise

---

### 1️⃣1️⃣ Re-raising exceptions

Write a program where:

* An exception is caught
* Logged using `print`
* Re-raised using `raise`

---

### 1️⃣2️⃣ Raise TypeError manually

Write a function `add_numbers(a, b)`:

* Raise `TypeError` if inputs are not integers
* Return sum otherwise

---

## 🔴 Advanced Level (Custom Exceptions)

### 1️⃣3️⃣ Create a custom exception

Create a custom exception class `InvalidPasswordError`.
Raise it if the password length is less than 8 characters.

---

### 1️⃣4️⃣ Custom exception in real scenario

Create a program for **login validation**:

* Raise `InvalidUserError` if username is incorrect
* Raise `InvalidPasswordError` if password is incorrect

---

### 1️⃣5️⃣ Exception chaining

Write a program that:

* Catches a `ValueError`
* Raises a `RuntimeError` using `raise ... from ...`

---

### 1️⃣6️⃣ Using assertions

Write a function that checks if a number is positive using `assert`.
Handle the `AssertionError`.

---

## 🔥 Expert-Level Challenge

### 1️⃣7️⃣ Robust input system

Build a function `get_valid_integer()` that:

* Keeps asking user input until a valid integer is entered
* Uses exception handling (no if-else)

---

### 1️⃣8️⃣ Banking system simulation

Create a mini banking system:

* Custom exceptions:

  * `InsufficientBalanceError`
  * `InvalidAmountError`
* Handle deposits and withdrawals
* Ensure program never crashes

---

## 🎯 Bonus Interview Question

> What is the difference between:

```python
raise
raise ValueError
raise ValueError("Invalid input")
```

