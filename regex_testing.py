import re

# Function for checking validity
def is_valid_title(title):
    # title_regex = r"(Sales Rep | Outside Sales Representative | Sale Delivery Representatives | Salesperson | Sales Professional)"
    # title_regex = r"[a-zA-Z]"
    title_regex = r"^(Sales(?: Rep(?:resentative)?|person| Professional)?|Outside Sales Representative|Sale Delivery Representatives)$"
    return re.match(title_regex, title) is not None

# List of job title to test
job_title_test = [
    "Sales Rep",
    "Outside Sales Representative",
    "Sale Delivery Representatives",
    "Salesperson",
    "Sales Professional"
]

# Test each job title and print the result
for job_title in job_title_test:
    if is_valid_title(job_title):
        print(f"{job_title} passed the regex test.")
    else:
        print(f"{job_title} didn't pass the regex test")


# Sure! Let's break down the regex pattern used in `title_regex`:

# ```regex
# ^(Sales(?: Rep(?:resentative)?|person| Professional)?|Outside Sales Representative|Sale Delivery Representatives)$
# ```

# ### Breakdown of the Pattern:

# 1. **`^`**: 
#    - This asserts the start of the string. It ensures that the match must begin at the start of the input string.

# 2. **`( ... )`**: 
#    - Parentheses are used to group parts of the regex together. This allows us to apply quantifiers or alternation to the entire group.

# 3. **`Sales`**: 
#    - This matches the literal string "Sales".

# 4. **`(?: ... )`**: 
#    - This is a non-capturing group. It groups the enclosed patterns without creating a backreference. This is useful for applying quantifiers or alternation without capturing the matched text.

# 5. **`Rep(?:resentative)?`**: 
#    - This matches "Rep" followed optionally by "resentative". 
#    - The `?` after `(?: ... )` means that the preceding element (in this case, "resentative") can appear 0 or 1 time. So, it matches both "Rep" and "Representative".

# 6. **`|`**: 
#    - This is the alternation operator, which works like a logical OR. It allows for matching one of several patterns.

# 7. **`person`**: 
#    - This matches the literal string "person". It allows for the title "Salesperson".

# 8. **`|Professional`**: 
#    - This matches the literal string "Professional". It allows for the title "Sales Professional".

# 9. **`?`**: 
#    - This applies to the entire non-capturing group `(?: Rep(?:resentative)?|person| Professional)?`, meaning that the entire group can appear 0 or 1 time. This allows for titles that do not include "Rep", "person", or "Professional".

# 10. **`Outside Sales Representative`**: 
#     - This matches the exact phrase "Outside Sales Representative". It is a complete title that is included in the alternation.

# 11. **`|Sale Delivery Representatives`**: 
#     - This matches the exact phrase "Sale Delivery Representatives". It is another complete title included in the alternation.

# 12. **`$`**: 
#     - This asserts the end of the string. It ensures that the match must end at the end of the input string.

# ### Summary of Matches:
# - **"Sales Rep"**: Matches "Sales" followed by "Rep".
# - **"Outside Sales Representative"**: Matches the full phrase.
# - **"Sale Delivery Representatives"**: Matches the full phrase.
# - **"Salesperson"**: Matches "Sales" followed by "person".
# - **"Sales Professional"**: Matches "Sales" followed by "Professional".

# This regex pattern effectively captures all the specified job titles while allowing for variations in wording.


