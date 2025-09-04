# ## Lets make soe simple python funcitons
# # Simplest function is just like this:
# def hello_world():
#     print("Hello, World!")

# # Lets call the function
# hello_world()

# # Function can take parameters:
# def add(a, b):
#     return a + b

# # Lets call the function
# print(add(1, 2))

##
# My dictionary
# my_dict = {
#     "name": "John",
#     "age": 30,
#     "city": "New York"
# }

# def change_dict(my_dict):
#     new_dict = my_dict.copy()
#     new_dict["name"] = "Jane"
#     return new_dict

# # Lets call the function
# new_dict = change_dict(my_dict)

# # Lets print the dictionary
# print(my_dict)

# Now lets do a function with a default parameter
# def add(a, b=1):
#     return a + b

# # Lets call the function
# print(add(1))
# print(add(1, 2))

# Use type hints
def addwithhints(a: int, b: int) -> int:
    return a + b

# Lets call the function
print(addwithhints(1.3, 2.34))