# A simple program to check if the given number is even or odd

n = int(input("Enter a number: ")) # Takes the number as input
if (n%2 == 0):
    print(f"{n} is an even number") # This if block checks for the number being divisible by 2 to test if its even or odd and prints if even
else:
    print(f"{n} is an odd number") # Else block will print if the number is odd
