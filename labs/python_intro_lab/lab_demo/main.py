import sys
# This is a relative import. It imports the 'greet' function from 'my_module.py'
# which is in the same directory.
from my_module import greet

def main():
    # sys.argv allows us to access command line arguments
    # sys.argv[0] is the script name itself
    if len(sys.argv) > 1:
        user_name = sys.argv[1]
    else:
        user_name = "Future Engineer"

    message = greet(user_name)
    print(message)
    print("\nIf you are seeing this, you have successfully run a Python script locally!")

# This block ensures that main() runs only when this script is executed directly,
# not when it is imported by another script.
if __name__ == "__main__":
    main()

