#EXAMPLE 12: palindrome program (Same word forwards and backwords, Example: racecar)
#Test if given string is a palindrome
string = input("Enter a word: ")
rev_string = ""
for char in string:
    # prepend each character to beginning of new variable
    # for prepend, the order is very important. it must be char+rev_string, if we switch the position, then it wil not be prepend.
    rev_string = rev_string + char
# Checking if palindrome (same forward and backwords)
if string == rev_string:
    print("Yes", string, "is a palindrome. ")
else:
    print("No", string, "is not a palindrome. ")
