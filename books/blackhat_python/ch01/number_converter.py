
def sum(number_one, number_two):
    number_one_int = convert_integer(number_one)
    number_two_int = convert_integer(number_two)

    result = number_one_int + number_two_int

    return result


def convert_integer(number_string):

    converted_integer = int(number_string)
    return converted_integer


def main():
    """ Entry point to the application"""
    answer = sum("1", "2")

    print(answer)


if __name__ == '__main__':
    main()
    
