from dateutil.parser import parse
from Levenshtein import distance # pip install python-Levenshtein

def closest_string(input_string, list1, list2 = None):
    # Initialize minimum distance to a large number and closest string to None
    min_dist = float('inf')
    closest_str = None

    # Check each string in the first list
    for string in list1:
        dist = distance(input_string, string)
        if dist < min_dist:
            min_dist = dist
            closest_str = string
    if list2 != None:
    # Check each string in the second list
        for string in list2:
            dist = distance(input_string, string)
            if dist < min_dist:
                min_dist = dist
                closest_str = string
    return closest_str

def preprocess_input_json(input_json: str):
    dict_keys_temp = [i for i in list(input_json.keys()) if 'year' not in i]

    for i in dict_keys_temp:
        temp_val = input_json[i]
        if temp_val.isnumeric():
            input_json[i] = int(temp_val)
        else:
            current_key = i[:-1]
            if current_key == 'month':
                closest_str = closest_string(temp_val, months)
                ind = months_s2i[closest_str]
                input_json[i] = ind
            else:
                closest_str = closest_string(temp_val, ordinal_numbers, numbers)
                ind = nums_s2i[closest_str]
                input_json[i] = ind

# Main function to convert natural language dates to machine-readable format
def convert_dates(input_json):
    # current_year = str(datetime.now().year)
    output_json = {}

    for i in range(1, 3):
        year_key = f"year{i}"
        month_key = f"month{i}"
        day_key = f"day{i}"
        date_key = f"date{i}"

        year = input_json[year_key]
        month = input_json[month_key]
        day = input_json[day_key]

        # Construct the date string
        date_str = f"{year}-{month}-{day}"
        # Parse the date string to ensure it's valid
        try:
            parsed_date = parse(date_str)
            output_json[date_key] = parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            output_json[date_key] = "Invalid date"

    return output_json
    # return json.dumps(output_json, ensure_ascii=False, indent=2)

months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']

ordinal_numbers = ['первый', 'второй', 'третий', 'четвертый', 'пятый', 'шестой', 'седьмой', 'восьмой', 'девятый', 'десятый',
           'одиннадцатый', 'двенадцатый', 'тринадцатый', 'четырнадцатый', 'пятнадцатый', 'шестнадцатый', 'семнадцатый',
           'восемнадцатый', 'девятнадцатый', 'двадцатый', 'двадцать первый', 'двадцать второй', 'двадцать третий',
           'двадцать четвертый', 'двадцать пятый', 'двадцать шестой', 'двадцать седьмой', 'двадцать восьмой',
           'двадцать девятый', 'тридцатый', 'тридцать первый']

numbers = ['один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять',
           'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать', 'пятнадцать', 'шестнадцать', 'семнадцать',
           'восемнадцать', 'девятнадцать', 'двадцать', 'двадцать один', 'двадцать два', 'двадцать три',
           'двадцать четыре', 'двадцать пять', 'двадцать шесть', 'двадцать семь', 'двадцать восемь',
           'двадцать девять', 'тридцать', 'тридцать один']

def create_element2id(input_list):
    return {val:ind for ind, val in enumerate(input_list, start=1)}

months_s2i = create_element2id(months)
ord_nums_s2i = create_element2id(ordinal_numbers)
nums_s2i = create_element2id(numbers)

nums_s2i.update(ord_nums_s2i)

# # Example usage
# input_json = {
#   "year1": 2021,
#   "month1": "сеньбря ",
#   "day1": "двеятнадцтое",
#   "year2": 2022,
#   "month2": "прпеля",
#   "day2": "шшестого"
# }

# preprocess_input_json(input_json)


