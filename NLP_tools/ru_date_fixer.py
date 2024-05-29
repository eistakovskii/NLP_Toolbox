from dateutil.parser import parse
import copy
from typing import Dict, List
from Levenshtein import distance # pip install python-Levenshtein

class DateParser():
    def __init__(self):

        """
        intiate the reference "database" with info about month numbers, day numbers, ordinal (e.g. first, second) 
        and regular (e.g. one, two, three). We also create a lookup dictionaries, i.e. day/month to number. For the
        month numbers there is a separate lookup (with values from 1 to 12) while for days of the month there is a merged
        lookup comprising two dictionaries: one for regular numbers, one for ordinal numbers - this way we may increase a
        potentially correct match when using Levenshtein distance
        """

        # initiate base lists/references
        self.months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
        self.ordinal_numbers = ['первый', 'второй', 'третий', 'четвертый', 'пятый', 'шестой', 'седьмой', 'восьмой', 'девятый', 'десятый',
           'одиннадцатый', 'двенадцатый', 'тринадцатый', 'четырнадцатый', 'пятнадцатый', 'шестнадцатый', 'семнадцатый',
           'восемнадцатый', 'девятнадцатый', 'двадцатый', 'двадцать первый', 'двадцать второй', 'двадцать третий',
           'двадцать четвертый', 'двадцать пятый', 'двадцать шестой', 'двадцать седьмой', 'двадцать восьмой',
           'двадцать девятый', 'тридцатый', 'тридцать первый']
        self.numbers = ['один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять',
           'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать', 'пятнадцать', 'шестнадцать', 'семнадцать',
           'восемнадцать', 'девятнадцать', 'двадцать', 'двадцать один', 'двадцать два', 'двадцать три',
           'двадцать четыре', 'двадцать пять', 'двадцать шесть', 'двадцать семь', 'двадцать восемь',
           'двадцать девять', 'тридцать', 'тридцать один']
        
        # create lookup dictionaries from the lists
        self.months_s2i = self.__create_element2id(self.months)
        self.ord_nums_s2i = self.__create_element2id(self.ordinal_numbers)
        self.nums_s2i = self.__create_element2id(self.numbers)
        # this is the main lookup for numbers
        self.nums_s2i.update(self.ord_nums_s2i) # we update the lookup with the same numbers (now we have ordinal numbers and regular) 
        # to increase the right match with levenshtein

    def __create_element2id(self, input_list: List[str]) -> Dict[str, int]:
        return {val:ind for ind, val in enumerate(input_list, start=1)}
    
    def closest_string(self, input_string: str, list1: List[str], list2:List[str] = None) -> str:
        """
        find a closest string from two lists of strings to an input string using Levenshtein distance
        """

        # initialize minimum distance to a large number and closest string to None
        min_dist = float('inf')
        closest_str = None

        # check each string in the first list
        for string in list1:
            dist = distance(input_string, string)
            if dist < min_dist:
                min_dist = dist
                closest_str = string
        if list2 != None:
        # check each string in the second list if applicable
            for string in list2:
                dist = distance(input_string, string)
                if dist < min_dist:
                    min_dist = dist
                    closest_str = string
        return closest_str
        
    def preprocess_input_json(self, input_json: str) -> Dict[str, str]:

        """
        fix the input dates in natural language and convert them to digits, e.g. "первоей мая 2023" -> "2023-05-01"
        """
        
        input_json_out = copy.copy(input_json)

        dict_keys_temp = [i for i in list(input_json_out.keys()) if 'year' not in i]

        for i in dict_keys_temp:
            temp_val = input_json_out[i]
            if temp_val.isnumeric():
                input_json_out[i] = int(temp_val)
            else:
                current_key = i[:-1]
                if current_key == 'month':
                    closest_str = self.closest_string(temp_val, self.months)
                    ind = self.months_s2i[closest_str]
                    input_json_out[i] = ind
                else:
                    closest_str = self.closest_string(temp_val, self.ordinal_numbers, self.numbers)
                    ind = self.nums_s2i[closest_str]
                    input_json_out[i] = ind
        
        return input_json_out

    def convert_dates(self, input_json: Dict[str, str]) -> Dict[str, str]:

        """
        convert the input json with tho dates in natural language into machine readable date

        expected input structure:

        {
            "year1": 2021,
            "month1": "сеньбря ",
            "day1": "двеятнадцтое",
            "year2": 2022,
            "month2": "прпеля",
            "day2": "шшестого"
            }

        output:

        {
            'date1': '2021-09-12', 
            'date2': '2022-04-06'
            }


        """
        
        input_json = self.preprocess_input_json(input_json) # translate anything in natural language into numerical format
        
        output_json = {}

        for i in range(1, 3):
            year_key = f"year{i}"
            month_key = f"month{i}"
            day_key = f"day{i}"
            date_key = f"date{i}"

            year = input_json[year_key]
            month = input_json[month_key]
            day = input_json[day_key]

            # construct the date string
            date_str = f"{year}-{month}-{day}"
            # parse the date string to ensure it's valid
            try:
                parsed_date = parse(date_str)
                output_json[date_key] = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                output_json[date_key] = "Invalid date"

        return output_json

# # Example usage
# input_json = {
#   "year1": 2021,
#   "month1": "сеньбря ",
#   "day1": "двеятнадцтое",
#   "year2": 2022,
#   "month2": "прпеля",
#   "day2": "шшестого"
# }

# parser = DateParser()

# print(parser.convert_dates(input_json))



